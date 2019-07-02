"""
Geometry container base class.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
from yt.extern.six.moves import cPickle
import weakref
from yt.utilities.on_demand_imports import _h5py as h5py
import numpy as np

from yt.config import ytcfg
from yt.funcs import iterable
from yt.io.io_handler import io_registry
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_root_only
from yt.utilities.exceptions import YTFieldNotFound

class Index(ParallelAnalysisInterface):
    """The base index class"""
    _unsupported_objects = ()
    _index_properties = ()

    def __init__(self, ds, dataset_type):
        ParallelAnalysisInterface.__init__(self)
        self.dataset = weakref.proxy(ds)
        self.ds = self.dataset

        self._initialize_state_variables()

        mylog.debug("Initializing data storage.")
        self._initialize_data_storage()

        mylog.debug("Setting up domain geometry.")
        self._setup_geometry()

        mylog.debug("Initializing data grid data IO")
        self._setup_data_io()

        # Note that this falls under the "geometry" object since it's
        # potentially quite expensive, and should be done with the indexing.
        mylog.debug("Detecting fields.")
        self._detect_output_fields()

    def _initialize_state_variables(self):
        self._parallel_locking = False
        self._data_file = None
        self._data_mode = None
        self.num_grids = None

    def _initialize_data_storage(self):
        if not ytcfg.getboolean('yt','serialize'): return
        fn = self.ds.storage_filename
        if fn is None:
            if os.path.isfile(os.path.join(self.directory,
                                "%s.yt" % self.ds.unique_identifier)):
                fn = os.path.join(self.directory,"%s.yt" % self.ds.unique_identifier)
            else:
                fn = os.path.join(self.directory,
                        "%s.yt" % self.dataset.basename)
        dir_to_check = os.path.dirname(fn)
        if dir_to_check == '':
            dir_to_check = '.'
        # We have four options:
        #    Writeable, does not exist      : create, open as append
        #    Writeable, does exist          : open as append
        #    Not writeable, does not exist  : do not attempt to open
        #    Not writeable, does exist      : open as read-only
        exists = os.path.isfile(fn)
        if not exists:
            writeable = os.access(dir_to_check, os.W_OK)
        else:
            writeable = os.access(fn, os.W_OK)
        writeable = writeable and not ytcfg.getboolean('yt','onlydeserialize')
        # We now have our conditional stuff
        self.comm.barrier()
        if not writeable and not exists: return
        if writeable:
            try:
                if not exists: self.__create_data_file(fn)
                self._data_mode = 'a'
            except IOError:
                self._data_mode = None
                return
        else:
            self._data_mode = 'r'

        self.__data_filename = fn
        self._data_file = h5py.File(fn, self._data_mode)

    def __create_data_file(self, fn):
        # Note that this used to be parallel_root_only; it no longer is,
        # because we have better logic to decide who owns the file.
        f = h5py.File(fn, 'a')
        f.close()

    def _setup_data_io(self):
        if getattr(self, "io", None) is not None: return
        self.io = io_registry[self.dataset_type](self.dataset)

    @parallel_root_only
    def save_data(self, array, node, name, set_attr=None, force=False, passthrough = False):
        """
        Arbitrary numpy data will be saved to the region in the datafile
        described by *node* and *name*.  If data file does not exist, it throws
        no error and simply does not save.
        """

        if self._data_mode != 'a': return
        try:
            node_loc = self._data_file[node]
            if name in node_loc and force:
                mylog.info("Overwriting node %s/%s", node, name)
                del self._data_file[node][name]
            elif name in node_loc and passthrough:
                return
        except:
            pass
        myGroup = self._data_file['/']
        for q in node.split('/'):
            if q: myGroup = myGroup.require_group(q)
        arr = myGroup.create_dataset(name,data=array)
        if set_attr is not None:
            for i, j in set_attr.items(): arr.attrs[i] = j
        self._data_file.flush()

    def _reload_data_file(self, *args, **kwargs):
        if self._data_file is None: return
        self._data_file.close()
        del self._data_file
        self._data_file = h5py.File(self.__data_filename, self._data_mode)

    def save_object(self, obj, name):
        """
        Save an object (*obj*) to the data_file using the Pickle protocol,
        under the name *name* on the node /Objects.
        """
        s = cPickle.dumps(obj, protocol=-1)
        self.save_data(np.array(s, dtype='c'), "/Objects", name, force = True)

    def load_object(self, name):
        """
        Load and return and object from the data_file using the Pickle protocol,
        under the name *name* on the node /Objects.
        """
        obj = self.get_data("/Objects", name)
        if obj is None:
            return
        obj = cPickle.loads(obj.value)
        if iterable(obj) and len(obj) == 2:
            obj = obj[1] # Just the object, not the ds
        if hasattr(obj, '_fix_pickle'): obj._fix_pickle()
        return obj

    def get_data(self, node, name):
        """
        Return the dataset with a given *name* located at *node* in the
        datafile.
        """
        if self._data_file is None:
            return None
        if node[0] != "/": node = "/%s" % node

        myGroup = self._data_file['/']
        for group in node.split('/'):
            if group:
                if group not in myGroup:
                    return None
                myGroup = myGroup[group]
        if name not in myGroup:
            return None

        full_name = "%s/%s" % (node, name)
        try:
            return self._data_file[full_name][:]
        except TypeError:
            return self._data_file[full_name]

    def _get_particle_type_counts(self):
        # this is implemented by subclasses
        raise NotImplementedError

    def _close_data_file(self):
        if self._data_file:
            self._data_file.close()
            del self._data_file
            self._data_file = None

    def _split_fields(self, fields):
        # This will split fields into either generated or read fields
        fields_to_read, fields_to_generate = [], []
        for ftype, fname in fields:
            if fname in self.field_list or (ftype, fname) in self.field_list:
                fields_to_read.append((ftype, fname))
            elif fname in self.ds.derived_field_list or (ftype, fname) in self.ds.derived_field_list:
                fields_to_generate.append((ftype, fname))
            else:
                raise YTFieldNotFound((ftype,fname), self.ds)
        return fields_to_read, fields_to_generate

    def _read_particle_fields(self, fields, dobj, chunk = None):
        if len(fields) == 0: return {}, []
        fields_to_read, fields_to_generate = self._split_fields(fields)
        if len(fields_to_read) == 0:
            return {}, fields_to_generate
        selector = dobj.selector
        if chunk is None:
            self._identify_base_chunk(dobj)
        chunks = self._chunk_io(dobj, cache = False)
        fields_to_return = self.io._read_particle_selection(
            chunks,
            selector,
            fields_to_read)
        return fields_to_return, fields_to_generate

    def _read_fluid_fields(self, fields, dobj, chunk = None):
        if len(fields) == 0: return {}, []
        fields_to_read, fields_to_generate = self._split_fields(fields)
        if len(fields_to_read) == 0:
            return {}, fields_to_generate
        selector = dobj.selector
        if chunk is None:
            self._identify_base_chunk(dobj)
            chunk_size = dobj.size
        else:
            chunk_size = chunk.data_size
        fields_to_return = self.io._read_fluid_selection(
            self._chunk_io(dobj),
            selector,
            fields_to_read,
            chunk_size)
        return fields_to_return, fields_to_generate

    def _chunk(self, dobj, chunking_style, ngz = 0, **kwargs):
        # A chunk is either None or (grids, size)
        if dobj._current_chunk is None:
            self._identify_base_chunk(dobj)
        if ngz != 0 and chunking_style != "spatial":
            raise NotImplementedError
        if chunking_style == "all":
            return self._chunk_all(dobj, **kwargs)
        elif chunking_style == "spatial":
            return self._chunk_spatial(dobj, ngz, **kwargs)
        elif chunking_style == "io":
            return self._chunk_io(dobj, **kwargs)
        else:
            raise NotImplementedError
