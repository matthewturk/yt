"""
Particle-only geometry handler




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import collections
import errno
import numpy as np
import os
import weakref

from yt.funcs import \
    get_pbar, \
    only_on_root
from yt.utilities.logger import ytLogger as mylog
from yt.geometry.geometry_handler import \
    Index, \
    YTDataChunk
from yt.geometry.particle_oct_container import ParticleBitmap
from yt.io.particle_block import ParticleBlock
from yt.utilities.lib.fnv_hash import fnv_hash

CHUNKSIZE = 64**3

class ParticleIndex(Index):
    """The Index subclass for particle datasets"""
    def __init__(self, ds, dataset_type):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        self.float_type = np.float64
        super(ParticleIndex, self).__init__(ds, dataset_type)
        self._initialize_index()

    def _setup_geometry(self):
        self.regions = None

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        return self.ds.arr(0, 'code_length')

    def _get_particle_type_counts(self):
        result = collections.defaultdict(lambda: 0)
        for df in self.data_files:
            for k in df.total_particles.keys():
                result[k] += df.total_particles[k]
        return dict(result)

    def convert(self, unit):
        return self.dataset.conversion_factors[unit]

    _data_files = None
    @property
    def data_files(self):
        if self._data_files is None:
            self._setup_filenames()
        return self._data_files

    _total_particles = None
    @property
    def total_particles(self):
        if self._total_particles is None:
            self._total_particles = sum(
                sum(d.total_particles.values()) for d in self.data_files)
        return self._total_particles

    def _setup_filenames(self):
        template = self.dataset.filename_template
        ndoms = self.dataset.file_count
        cls = self.dataset._file_class
        self._data_files = []
        fi = 0
        for i in range(int(ndoms)):
            df = cls(self.dataset, self.io, template % {'num':i}, fi)
            if max(df.total_particles.values()) == 0:
                continue
            fi += 1
            self._data_files.append(df)

    def _initialize_index(self):
        ds = self.dataset
        only_on_root(mylog.info, "Allocating for %0.3e particles",
                     self.total_particles, global_rootonly = True)

        # if we have not yet set domain_left_edge and domain_right_edge then do
        # an I/O pass over the particle coordinates to determine a bounding box
        if self.ds.domain_left_edge is None:
            min_ppos = np.empty(3, dtype='float64')
            min_ppos[:] = np.nan
            max_ppos = np.empty(3, dtype='float64')
            max_ppos[:] = np.nan
            only_on_root(
                mylog.info,
                'Bounding box cannot be inferred from metadata, reading '
                'particle positions to infer bounding box')
            for df in self.data_files:
                for _, ppos in self.io._yield_coordinates(df):
                    min_ppos = np.nanmin(np.vstack([min_ppos, ppos]), axis=0)
                    max_ppos = np.nanmax(np.vstack([max_ppos, ppos]), axis=0)
            only_on_root(
                mylog.info,
                'Load this dataset with bounding_box=[%s, %s] to avoid I/O '
                'overhead from inferring bounding_box.' % (min_ppos, max_ppos))
            ds.domain_left_edge = ds.arr(1.05*min_ppos, 'code_length')
            ds.domain_right_edge = ds.arr(1.05*max_ppos, 'code_length')
            ds.domain_width = ds.domain_right_edge - ds.domain_left_edge
            
        # use a trivial morton index for datasets containing a single chunk
        self._num_file_chunks = sum(_._num_chunks for _ in self.data_files)
        if self._num_file_chunks == 1:
            order1 = 1
            order2 = 1
        else:
            order1 = ds.index_order[0]
            order2 = ds.index_order[1]

        if order1 == 1 and order2 == 1:
            dont_cache = True
        else:
            dont_cache = False

        if not hasattr(self.ds, '_file_hash'):
            self.ds._file_hash = self._generate_hash()

        self.regions = ParticleBitmap(
            ds.domain_left_edge, ds.domain_right_edge,
            ds.periodicity, self.ds._file_hash,
            self._num_file_chunks,
            index_order1=order1,
            index_order2=order2)

        # Load Morton index from file if provided
        if getattr(ds, 'index_filename', None) is None:
            fname = ds.parameter_filename + ".index{}_{}.ewah".format(
                self.regions.index_order1, self.regions.index_order2)
        else:
            fname = ds.index_filename

        try:
            rflag = self.regions.load_bitmasks(fname)
            rflag = self.regions.check_bitmasks()
            self._initialize_frontend_specific()
            if rflag == 0:
                raise OSError
        except OSError:
            self.regions.reset_bitmasks()
            self._initialize_coarse_index()
            self._initialize_refined_index()
            wdir = os.path.dirname(fname)
            if not dont_cache and os.access(wdir, os.W_OK):
                # Sometimes os mis-reports whether a directory is writable,
                # So pass if writing the bitmask file fails.
                try:
                    self.regions.save_bitmasks(fname)
                except OSError:
                    pass
            rflag = self.regions.check_bitmasks()

    def _initialize_coarse_index(self):
        pb = get_pbar("Initializing coarse index ", self._num_file_chunks)
        chunk_map = {}
        for i, data_file in enumerate(self.data_files):
            pb.update(i)
            to_set = set([])
            for ci, (ptype, pos) in self.io._yield_coordinates(data_file):
                global_chunk_id = chunk_map.setdefault((data_file.file_id, ci, ptype),
                                                       len(chunk_map))
                to_set.add(global_chunk_id)
                ds = self.ds
                if hasattr(ds, '_sph_ptype') and ptype == ds._sph_ptype:
                    hsml = self.io._get_smoothing_length(
                        data_file, ptype, pos.dtype)
                else:
                    hsml = None
                if hasattr(pos, 'in_units'):
                    pos = pos.in_units("code_length")
                self.regions._coarse_index_data_file(
                    pos, hsml, global_chunk_id)
                #self._chunk_map.append((data_file, ci))
            for chunk_id in sorted(to_set):
                self.regions._set_coarse_index_data_file(chunk_id)
        self._chunk_file_map = {v2:v1 for v1, v2 in chunk_map.items()}
        self._chunk_map = chunk_map
        pb.finish()
        self.regions.find_collisions_coarse()

    def _initialize_refined_index(self):
        mask = self.regions.masks.sum(axis=1).astype('uint8')
        max_npart = max(sum(d.total_particles.values())
                        for d in self.data_files) * 28
        sub_mi1 = np.zeros(max_npart, "uint64")
        sub_mi2 = np.zeros(max_npart, "uint64")
        pb = get_pbar("Initializing refined index", self._num_file_chunks)
        for i, data_file in enumerate(self.data_files):
            pb.update(i)
            nsub_mi = 0
            for ci, (ptype, pos) in self.io._yield_coordinates(data_file):
                if hasattr(self.ds, '_sph_ptype') and ptype == self.ds._sph_ptype:
                    hsml = self.io._get_smoothing_length(
                        data_file, ptype, pos.dtype)
                else:
                    hsml = None
                nsub_mi = self.regions._refined_index_data_file(
                    pos, hsml, mask, sub_mi1, sub_mi2,
                    self._chunk_map[data_file.file_id, ci, ptype], nsub_mi)
            self.regions._set_refined_index_data_file(
                sub_mi1, sub_mi2,
                self._chunk_map[data_file.file_id, ci, ptype], nsub_mi)
        pb.finish()
        self.regions.find_collisions_refined()

    def _detect_output_fields(self):
        # TODO: Add additional fields
        self._setup_filenames()
        dsl = []
        units = {}
        pcounts = self._get_particle_type_counts()
        for dom in self.data_files:
            fl, _units = self.io._identify_fields(dom)
            units.update(_units)
            dom._calculate_offsets(fl, pcounts)
            for f in fl:
                if f not in dsl: dsl.append(f)
        self.field_list = dsl
        ds = self.dataset
        ds.particle_types = tuple(set(pt for pt, ds in dsl))
        # This is an attribute that means these particle types *actually*
        # exist.  As in, they are real, in the dataset.
        ds.field_units.update(units)
        ds.particle_types_raw = ds.particle_types

    def _identify_base_chunk(self, dobj):
        # Must check that chunk_info contains the right number of ghost zones
        if getattr(dobj, "_chunk_info", None) is None:
            if isinstance(dobj, ParticleBlock):
                dobj._chunk_info = [dobj]
            else:
                # TODO: only return files
                if getattr(dobj.selector, 'is_all_data', False):
                    dfi, chunk_masks, addfi = self.regions.identify_chunk_masks(
                        dobj.selector)
                    nchunks = len(chunk_masks)
                else:
                    nchunks = self.regions.nchunks
                    dfi = np.arange(nchunks)
                # At this point, dfi is the index into self._chunk_map.  So
                # we can get the unique data files by looking at the
                # numbers.  Our objects will now be the data file and the
                # chunk IDs.
                # Below, you shall find some code that could be done in a
                # set of nested list comprehensions and iterators and
                # whatnot, but in the interest of clarity, it is written
                # out.
                dobj._chunk_info = [None for _ in range(nchunks)]
                for i, d in enumerate(dfi):
                    data_file_id, chunk_id, ptype = self._chunk_file_map[d]
                    df = self.data_files[data_file_id]
                    domain_id = i + 1
                    dobj._chunk_info[i] = ParticleFile(
                        dobj, df, chunk_id, ptype, domain_id = domain_id)
                # NOTE: One fun thing about the way IO works is that it
                # consolidates things quite nicely.  So we should feel free to
                # create as many objects as part of the chunk as we want, since
                # it'll take the set() of them.  So if we break stuff up like
                # this here, we end up in a situation where we have the ability
                # to break things down further later on for buffer zones and the
                # like.
        dobj._current_chunk, = self._chunk_all(dobj)

    def _chunk_all(self, dobj):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        yield YTDataChunk(dobj, "all", oobjs, None)

    def _chunk_spatial(self, dobj, ngz, sort = None, preload_fields = None):
        sobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for og in sobjs:
            with og._expand_data_files():
                if ngz > 0:
                    g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
                else:
                    g = og
                yield YTDataChunk(dobj, "spatial", [g])

    def _chunk_io(self, dobj, cache = True, local_only = False):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for container in oobjs:
            yield YTDataChunk(dobj, "io", [container], None, cache = cache)

    def _generate_hash(self):
        # Generate an FNV hash by creating a byte array containing the
        # modification time of as well as the first and last 1 MB of data in
        # every output file
        ret = bytearray()
        for pfile in self.data_files:
            try:
                mtime = os.path.getmtime(pfile.filename)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    # this is an in-memory file so we return with a dummy
                    # value
                    return -1
                else:
                    raise
            ret.extend(str(mtime).encode('utf-8'))
            size = os.path.getsize(pfile.filename)
            if size > 1e6:
                size = int(1e6)
            with open(pfile.filename, 'rb') as fh:
                # read in first and last 1 MB of data
                data = fh.read(size)
                fh.seek(-size, os.SEEK_END)
                data = fh.read(size)
                ret.extend(data)
            return fnv_hash(ret)

    def _initialize_frontend_specific(self):
        """This is for frontend-specific initialization code

        If there are frontend-specific things that need to be set while 
        creating the index, this function forces these operations to happen
        in cases where we are reloading the index from a sidecar file.
        """
        pass
