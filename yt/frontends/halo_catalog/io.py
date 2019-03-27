"""
HaloCatalog data-file handling function




"""

#-----------------------------------------------------------------------------
# Copyright (c) yt Development Team. All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import contextlib

from yt.utilities.on_demand_imports import _h5py as h5py
from yt.utilities.io_handler import \
    ParticleIOHandler
from yt.data_objects.particle_store import \
    ParticleFile
from yt.funcs import \
    parse_h5_attr

class IOHandlerHaloCatalogHDF5(ParticleIOHandler):
    _dataset_type = "halocatalog_hdf5"

class HaloCatalogHDF5File(ParticleFile):
    def __init__(self, ds, io, filename, file_id):
        self.filename = filename
        with self._open_file() as f:
            self.header = dict((field, parse_h5_attr(f, field)) \
                               for field in f.attrs.keys())
        super(HaloCatalogHDF5File, self).__init__(
            ds, io, filename, file_id)

    def _read_particle_positions(self, ptype, state=None):
        """
        Read all particle positions in this file.
        """

        if state is None:
            f, (si, ei) = h5py.File(self.filename, "r"), (None, None)
        else:
            f, (si, ei) = state

        units = parse_h5_attr(f['particle_position_x'], "units")
        pcount = self.total_particles[ptype]
        if None not in (ei, si):
            pcount = min(pcount, ei-si)
        pos = np.empty((pcount, 3), dtype="float64")
        for i, ax in enumerate('xyz'):
            pos[:, i] = f["particle_position_%s" % ax][si:ei]
        pos = self.ds.arr(pos, units)

        if state is None:
            f.close()

        return pos

    def _iter_fields(self, ptype, field_list, state=None):
        if state is None:
            f, (si, ei) = h5py.File(self.filename, 'r'), (None, None)
        else:
            f, (si, ei) = state

        for field in field_list:
            yield (ptype, field), f[field][si:ei]

        if state is None:
            f.close()

    @contextlib.contextmanager
    def _open_file(self):
        with h5py.File(self.filename, 'r') as f:
            yield f

    def _count_particles(self):
        return {'halos': self.header['num_halos']}

    def _identify_fields(self):
        with self._open_file() as f:
            fields = [("halos", field) for field in f]
            units = dict([(("halos", field),
                           parse_h5_attr(f[field], "units"))
                          for field in f])
        return fields, units
