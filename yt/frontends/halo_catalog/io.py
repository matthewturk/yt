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
        with h5py.File(filename, "r") as f:
            self.header = dict((field, parse_h5_attr(f, field)) \
                               for field in f.attrs.keys())
        super(HaloCatalogHDF5File, self).__init__(
            ds, io, filename, file_id)

    def _read_particle_positions(self, ptype, f=None):
        """
        Read all particle positions in this file.
        """

        if f is None:
            close = True
            f = h5py.File(self.filename, "r")
        else:
            close = False

        units = parse_h5_attr(f['particle_position_x'], "units")
        pcount = self.header["num_halos"]
        pos = np.empty((pcount, 3), dtype="float64")
        for i, ax in enumerate('xyz'):
            pos[:, i] = f["particle_position_%s" % ax][()]
        pos = self.ds.arr(pos, units)

        if close:
            f.close()

        return pos

    def _read_particle_fields(self, ptype, field_list, frange=None):
        if frange is None:
            frange = (None, None)
        si, ei = frange

        f = h5py.File(self.filename, 'r')

        for field in field_list:
            yield field, f[field][si:ei]

        f.close()

    def _count_particles(self):
        return {'halos': self.header['num_halos']}

    def _identify_fields(self):
        with h5py.File(self.filename, "r") as f:
            fields = [("halos", field) for field in f]
            units = dict([(("halos", field),
                           parse_h5_attr(f[field], "units"))
                          for field in f])
        return fields, units


