"""
HaloCatalog data-file handling function




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.funcs import \
    mylog, \
    parse_h5_attr
from yt.units.yt_array import \
    uvstack
from yt.utilities.on_demand_imports import _h5py as h5py
from yt.utilities.exceptions import YTDomainOverflow
from yt.utilities.io_handler import \
    ParticleIOHandler
from yt.utilities.lib.geometry_utils import compute_morton


class IOHandlerHaloCatalogHDF5(ParticleIOHandler):
    _dataset_type = "halocatalog_hdf5"

    def _initialize_index(self, data_file, regions):
        pcount = data_file.header["num_halos"]
        morton = np.empty(pcount, dtype='uint64')
        mylog.debug("Initializing index % 5i (% 7i particles)",
                    data_file.file_id, pcount)
        ind = 0
        if pcount == 0: return None
        ptype = 'halos'
        with h5py.File(data_file.filename, "r") as f:
            if not f.keys(): return None
            units = parse_h5_attr(f["particle_position_x"], "units")
            pos = data_file._get_particle_positions(ptype, f=f)
            pos = data_file.ds.arr(pos, units). to("code_length")
            dle = self.ds.domain_left_edge.to("code_length")
            dre = self.ds.domain_right_edge.to("code_length")
            if np.any(pos.min(axis=0) < dle) or \
               np.any(pos.max(axis=0) > dre):
                raise YTDomainOverflow(pos.min(axis=0),
                                       pos.max(axis=0),
                                       dle, dre)
            regions.add_data_file(pos, data_file.file_id)
            morton[ind:ind+pos.shape[0]] = compute_morton(
                pos[:,0], pos[:,1], pos[:,2], dle, dre)
        return morton

