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
from yt.utilities.on_demand_imports import _h5py as h5py
from yt.utilities.exceptions import YTDomainOverflow
from yt.utilities.io_handler import \
    ParticleIOHandler
from yt.utilities.lib.geometry_utils import compute_morton


class IOHandlerHaloCatalogHDF5(ParticleIOHandler):
    _dataset_type = "halocatalog_hdf5"
