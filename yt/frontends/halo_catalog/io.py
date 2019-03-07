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

from yt.utilities.io_handler import \
    ParticleIOHandler

class IOHandlerHaloCatalogHDF5(ParticleIOHandler):
    _dataset_type = "halocatalog_hdf5"
