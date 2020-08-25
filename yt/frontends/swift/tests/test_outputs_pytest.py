import numpy as np
import pytest

from yt.frontends.swift.api import SwiftDataset
from yt.testing import ParticleSelectionComparison, assert_almost_equal
from yt.utilities.on_demand_imports import _h5py as h5py

# Test data
keplerian_ring = "KeplerianRing/keplerian_ring_0020.hdf5"
EAGLE_6 = "EAGLE_6/eagle_0005.hdf5"

# Combined the tests for loading a file and ensuring the units have been
# implemented correctly to save time on re-loading a dataset
@pytest.mark.answer_test
class TestSwift:
    answer_file = None
    saved_hashes = None

    @pytest.mark.parametrize("ds", [keplerian_ring], indirect=True)
    def test_non_cosmo_dataset(self, ds):
        assert type(ds) is SwiftDataset
        field = ("gas", "density")
        ad = ds.all_data()
        yt_density = ad[field]
        yt_coords = ad[(field[0], "position")]
        # load some data the old fashioned way
        fh = h5py.File(ds.parameter_filename, "r")
        part_data = fh["PartType0"]
        # set up a conversion factor by loading the unit mas and unit length in cm,
        # and then converting to proper coordinates
        units = fh["Units"]
        units = dict(units.attrs)
        density_factor = float(units["Unit mass in cgs (U_M)"])
        density_factor /= float(units["Unit length in cgs (U_L)"]) ** 3
        # now load the raw density and coordinates
        raw_density = part_data["Density"][:].astype("float64") * density_factor
        raw_coords = part_data["Coordinates"][:].astype("float64")
        fh.close()
        # sort by the positions - yt often loads in a different order
        ind_raw = np.lexsort((raw_coords[:, 2], raw_coords[:, 1], raw_coords[:, 0]))
        ind_yt = np.lexsort((yt_coords[:, 2], yt_coords[:, 1], yt_coords[:, 0]))
        raw_density = raw_density[ind_raw]
        yt_density = yt_density[ind_yt]
        # make sure we are comparing fair units
        assert str(yt_density.units) == "g/cm**3"
        # make sure the actual values are the same
        assert_almost_equal(yt_density.d, raw_density)

    @pytest.mark.parametrize("ds", [keplerian_ring], indirect=True)
    def test_non_cosmo_dataset_selection(self, ds):
        psc = ParticleSelectionComparison(ds)
        psc.run_defaults()

    @pytest.mark.parametrize("ds", [EAGLE_6], indirect=True)
    def test_cosmo_dataset(self, ds):
        assert type(ds) == SwiftDataset
        field = ("gas", "density")
        ad = ds.all_data()
        yt_density = ad[field]
        yt_coords = ad[(field[0], "position")]
        # load some data the old fashioned way
        fh = h5py.File(ds.parameter_filename, "r")
        part_data = fh["PartType0"]
        # set up a conversion factor by loading the unit mas and unit length in cm,
        # and then converting to proper coordinates
        units = fh["Units"]
        units = dict(units.attrs)
        density_factor = float(units["Unit mass in cgs (U_M)"])
        density_factor /= float(units["Unit length in cgs (U_L)"]) ** 3
        # add the redshift factor
        header = fh["Header"]
        header = dict(header.attrs)
        density_factor *= (1.0 + float(header["Redshift"])) ** 3
        # now load the raw density and coordinates
        raw_density = part_data["Density"][:].astype("float64") * density_factor
        raw_coords = part_data["Coordinates"][:].astype("float64")
        fh.close()
        # sort by the positions - yt often loads in a different order
        ind_raw = np.lexsort((raw_coords[:, 2], raw_coords[:, 1], raw_coords[:, 0]))
        ind_yt = np.lexsort((yt_coords[:, 2], yt_coords[:, 1], yt_coords[:, 0]))
        raw_density = raw_density[ind_raw]
        yt_density = yt_density[ind_yt]
        # make sure we are comparing fair units
        assert str(yt_density.units) == "g/cm**3"
        # make sure the actual values are the same
        assert_almost_equal(yt_density.d, raw_density)

    @pytest.mark.parametrize("ds", [EAGLE_6], indirect=True)
    def test_cosmo_dataset_selection(self, ds):
        psc = ParticleSelectionComparison(ds)
        psc.run_defaults()
