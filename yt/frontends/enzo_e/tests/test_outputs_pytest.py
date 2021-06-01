"""
Title: test_enzo_p.py
Purpose: Enzo-P frontend tests
    Copyright (c) 2017, yt Development Team.
    Distributed under the terms of the Modified BSD License.
    The full license is in the file COPYING.txt, distributed with this
    software.
"""
import numpy as np
import pytest

from yt.utilities.on_demand_imports import \
    _h5py as h5py

from yt.testing import \
    assert_equal, \
    requires_file, \
    assert_array_equal
from yt.frontends.enzo_p.api import EnzoPDataset
from yt.utilities.answer_testing.answer_tests import \
    pixelized_projection_values, field_values
from yt.utilities.answer_testing import utils


# Test data
hello_world = "hello-0210/hello-0210.block_list"
ep_cosmo = "ENZOP_DD0140/ENZOP_DD0140.block_list"


# Global field info
_fields = ("density", "total_energy",
           "velocity_x", "velocity_y")
_pfields = ("particle_position_x", "particle_position_y",
            "particle_position_z", "particle_velocity_x",
            "particle_velocity_y", "particle_velocity_z")


@pytest.mark.answer_test
@pytest.mark.usefixtures('answer_file')
class TestEnzoP:

    @pytest.mark.parametrize('ds', [hello_world], indirect=True)
    def test_EnzoPDataset(self, ds):
        assert isinstance(ds, EnzoPDataset)

    @pytest.mark.usefixtures('hashing')
    @pytest.mark.parametrize('ds', [hello_world], indirect=True)
    def test_hello_world(self, f, a, d, w, ds):
        ppv = pixelized_projection_values(ds, a, f, w, d)
        self.hashes.update({'pixelized_projection_values' : ppv})
        fv = field_values(ds, f, d)
        self.hashes.update({'field_values' : fv})
        dobj = utils.create_obj(ds, d)
        s1 = dobj["ones"].sum()
        s2 = sum(mask.sum() for block, mask in dobj.blocks)
        assert_equal(s1, s2)

    @pytest.mark.usefixtures('hashing')
    @pytest.mark.parametrize('ds', [ep_cosmo], indirect=True)
    def test_particle_fields(self, f, d, ds):
        fv = field_values(ds, f, d, particle_type=True)
        self.hashes.update({'field_values' : fv})
        dobj = utils.create_obj(ds, d)
        s1 = dobj["ones"].sum()
        s2 = sum(mask.sum() for block, mask in dobj.blocks)
        assert_equal(s1, s2)

    @pytest.mark.parametrize('ds', [hello_world], indirect=True)
    def test_hierarchy(self, ds):
        fh = h5py.File(ds.index.grids[0].filename, "r")
        for grid in ds.index.grids:
            assert_array_equal(
                grid.LeftEdge.d, fh[grid.block_name].attrs["enzo_GridLeftEdge"])
            assert_array_equal(
                ds.index.grid_left_edge[grid.id], grid.LeftEdge)
            assert_array_equal(
                ds.index.grid_right_edge[grid.id], grid.RightEdge)
            for child in grid.Children:
                assert (child.LeftEdge >= grid.LeftEdge).all()
                assert (child.RightEdge <= grid.RightEdge).all()
                assert_equal(child.Parent.id, grid.id)
        fh.close()

    @pytest.mark.parametrize('ds', [ep_cosmo], indirect=True)
    def test_critical_density(self, ds):
        c1 = (ds.r["dark", "particle_mass"].sum() +
              ds.r["gas", "cell_mass"].sum()) / \
              ds.domain_width.prod() / ds.critical_density
        c2 = ds.omega_matter * (1 + ds.current_redshift)**3 / \
          (ds.omega_matter * (1 + ds.current_redshift)**3 + ds.omega_lambda)
        assert np.abs(c1 - c2) / max(c1, c2) < 1e-3
