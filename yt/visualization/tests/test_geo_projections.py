"""
Tests for making unstructured mesh slices

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------
import os
import tempfile
import unittest

    ytcfg["yt", "internals", "within_testing"] = True

import yt
from yt.testing import fake_amr_ds, requires_module
from yt.utilities.answer_testing.answer_tests import generic_image
from yt.visualization.geo_plot_utils import get_mpl_transform, transform_list


    slice_image.__name__ = f"slice_{test_prefix}"
    test = GenericImageTest(ds, slice_image, decimals)
    test.prefix = test_prefix
    test.answer_name = test_name
    return test


@requires_module("cartopy")
@attr(ANSWER_TEST_TAG)
def test_geo_slices_amr():
    ds = fake_amr_ds(geometry="geographic")
    for transform in transform_list:
        if transform == "UTM":
            # requires additional argument so we skip
            continue
        if transform == "OSNI":
            # avoid crashes, see https://github.com/SciTools/cartopy/issues/1177
            continue
        for field in ds.field_list:
            prefix = f"{field[0]}_{field[1]}_{transform}"
            yield compare(
                ds,
                field,
                "altitude",
                test_prefix=prefix,
                test_name="geo_slices_amr",
                projection=transform,
            )


class TestGeoProjections(unittest.TestCase):
    def setUp(self):
        self.ds = fake_amr_ds(geometry="geographic")

    def tearDown(self):
        del self.ds
        del self.slc

    def test_geo_projection_setup(self):

        from yt.utilities.on_demand_imports import _cartopy as cartopy

        axis = "altitude"
        self.slc = yt.SlicePlot(self.ds, axis, ("stream", "Density"), origin="native")

        assert isinstance(self.slc._projection, cartopy.crs.Mollweide)
        assert isinstance(self.slc._transform, cartopy.crs.PlateCarree)
        assert self.ds.coordinates.data_projection[axis] == "Mollweide"
        assert self.ds.coordinates.data_transform[axis] == "PlateCarree"
        assert isinstance(
            self.slc._projection,
            type(self.slc.plots[("stream", "Density")].axes.projection),
        )

    def test_geo_projections(self):
        from yt.utilities.on_demand_imports import _cartopy as cartopy

        self.slc = yt.SlicePlot(
            self.ds, "altitude", ("stream", "Density"), origin="native"
        )

        for transform in transform_list:
            if transform == "UTM":
                # this requires special arguments so let's skip it
                continue
            if transform == "OSNI":
                # avoid crashes, see https://github.com/SciTools/cartopy/issues/1177
                continue
            self.slc.set_mpl_projection(transform)
            proj_type = type(get_mpl_transform(transform))

            assert isinstance(self.slc._projection, proj_type)
            assert isinstance(self.slc._transform, cartopy.crs.PlateCarree)
            assert isinstance(
                self.slc.plots[("stream", "Density")].axes.projection, proj_type
            )

    def test_projection_object(self):
        from yt.utilities.on_demand_imports import _cartopy as cartopy

        shortlist = ["Orthographic", "PlateCarree", "Mollweide"]

        for transform in shortlist:
            projection = get_mpl_transform(transform)
            proj_type = type(projection)
            self.slc = yt.SlicePlot(
                self.ds, "altitude", ("stream", "Density"), origin="native"
            )
            self.slc.set_mpl_projection(projection)

            assert isinstance(self.slc._projection, proj_type)
            assert isinstance(self.slc._transform, cartopy.crs.PlateCarree)
            assert isinstance(
                self.slc.plots[("stream", "Density")].axes.projection, proj_type
            )

    def test_nondefault_transform(self):
        from yt.utilities.on_demand_imports import _cartopy as cartopy

        axis = "altitude"
        self.ds.coordinates.data_transform[axis] = "Miller"
        self.slc = yt.SlicePlot(self.ds, axis, ("stream", "Density"), origin="native")

        shortlist = ["Orthographic", "PlateCarree", "Mollweide"]

        for transform in shortlist:

            self.slc.set_mpl_projection(transform)
            proj_type = type(get_mpl_transform(transform))

            assert isinstance(self.slc._projection, proj_type)
            assert isinstance(self.slc._transform, cartopy.crs.Miller)
            assert self.ds.coordinates.data_projection[axis] == "Mollweide"
            assert self.ds.coordinates.data_transform[axis] == "Miller"
            assert isinstance(
                self.slc.plots[("stream", "Density")].axes.projection, proj_type
            )



class TestNonGeoProjections(unittest.TestCase):
    def setUp(self):
        self.ds = fake_amr_ds()

    def tearDown(self):
        del self.ds
        del self.slc

    def test_projection_setup(self):
        axis = "x"
        self.slc = yt.SlicePlot(self.ds, axis, ("stream", "Density"), origin="native")

        assert self.ds.coordinates.data_projection[axis] is None
        assert self.ds.coordinates.data_transform[axis] is None
        assert self.slc._projection is None
        assert self.slc._transform is None
