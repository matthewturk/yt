"""
Tests for making slices through raw fields

    ytcfg["yt", "internals", "within_testing"] = True

#-----------------------------------------------------------------------------
# Copyright (c) 2017, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import pytest

def compare(ds, field, test_prefix, decimals=12):
    def slice_image(filename_prefix):
        sl = yt.SlicePlot(ds, "z", field)
        sl.set_log("all", False)
        image_file = sl.save(filename_prefix)
        return image_file

    slice_image.__name__ = f"slice_{test_prefix}"
    test = GenericImageTest(ds, slice_image, decimals)
    test.prefix = test_prefix
    return test


raw_fields = "Laser/plt00015"

@requires_ds(raw_fields)
def test_raw_field_slices():
    ds = data_dir_load(raw_fields)
    for field in _raw_field_names:
        yield compare(ds, field, f"answers_raw_{field[1]}")
