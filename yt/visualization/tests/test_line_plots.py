"""
Tests for making line plots

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2017, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------
import os
import tempfile

import pytest

import yt
from yt.testing import assert_equal, fake_random_ds
from yt.utilities.answer_testing.answer_tests import generic_image
from yt.visualization.line_plot import _validate_point


def image_from_plot(plot):
    tmpfd, tmpfname = tempfile.mkstemp(suffix=".png")
    os.close(tmpfd)
    plot.save(tmpfname)
    return tmpfname


@pytest.mark.answer_test
@pytest.mark.usefixtures("temp_dir", "hashing")
class TestLinePlots:
    def test_line_plot(self):
        ds = fake_random_ds(4)
        fields = [field for field in ds.field_list if field[0] == "stream"]
        field_labels = {f: f[1] for f in fields}
        plot = yt.LinePlot(
            ds, fields, (0, 0, 0), (1, 1, 0), 1000, field_labels=field_labels
        )
        plot.annotate_legend(fields[0])
        plot.annotate_legend(fields[1])
        plot.set_x_unit("cm")
        plot.set_unit(fields[0], "kg/cm**3")
        plot.annotate_title(fields[0], "Density Plot")
        img_fname = image_from_plot(plot)
        gi = generic_image(img_fname)
        self.hashes.update({"generic_image": gi})

    def test_multi_line_plot(self):
        ds = fake_random_ds(4)
        fields = [field for field in ds.field_list if field[0] == "stream"]
        field_labels = {f: f[1] for f in fields}
        lines = []
        lines.append(
            yt.LineBuffer(ds, [0.25, 0, 0], [0.25, 1, 0], 100, label="x = 0.5")
        )
        lines.append(yt.LineBuffer(ds, [0.5, 0, 0], [0.5, 1, 0], 100, label="x = 0.5"))
        plot = yt.LinePlot.from_lines(ds, fields, lines, field_labels=field_labels)
        plot.annotate_legend(fields[0])
        plot.annotate_legend(fields[1])
        img_fname = image_from_plot(plot)
        gi = generic_image(img_fname)
        self.hashes.update({"generic_image": gi})


def test_line_buffer():
    ds = fake_random_ds(32)
    lb = yt.LineBuffer(ds, (0, 0, 0), (1, 1, 1), 512, label="diag")
    lb["density"]
    lb["velocity_x"]
    assert_equal(lb["density"].size, 512)
    lb["density"] = 0
    assert_equal(lb["density"], 0)
    assert_equal(set(lb.keys()), set(["density", "velocity_x"]))
    del lb["velocity_x"]
    assert_equal(set(lb.keys()), set(["density"]))


def test_validate_point():
    ds = fake_random_ds(3)
    with pytest.raises(RuntimeError) as ex:
        _validate_point(0, ds, start=True)
    assert_equal(str(ex.exception), "Input point must be array-like")

    with pytest.raises(RuntimeError) as ex:
        _validate_point(ds.arr([[0], [1]], "code_length"), ds, start=True)
    assert_equal(str(ex.exception), "Input point must be a 1D array")

    with pytest.raises(RuntimeError) as ex:
        _validate_point(ds.arr([0, 1], "code_length"), ds, start=True)
    assert_equal(
        str(ex.exception), "Input point must have an element for each dimension"
    )

    ds = fake_random_ds([32, 32, 1])
    _validate_point(ds.arr([0, 1], "code_length"), ds, start=True)
    _validate_point(ds.arr([0, 1], "code_length"), ds)
