"""
Tests for making line plots

import yt
from yt.testing import ANSWER_TEST_TAG, assert_equal, fake_random_ds
from yt.utilities.answer_testing.framework import GenericImageTest
from yt.visualization.line_plot import _validate_point


def setup():
    """Test specific setup."""
    from yt.config import ytcfg

    ytcfg["yt", "internals", "within_testing"] = True


def compare(ds, plot, test_prefix, test_name, decimals=12):
    def image_from_plot(filename_prefix):
        return plot.save(filename_prefix)

    image_from_plot.__name__ = f"line_{test_prefix}"
    test = GenericImageTest(ds, image_from_plot, decimals)
    test.prefix = test_prefix
    test.answer_name = test_name
    return test

import pytest

import yt
from yt.testing import assert_equal, fake_random_ds
from yt.utilities.answer_testing.answer_tests import generic_image_test
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
        gi = generic_image_test(img_fname)
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
        gi = generic_image_test(img_fname)
        self.hashes.update({"generic_image": gi})


def test_line_buffer():
    ds = fake_random_ds(32)
    lb = yt.LineBuffer(ds, (0, 0, 0), (1, 1, 1), 512, label="diag")
    lb[("gas", "density")]
    lb[("gas", "velocity_x")]
    assert_equal(lb[("gas", "density")].size, 512)
    lb[("gas", "density")] = 0
    assert_equal(lb[("gas", "density")], 0)
    assert_equal(set(lb.keys()), {("gas", "density"), ("gas", "velocity_x")})
    del lb[("gas", "velocity_x")]
    assert_equal(set(lb.keys()), {("gas", "density")})


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
