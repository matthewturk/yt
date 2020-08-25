import numpy as np
from nose.plugins.attrib import attr

from yt.testing import ANSWER_TEST_TAG, fake_vr_orientation_test_ds
from yt.utilities.answer_testing.framework import (
    GenericImageTest,
    VRImageComparisonTest,
)
from yt.visualization.volume_rendering.api import (
    ColorTransferFunction,
    Scene,
    create_volume_source,
    off_axis_projection,
)


@attr(ANSWER_TEST_TAG)
def test_orientation():
    ds = fake_vr_orientation_test_ds()

    sc = Scene()

    vol = create_volume_source(ds, field=("gas", "density"))
    sc.add_source(vol)

    tf = vol.transfer_function
    tf = ColorTransferFunction((0.1, 1.0))
    tf.sample_colormap(1.0, 0.01, colormap="coolwarm")
    tf.sample_colormap(0.8, 0.01, colormap="coolwarm")
    tf.sample_colormap(0.6, 0.01, colormap="coolwarm")
    tf.sample_colormap(0.3, 0.01, colormap="coolwarm")

    n_frames = 1
    orientations = [[-0.3, -0.1, 0.8]]

    theta = np.pi / n_frames
    test_name = "vr_orientation"

    for lens_type, decimals in [("perspective", 12), ("plane-parallel", 2)]:
        # set a much lower precision for plane-parallel tests, see
        # https://github.com/yt-project/yt/issue/3069
        # https://github.com/yt-project/yt/pull/3068
        # https://github.com/yt-project/yt/pull/3294
        frame = 0

@pytest.mark.answer_test
@pytest.mark.usefixtures("hashing")
class TestVROrientation:
    answer_file = None
    saved_hashes = None

    def test_vr_images(self, ds_vr, sc, lens_type):
        n_frames = 1
        theta = np.pi / n_frames
        cam = sc.add_camera(ds_vr, lens_type=lens_type)
        cam.resolution = (1000, 1000)
        cam.position = ds_vr.arr(np.array([-4.0, 0.0, 0.0]), "code_length")
        cam.switch_orientation(
            normal_vector=[1.0, 0.0, 0.0], north_vector=[0.0, 0.0, 1.0]
        )
        cam.set_width(ds_vr.domain_width * 2.0)
        test1 = VR_image_comparison(sc)
        self.hashes.update({"test1": test1})
        for i in range(n_frames):
            center = ds_vr.arr([0, 0, 0], "code_length")
            cam.yaw(theta, rot_center=center)
            test2 = VR_image_comparison(sc)
            # Updating nested dictionaries doesn't add the new key, it
            # overwrites the old one (so d.update({'key1' : {'subkey1' : 1}})
            # is d = {'key1' : {'subkey1' : 1}}. Then if you do
            # d.update({'key1' : {'subkey2' : 2}}), d = {'key1' : 'subkey2':2}},
            # so to add subkey2 to key1's subdictionary, you need to do
            # d['key1'].update({'subkey2' : 2}))
            if "test2" not in self.hashes:
                self.hashes.update({"test2": {str(i): test2}})
            else:
                self.hashes["test2"].update({str(i): test2})
        for i in range(n_frames):
            theta = np.pi / n_frames
            center = ds_vr.arr([0, 0, 0], "code_length")
            cam.pitch(theta, rot_center=center)
            test3 = VR_image_comparison(sc)
            if "test3" not in self.hashes:
                self.hashes.update({"test3": {str(i): test3}})
            else:
                self.hashes["test3"].update({str(i): test3})
        for i in range(n_frames):
            theta = np.pi / n_frames
            center = ds_vr.arr([0, 0, 0], "code_length")
            cam.roll(theta, rot_center=center)
            desc = "roll_%s_%04d" % (lens_type, frame)
            test4 = VRImageComparisonTest(sc, ds, desc, decimals)
            test4.answer_name = test_name
            yield test4

    center = [0.5, 0.5, 0.5]
    width = [1.0, 1.0, 1.0]

    for i, orientation in enumerate(orientations):
        image = off_axis_projection(
            ds, center, orientation, width, 512, ("gas", "density"), no_ghost=False
        )

        def offaxis_image_func(filename_prefix):
            return image.write_image(filename_prefix)

        test5 = GenericImageTest(ds, offaxis_image_func, decimals)
        test5.prefix = f"oap_orientation_{i}"
        test5.answer_name = test_name
        yield test5
