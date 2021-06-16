import numpy as np

from yt.fields.field_info_container import FieldInfoContainer
from yt.units import amu, cm

from .definitions import elementRegister


class QMCFieldInfo(FieldInfoContainer):
    known_other_fields = ()

    known_particle_fields = (
        ("positions", ("code_length", ["particle_position"], None)),
        ("numbers", ("", [], None)),
    )

    def __init__(self, ds, field_list, slice_info=None):
        super().__init__(ds, field_list, slice_info=slice_info)

    def setup_particle_fields(self, ptype, *args, **kwargs):
        super().setup_particle_fields(ptype, *args, **kwargs)
        self._setup_masses()
        self._setup_densities()

    def _setup_masses(self):
        """
        Maps the element numbers from the numbers field to element
        masses.
        """

        def _atomic_mass(field, data):
            mass = data["io", "numbers"].d.copy() * amu
            return mass * elementRegister[int(data[("io", "numbers")].d[0])][2]

        self.add_field(
            ("io", "mass"),
            sampling_type="particle",
            function=_atomic_mass,
            units="amu",
        )

    def _setup_densities(self, n_neighbors=8, sph_ptype="io"):
        l_unit = "angstrom"

        # def _hsml(field, data):
        #     hsml = data.ds.index.io._generate_smoothing_length(data.ds.index)
        #     data[("io", "smoothing_length")] = data.ds.arr(hsml, l_unit)
        #     return data[("io", "smoothing_length")]
        def _hsml(field, data):
            hsml = (
                np.ones((data.ds.particle_type_counts[sph_ptype],), dtype=np.float64)
                * 10.0
            )
            data[("io", "smoothing_length")] = data.ds.arr(hsml, l_unit)
            return data[("io", "smoothing_length")]

        def _density(field, data):
            mass = data["io", "numbers"].d * amu / cm ** 3
            return mass

        self.add_field(
            ("io", "smoothing_length"),
            sampling_type="particle",
            function=_hsml,
            units="angstrom",
        )
        self.add_field(
            ("io", "density"),
            sampling_type="particle",
            function=_density,
            units="amu/angstrom**3",
        )
        self.alias(("io", "density"), ("io", "Density"))
