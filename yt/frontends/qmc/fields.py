import numpy as np

from yt.fields.field_info_container import FieldInfoContainer
from yt.units import amu
from yt.utilities.lib.cykdtree import PyKDTree
from yt.utilities.lib.particle_kdtree_tools import estimate_density

from .definitions import elementRegister


class QMCFieldInfo(FieldInfoContainer):
    known_other_fields = ()

    known_particle_fields = (
        ("positions", ("code_length", ["particle_position"], None)),
        ("numbers", ("", [], None)),
        ("density", ("code_mass / code_length**3", ["density"], None)),
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
        m_unit = "amu"
        d_unit = "amu / angstrom**3"

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
            pos = data["io", "particle_position"].to(l_unit).d
            mass = data["io", "mass"].to(m_unit).d
            hsml = data[("io", "smoothing_length")].to(l_unit).d
            # NOTE: This is VERY BAD and should be fixed, but is done to get
            # to the next step, for now. Despite being marked as a vector field,
            # pos is coming in with shape (1,), even though yt reads it as (N,3),
            # which is right... the kdtree needs the positions to be the right
            # shape, is why this is done
            if pos.shape == (1,):
                pos = np.ones((1, 3))
            # Construct k-d tree
            kdtree = PyKDTree(
                pos.astype("float64"),
                left_edge=data.ds.domain_left_edge.to_value(l_unit),
                right_edge=data.ds.domain_right_edge.to_value(l_unit),
                periodic=data.ds.periodicity,
                leafsize=2 * int(n_neighbors),
            )
            order = np.argsort(kdtree.idx)
            # Add density field
            dens = estimate_density(
                pos[kdtree.idx],
                mass[kdtree.idx],
                hsml[kdtree.idx],
                kdtree,
                kernel_name=data.ds.kernel_name,
            )
            dens = dens[order]
            data[("io", "density")] = data.ds.arr(dens, d_unit)
            return data[("io", "density")]

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
