import numpy as np

from yt.fields.field_info_container import FieldInfoContainer
from yt.units import amu
from yt.utilities.lib.cykdtree import PyKDTree
from yt.utilities.lib.particle_kdtree_tools import estimate_density
from yt.utilities.logger import ytLogger as mylog

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

    def _setup_densities(self, n_neighbors=8, kernel="cubic", sph_ptype="io"):
        """Add SPH fields for the specified particle type.

        For a particle type with "particle_position" and "particle_mass" already
        defined, this method adds the "smoothing_length" and "density" fields.
        "smoothing_length" is computed as the distance to the nth nearest
        neighbor. "density" is computed as the SPH (gather) smoothed mass. The
        SPH fields are added only if they don't already exist.

        Parameters
        ----------
        n_neighbors : int
            The number of neighbors to use in smoothing length computation.
        kernel : str
            The kernel function to use in density estimation.
        sph_ptype : str
            The SPH particle type. Each dataset has one sph_ptype only. This
            method will overwrite existing sph_ptype of the dataset.

        """
        mylog.info("Generating SPH fields")
        # Unify units
        # l_unit = "code_length"
        # m_unit = "code_mass"
        # d_unit = "code_mass / code_length**3"
        # NOTE: For some reason "code_mass" doesn't show up in the unit registry,
        # so the .to() step on mass below fails. My guess is it has something
        # to do with the fact that mass is a derived field, but I'm not sure
        # what's going on
        l_unit = "angstrom"
        m_unit = "amu"
        d_unit = "amu / angstrom**3"

        def _density(field, data):
            # Read basic fields
            pos = data["io", "particle_positions"].to(l_unit).d
            mass = data["io", "mass"].to(m_unit).d
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

            def exists(fname):
                if ("gas", fname) in data.ds.derived_field_list:
                    mylog.info(
                        "Field ('%s','%s') already exists. Skipping", sph_ptype, fname
                    )
                    return True
                else:
                    mylog.info("Generating field ('%s','%s')", "gas", fname)
                    return False

            # Add smoothing length field
            fname = "smoothing_length"
            if not exists(fname):
                hsml = data.ds.index.io._generate_smoothing_length(data.ds.index)
                data[("gas", "smoothing_length")] = (hsml, l_unit)
            else:
                hsml = data["gas", fname].to(l_unit).d
            # Add density field
            fname = "density"
            if not exists(fname):
                dens = estimate_density(
                    pos[kdtree.idx],
                    mass[kdtree.idx],
                    hsml[kdtree.idx],
                    kdtree,
                    kernel_name=kernel,
                )
                dens = dens[order]
                data[("gas", "density")] = (dens, d_unit)
            return data[("gas", "density")]

        self.add_field(
            ("gas", "density"),
            sampling_type="particle",
            function=_density,
            units="amu/angstrom**3",
        )
