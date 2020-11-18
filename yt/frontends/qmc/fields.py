from yt.fields.field_info_container import FieldInfoContainer
from yt.utilities.logger import ytLogger as mylog

from .definitions import elementRegister


class QMCFieldInfo(FieldInfoContainer):
    known_other_fields = ()

    known_particle_fields = (
        ("positions", ("code_length", [], None)),
        ("numbers", ("", [], None)),
        ("mass", ("code_mass", [], None)),
        ("density", ("code_mass/code_length**3", [], None)),
    )

    def __init__(self, ds, field_list, slice_info=None):
        super(QMCFieldInfo, self).__init__(ds, field_list, slice_info=slice_info)

    def setup_particle_fields(self, ptype, *args, **kwargs):
        self._setup_masses()
        self._setup_densities()
        super().setup_particle_fields(ptype, *args, **kwargs)

    def _setup_masses(self):
        """
        Maps the element numbers from the numbers field to element
        masses.
        """
        def _atomic_mass(field, data):
            return elementRegister[data[("io", "numbers")]]["mass"]
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
        l_unit = "code_length"
        m_unit = "code_mass"
        d_unit = "code_mass / code_length**3"
        def _density(field, data):
            # Read basic fields
            ad = data.ds.all_data()
            pos = ad[sph_ptype, "positions"].to(l_unit).d
            mass = ad[sph_ptype, "mass"].to(m_unit).d
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
                if (sph_ptype, fname) in data.ds.derived_field_list:
                    mylog.info(
                        "Field ('%s','%s') already exists. Skipping", sph_ptype, fname
                    )
                    return True
                else:
                    mylog.info("Generating field ('%s','%s')", sph_ptype, fname)
                    return False
            data = {}
            # Add smoothing length field
            fname = "smoothing_length"
            if not exists(fname):
                hsml = data.ds.index.io._generate_smoothing_length(data.ds.index)
                data[(sph_ptype, "smoothing_length")] = (hsml, l_unit)
            else:
                hsml = ad[sph_ptype, fname].to(l_unit).d
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
                data[(sph_ptype, "density")] = (dens, d_unit)
                return data[(sph_ptype, "density")]
        self.add_field(
            ("io", "density"),
            sampling_type="particle",
            function=_density,
            units="amu/angstrom**3",
        )
