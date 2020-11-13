import os
import uuid

from ase.io.formats import filetype
from ase.io.formats import UnknownFileTypeError
import numpy as np

from yt.data_objects.static_output import ParticleDataset
from yt.data_objects.static_output import ParticleFile
from yt.geometry.particle_geometry_handler import ParticleIndex
from yt.utilities.logger import ytLogger as mylog

from .definitions import qmc_unit_sys
from .fields import QMCFieldInfo


class QMCIndex(ParticleIndex):
    def __init__(self, ds, dataset_type):
        super().__init__(ds, dataset_type)
        self._initialize_index()

    def _initialize_index(self):
        ds = self.dataset
        ds._file_hash = self._generate_hash()
        super()._initialize_index()

    def _initialize_frontend_specific(self):
        super()._initialize_frontend_specific()

    def _generate_kdtree(self, fname):
        from yt.utilities.lib.cykdtree import PyKDTree

        if fname is not None:
            if os.path.exists(fname):
                mylog.info("Loading KDTree from %s", os.path.basename(fname))
                kdtree = PyKDTree.from_file(fname)
                if kdtree.data_version != self.ds._file_hash:
                    mylog.info("Detected hash mismatch, regenerating KDTree")
                else:
                    self._kdtree = kdtree
                    return
        positions = []
        for data_file in self.data_files:
            for _, ppos in self.io._yield_coordinates(
                data_file, needed_ptype=self.ds._sph_ptypes[0]
            ):
                positions.append(ppos)
        if positions == []:
            self._kdtree = None
            return
        positions = np.concatenate(positions)
        mylog.info("Allocating KDTree for %s particles", positions.shape[0])
        self._kdtree = PyKDTree(
            positions.astype("float64"),
            left_edge=self.ds.domain_left_edge,
            right_edge=self.ds.domain_right_edge,
            periodic=np.array(self.ds.periodicity),
            leafsize=2 * int(self.ds._num_neighbors),
            data_version=self.ds._file_hash,
        )
        if fname is not None:
            self._kdtree.save(fname)

    @property
    def kdtree(self):
        if hasattr(self, "_kdtree"):
            return self._kdtree
        ds = self.ds
        if getattr(ds, "kdtree_filename", None) is None:
            if os.path.exists(ds.parameter_filename):
                fname = ds.parameter_filename + ".kdtree"
            else:
                # we don't want to write to disk for in-memory data
                fname = None
        else:
            fname = ds.kdtree_filename
        self._generate_kdtree(fname)
        return self._kdtree


class QMCFile(ParticleFile):
    def __init__(self, ds, io, filename, file_id, range=None):
        super().__init__(ds, io, filename, file_id, range)


class QMCDataset(ParticleDataset):
    _index_class = QMCIndex
    _file_class = QMCFile
    _field_info_class = QMCFieldInfo
    _particle_coordinates_name = "Coordinates"
    _suffix = ""
    _sph_ptypes = ("io",)
    _num_neighbors = 8

    def __init__(self, filename, dataset_type="qmc", unit_system=qmc_unit_sys):
        if self._instantiated:
            return
        self.domain_left_edge = None
        self.domain_right_edge = None
        self.domain_dimensions = np.ones(3, "int32")
        self.periodicity = (True, True, True)
        self.gen_hsmls = True
        self._unit_system=unit_system
        super().__init__(filename, dataset_type, unit_system=unit_system)
        self.add_sph_fields()

    def __repr__(self):
        return os.path.basename(self.parameter_filename).split(".")[0]

    def _parse_parameter_file(self):
        self.unique_identifier = uuid.uuid4()
        self.parameters = {}
        self.domain_left_edge = np.array([0., 0., 0.], np.float)
        self.domain_right_edge = np.array([1., 1., 1.], np.float)
        self.dimensionality = 3
        self.domain_dimensions = np.array([1, 1, 1], np.int)
        self.periodicity = (True, True, True)
        self.current_time = 0.0
        self.cosmological_simulation = 0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0
        self.geometry = "cartesian"
        self.filename_template = self.parameter_filename
        self.file_count = 1

    def _set_code_unit_attributes(self):
        self.length_unit = self.quan(1.0, self._unit_system["length"])
        self.mass_unit = self.quan(1.0, self._unit_system["mass"])
        self.time_unit = self.quan(1.0, self._unit_system["time"])
        self.temperature_unit = self.quan(1.0, self._unit_system["temperature"])

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        try:
            file_format = filetype(args[0], guess=False)
        except UnknownFileTypeError:
            return False
        return True

    def add_sph_fields(self, n_neighbors=8, kernel="cubic", sph_ptype="io"):
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
        # Read basic fields
        ad = self.all_data()
        pos = ad[sph_ptype, "particle_position"].to(l_unit).d
        mass = ad[sph_ptype, "particle_mass"].to(m_unit).d
        # Construct k-d tree
        kdtree = PyKDTree(
            pos.astype("float64"),
            left_edge=self.domain_left_edge.to_value(l_unit),
            right_edge=self.domain_right_edge.to_value(l_unit),
            periodic=self.periodicity,
            leafsize=2 * int(n_neighbors),
        )
        order = np.argsort(kdtree.idx)
        def exists(fname):
            if (sph_ptype, fname) in self.derived_field_list:
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
            hsml = self.index.io._generate_smoothing_length(self.index)
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
        # Add fields
        self._sph_ptypes = (sph_ptype,)
        self.index.update_data(data)
        self.num_neighbors = n_neighbors
