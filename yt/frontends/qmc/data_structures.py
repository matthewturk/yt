import os
import uuid

import numpy as np
from ase.io.formats import UnknownFileTypeError, filetype

from yt.data_objects.static_output import ParticleDataset, ParticleFile
from yt.geometry.particle_geometry_handler import ParticleIndex
from yt.units import dimensions
from yt.units.unit_registry import UnitRegistry
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
        self.io._generate_smoothing_length(self)
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
        particle_positions = []
        for data_file in self.data_files:
            for _, ppos in self.io._yield_coordinates(
                data_file, needed_ptype=self.ds._sph_ptypes[0]
            ):
                particle_positions.append(ppos)
        if particle_positions == []:
            self._kdtree = None
            return
        particle_positions = np.concatenate(particle_positions)
        mylog.info("Allocating KDTree for %s particles", particle_positions.shape[0])
        self._kdtree = PyKDTree(
            particle_positions.astype("float64"),
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
    default_kernel_name = "constant"
    _index_class = QMCIndex
    _file_class = QMCFile
    _field_info_class = QMCFieldInfo
    _particle_coordinates_name = "Coordinates"
    _suffix = ""
    _sph_ptypes = ("io",)
    _num_neighbors = 8

    def __init__(
        self, filename, dataset_type="qmc", kernel_name=None, unit_system=qmc_unit_sys
    ):
        if self._instantiated:
            return
        self.domain_left_edge = None
        self.domain_right_edge = None
        self.domain_dimensions = np.ones(3, "int32")
        self._periodicity = (True, True, True)
        self.gen_hsmls = True
        self._unit_system = unit_system
        if kernel_name is None:
            self.kernel_name = self.default_kernel_name
        else:
            self.kernel_name = kernel_name
        super().__init__(filename, dataset_type, unit_system=unit_system)

    def _create_unit_registry(self, unit_system):
        # Overwrites Dataset's _create_unit_registry method to use the
        # custom unit system for qmc and remove astro-specific units
        self.unit_registry = UnitRegistry(unit_system=unit_system)
        self.unit_registry.add("code_length", 1.0, dimensions.length)
        self.unit_registry.add("code_mass", 1.0, dimensions.mass)
        self.unit_registry.add("code_density", 1.0, dimensions.density)
        self.unit_registry.add("code_time", 1.0, dimensions.time)
        self.unit_registry.add("code_temperature", 1.0, dimensions.temperature)
        self.unit_registry.add("code_velocity", 1.0, dimensions.velocity)
        self.unit_registry.add("code_pressure", 1.0, dimensions.pressure)
        self.unit_registry.add(
            "code_specific_energy", 1.0, dimensions.energy / dimensions.mass
        )
        self.unit_registry.add("h", 1.0, dimensions.dimensionless, r"h")

    def __repr__(self):
        return os.path.basename(self.parameter_filename).split(".")[0]

    def _parse_parameter_file(self):
        self.unique_identifier = uuid.uuid4()
        self.parameters = {}
        self.domain_left_edge = np.array([0.0, 0.0, 0.0], np.float)
        self.domain_right_edge = np.array([1.0, 1.0, 1.0], np.float)
        self.dimensionality = 3
        self.domain_dimensions = np.array([1, 1, 1], np.int)
        self._periodicity = (True, True, True)
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
            if filetype(args[0], guess=False):
                return True
        except UnknownFileTypeError:
            return False
