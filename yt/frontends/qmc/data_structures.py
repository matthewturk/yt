import os
import uuid

from ase.io.formats import filetype
from ase.io.formats import UnknownFileTypeError
import numpy as np

from yt.data_objects.static_output import ParticleDataset
from yt.data_objects.static_output import ParticleFile
from yt.geometry.particle_geometry_handler import ParticleIndex

from .definitions import qmc_unit_sys
from .fields import QMCFieldInfo


class QMCIndex(ParticleIndex):
    def __init__(self, ds, dataset_type):
        super().__init__(ds, dataset_type)
        self._initialize_index()

    def _initialize_index(self):
        super()._initialize_index()

    def _initialize_frontend_specific(self):
        super()._initialize_frontend_specific()


class QMCFile(ParticleFile):
    def __init__(self, ds, io, filename, file_id, range=None):
        super().__init__(ds, io, filename, file_id, range)


class QMCDataset(ParticleDataset):
    _index_class = QMCIndex
    _file_class = QMCFile
    _field_info_class = QMCFieldInfo
    _particle_coordinates_name = "Coordinates"
    _suffix = ""

    def __init__(self, filename, dataset_type="qmc", unit_system=qmc_unit_sys):
        if self._instantiated:
            return
        self.domain_left_edge = None
        self.domain_right_edge = None
        self.domain_dimensions = np.ones(3, "int32")
        self.periodicity = (True, True, True)
        self._unit_system=unit_system
        super().__init__(filename, dataset_type, unit_system=unit_system)

    def __repr__(self):
        return os.path.basename(self.parameter_filename).split(".")[0]

    def _parse_parameter_file(self):
        self.unique_identifier = uuid.uuid4()
        self.parameters = {}
        self.domain_left_edge = np.array([0., 0., 0.], np.float)
        self.domain_left_edge = np.array([1., 1., 1.], np.float)
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
            file_format = filetype(args[0])
        except UnknownFileTypeError:
            return False
        return True
