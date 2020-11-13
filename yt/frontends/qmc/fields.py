from yt.fields.field_info_container import FieldInfoContainer

from .definitions import elementRegister


class QMCFieldInfo(FieldInfoContainer):
    known_other_fields = ()

    known_particle_fields = (
        ("positions", ("code_length", [], None)),
        ("numbers", ("", [], None)),
    )

    def __init__(self, ds, field_list, slice_info=None):
        super(QMCFieldInfo, self).__init__(ds, field_list, slice_info=slice_info)

    def setup_particle_fields(self, ptype, *args, **kwargs):
        super().setup_particle_fields(ptype, *args, **kwargs)
        self._setup_masses()

    def _setup_masses(self):
        """
        Maps the element numbers from the numbers field to element
        masses.
        """
        def _atomic_mass(field, data):
            return elementRegister[str(data[("io", "numbers")])]["mass"]
        self.add_field(
            ("io", "mass"),
            sampling_type="particle",
            function=_atomic_mass,
            units="amu",
        )
