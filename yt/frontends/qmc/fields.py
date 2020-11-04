from yt.fields.field_info_container import FieldInfoContainer


class QMCFieldInfo(FieldInfoContainer):
    known_other_fields = ()

    known_particle_fields = (
        ("positions", ("code_length", [], None)),
        ("numbers", ("", [], None)),
    )

    def setup_particle_fields(self, ptype, *args, **kwargs):
        pass

    def setup_fluid_index_fields(self):
        pass
