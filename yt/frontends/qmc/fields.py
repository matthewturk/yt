from yt.fields.field_info_container import FieldInfoContainer


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
