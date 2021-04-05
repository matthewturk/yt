"""
Skeleton-specific fields



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.fields.field_info_container import \
    FieldInfoContainer

# We need to specify which fields we might have in our dataset.  The field info
# container subclass here will define which fields it knows about.  There are
# optionally methods on it that get called which can be subclassed.

rho_units = "code_mass / code_length**3"

class IHarmFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ( "RHO", (rho_units, ["density"], None)),
        ( "UU", (rho_units, [], None)),
        ( "U1", (rho_units, [], None)),
        ( "U2", (rho_units, [], None)),
        ( "U3", (rho_units, [], None)),
        ( "B1", (rho_units, [], None)),
        ( "B2", (rho_units, [], None)),
        ( "B3", (rho_units, [], None)),
    )

    def __init__(self, ds, field_list):
        super(IHarmFieldInfo, self).__init__(ds, field_list)
        """ TODO do stuff here ?
        """
        # If you want, you can check self.field_list

    def setup_fluid_fields(self):
        # Here we do anything that might need info about the dataset.
        # You can use self.alias, self.add_output_field (for on-disk fields)
        # and self.add_field (for derived fields).
        pass

