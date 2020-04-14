import os
import pkg_resources
from numbers import Number as numeric_type

import numpy as np
import yaml

from yt.fields.derived_field import NullFunc, OtherFieldInfo
from yt.fields.field_info_container import \
    FieldInfoContainer
from yt.fields.particle_fields import \
    particle_deposition_functions, \
    particle_vector_functions, \
    particle_scalar_functions, \
    standard_particle_fields
from yt.funcs import mylog
from yt.units.dimensions import dimensionless
from yt.units.unit_object import Unit
from yt.utilities.physical_constants import \
    me, \
    mp

vel_units = "code_velocity"

known_species_names = {
    'HI'      : 'H_p0',
    'HII'     : 'H_p1',
    'HeI'     : 'He_p0',
    'HeII'    : 'He_p1',
    'HeIII'   : 'He_p2',
    'H2I'     : 'H2_p0',
    'H2II'    : 'H2_p1',
    'HM'      : 'H_m1',
    'HeH'     : 'HeH_p0',
    'DI'      : 'D_p0',
    'DII'     : 'D_p1',
    'HDI'     : 'HD_p0',
    'Electron': 'El',
    'OI'      : 'O_p0',
    'OII'     : 'O_p1',
    'OIII'    : 'O_p2',
    'OIV'     : 'O_p3',
    'OV'      : 'O_p4',
    'OVI'     : 'O_p5',
    'OVII'    : 'O_p6',
    'OVIII'   : 'O_p7',
    'OIX'     : 'O_p8',
}

NODAL_FLAGS = {
    'BxF': [1, 0, 0],
    'ByF': [0, 1, 0],
    'BzF': [0, 0, 1],
    'Ex': [0, 1, 1],
    'Ey': [1, 0, 1],
    'Ez': [1, 1, 0],
    'AvgElec0': [0, 1, 1],
    'AvgElec1': [1, 0, 1],
    'AvgElec2': [1, 1, 0],
}


def _load_other_fields(frontend):
    """
    Data stored as a dictionary. The keys are "known_other_fields"
    and "known_particle_fields". The value for each key is a list
    of dictionaries. Each of these inner dictionaries contains the
    field attributes.
    """
    file_name = frontend + "_known_other_fields.yaml"
    file_name = os.path.join("frontends", frontend, file_name)
    file_object = pkg_resources.resource_stream("yt", file_name)
    known_field_data = yaml.safe_load(file_object)
    known_other_fields = {} 
    known_particle_fields = {}
    for field_type, field_list in known_field_data.items():
        for field in field_list:
            other_field = OtherFieldInfo(**field)
            if field_type == "known_other_fields":
                known_other_fields[other_field.code_name] = other_field
            elif field_type == "known_particle_fields":
                known_particle_fields[other_field.code_name] = other_field
            else:
                raise KeyError
    return known_other_fields, known_particle_fields


class EnzoFieldInfo(FieldInfoContainer):
    def __init__(self, ds, field_list):
        self.known_other_fields, self.known_particle_fields = _load_other_fields("enzo")
        hydro_method = ds.parameters.get("HydroMethod", None)
        if hydro_method is None:
            hydro_method = ds.parameters["Physics"]["Hydro"]["HydroMethod"]
        if hydro_method == 2:
            sl_left = slice(None,-2,None)
            sl_right = slice(1,-1,None)
            div_fac = 1.0
        else:
            sl_left = slice(None,-2,None)
            sl_right = slice(2,None,None)
            div_fac = 2.0
        slice_info = (sl_left, sl_right, div_fac)
        # Here we do the work done in FieldInfoContainer.__init__ until
        # all of the frontends have been ported over and the changes
        # can be put back into that method. This is to isolate EnzoFieldInfo
        # from the rest of the frontend FieldInfo classes and not break
        # anything
        self._show_field_errors = []
        self.ds = ds
        # Now we start setting things up.
        self.field_list = field_list
        self.slice_info = slice_info
        self.field_aliases = {}
        self.species_names = []
        self.setup_fluid_aliases()
        # super(EnzoFieldInfo, self).__init__(ds, field_list, slice_info)
        # setup nodal flag information
        for field in NODAL_FLAGS:
            if ('enzo', field) in self:
                finfo = self['enzo', field]
                finfo.nodal_flag = np.array(NODAL_FLAGS[field])

    def setup_fluid_aliases(self, ftype='gas'):
        """
        Overloaded from FieldInfoContainer's version until all frontends
        have been ported over to using pydantic.
        """
        for field in sorted(self.field_list):
            if not isinstance(field, tuple):
                raise RuntimeError
            if field[0] in self.ds.particle_types:
                continue
            # See if field is in known_other_fields. If not, use a
            # default empty attribute tuple
            args = self.known_other_fields.get(field[1], ("", [], None))
            if isinstance(args, tuple):
                units, aliases, display_name = args
            # If the field is in known_other_fields, the value returned
            # by get() is an OtherFieldInfo instance, so we unpack it
            elif isinstance(args, OtherFieldInfo):
                units = args.units
                aliases = args.aliases
                display_name = args.display_name
                args = [args.units]
            else:
                raise ValueError
            # We allow field_units to override this.  First we check if the
            # field *name* is in there, then the field *tuple*.
            units = self.ds.field_units.get(field[1], units)
            units = self.ds.field_units.get(field, units)
            if not isinstance(units, str) and args[0] != "":
                units = "((%s)*%s)" % (args[0], units)
            if isinstance(units, (numeric_type, np.number, np.ndarray)) and \
                args[0] == "" and units != 1.0:
                mylog.warning("Cannot interpret units: %s * %s, " +
                              "setting to dimensionless.", units, args[0])
                units = ""
            elif units == 1.0:
                units = ""
            self.add_output_field(field, sampling_type="cell",units = units,
                                  display_name = display_name)
            for alias in aliases:
                self.alias((ftype, alias), field)

    def add_species_field(self, species):
        # This is currently specific to Enzo.  Hopefully in the future we will
        # have deeper integration with other systems, such as Dengo, to provide
        # better understanding of ionization and molecular states.
        #
        # We have several fields to add based on a given species field.  First
        # off, we add the species field itself.  Then we'll add a few more
        # items...
        #
        self.add_output_field(("enzo", "%s_Density" % species),
                              sampling_type="cell",
                              take_log=True,
                              units="code_mass/code_length**3")
        yt_name = known_species_names[species]
        # don't alias electron density since mass is wrong
        if species != "Electron":
            self.alias(("gas", "%s_density" % yt_name),
                       ("enzo", "%s_Density" % species))

    def setup_species_fields(self):
        species_names = [fn.rsplit("_Density")[0] for ft, fn in
                         self.field_list if fn.endswith("_Density")]
        species_names = [sp for sp in species_names
                         if sp in known_species_names]
        def _electron_density(field, data):
            return data["Electron_Density"] * (me/mp)
        self.add_field(("gas", "El_density"),
                       sampling_type="cell",
                       function = _electron_density,
                       units = self.ds.unit_system["density"])
        for sp in species_names:
            self.add_species_field(sp)
            self.species_names.append(known_species_names[sp])
        self.species_names.sort()  # bb #1059

    def setup_fluid_fields(self):
        from yt.fields.magnetic_field import \
            setup_magnetic_field_aliases
        # Now we conditionally load a few other things.
        params = self.ds.parameters
        multi_species = params.get("MultiSpecies", None)
        dengo = params.get("DengoChemistryModel", 0)
        if multi_species is None:
            multi_species = params["Physics"]["AtomicPhysics"]["MultiSpecies"]
        if multi_species > 0 or dengo == 1:
            self.setup_species_fields()
        self.setup_energy_field()
        setup_magnetic_field_aliases(self, "enzo", ["B%s" % ax for ax in "xyz"])

    def setup_energy_field(self):
        unit_system = self.ds.unit_system
        # We check which type of field we need, and then we add it.
        ge_name = None
        te_name = None
        params = self.ds.parameters
        multi_species = params.get("MultiSpecies", None)
        if multi_species is None:
            multi_species = params["Physics"]["AtomicPhysics"]["MultiSpecies"]
        hydro_method = params.get("HydroMethod", None)
        if hydro_method is None:
            hydro_method = params["Physics"]["Hydro"]["HydroMethod"]
        dual_energy = params.get("DualEnergyFormalism", None)
        if dual_energy is None:
            dual_energy = params["Physics"]["Hydro"]["DualEnergyFormalism"]
        if ("enzo", "Gas_Energy") in self.field_list:
            ge_name = "Gas_Energy"
        elif ("enzo", "GasEnergy") in self.field_list:
            ge_name = "GasEnergy"
        if ("enzo", "Total_Energy") in self.field_list:
            te_name = "Total_Energy"
        elif ("enzo", "TotalEnergy") in self.field_list:
            te_name = "TotalEnergy"

        if hydro_method == 2:
            self.add_output_field(("enzo", te_name),
                                  sampling_type="cell",
                                  units="code_velocity**2")
            self.alias(("gas", "thermal_energy"), ("enzo", te_name))
            def _ge_plus_kin(field, data):
                ret = data[te_name] + 0.5*data["velocity_x"]**2.0
                if data.ds.dimensionality > 1:
                    ret += 0.5*data["velocity_y"]**2.0
                if data.ds.dimensionality > 2:
                    ret += 0.5*data["velocity_z"]**2.0
                return ret
            self.add_field(
                ("gas", "total_energy"), sampling_type="cell",
                function = _ge_plus_kin,
                units = unit_system["specific_energy"])
        elif dual_energy == 1:
            self.add_output_field(
                ("enzo", te_name),
                sampling_type="cell",
                units = "code_velocity**2")
            self.alias(
                ("gas", "total_energy"),
                ("enzo", te_name),
                units = unit_system["specific_energy"])
            self.add_output_field(
                ("enzo", ge_name), sampling_type="cell",
                units="code_velocity**2")
            self.alias(
                ("gas", "thermal_energy"),
                ("enzo", ge_name),
                units = unit_system["specific_energy"])
        elif hydro_method in (4, 6):
            self.add_output_field(
                ("enzo", te_name),
                sampling_type="cell",
                units="code_velocity**2")
            # Subtract off B-field energy
            def _sub_b(field, data):
                ret = data[te_name] - 0.5*data["velocity_x"]**2.0
                if data.ds.dimensionality > 1:
                    ret -= 0.5*data["velocity_y"]**2.0
                if data.ds.dimensionality > 2:
                    ret -= 0.5*data["velocity_z"]**2.0
                ret -= data["magnetic_energy"]/data["density"]
                return ret
            self.add_field(
                ("gas", "thermal_energy"),
                sampling_type="cell",
                function=_sub_b,
                units=unit_system["specific_energy"])
        else: # Otherwise, we assume TotalEnergy is kinetic+thermal
            self.add_output_field(
                ("enzo", te_name),
                sampling_type="cell",
                units = "code_velocity**2")
            self.alias(
                ("gas", "total_energy"),
                ("enzo", te_name),
                units = unit_system["specific_energy"])
            def _tot_minus_kin(field, data):
                ret = data[te_name] - 0.5*data["velocity_x"]**2.0
                if data.ds.dimensionality > 1:
                    ret -= 0.5*data["velocity_y"]**2.0
                if data.ds.dimensionality > 2:
                    ret -= 0.5*data["velocity_z"]**2.0
                return ret
            self.add_field(
                ("gas", "thermal_energy"),
                sampling_type="cell",
                function=_tot_minus_kin,
                units=unit_system["specific_energy"])
        if multi_species == 0 and 'Mu' in params:
            def _mean_molecular_weight(field, data):
                return params["Mu"]*data['index', 'ones']

            self.add_field(
                ("gas", "mean_molecular_weight"),
                sampling_type="cell",
                function=_mean_molecular_weight,
                units="")

            def _number_density(field, data):
                return data['gas', 'density']/(mp*params['Mu'])

            self.add_field(
                ("gas", "number_density"),
                sampling_type="cell",
                function=_number_density,
                units=unit_system["number_density"])

    def setup_particle_fields(self, ptype):
        def _age(field, data):
            return data.ds.current_time - data["creation_time"]
        self.add_field((ptype, "age"),
                       sampling_type="particle",
                       function=_age,
                       units = "yr")
        #super(EnzoFieldInfo, self).setup_particle_fields(ptype)
        self._overload_setup_particle_fields(ptype)

    def _overload_setup_particle_fields(self, ptype, ftype='gas', num_neighbors=64):
        """
        Overloaded from FieldInfoContainer's version until all frontends
        have been ported over to using pydantic.
        """
        skip_output_units = ("code_length",)
        for f in sorted(self.known_particle_fields):
            units = self.known_particle_fields[f].units
            aliases = self.known_particle_fields[f].aliases
            dn = self.known_particle_fields[f].display_name
            units = self.ds.field_units.get((ptype, f), units)
            output_units = units
            if (f in aliases or ptype not in self.ds.particle_types_raw) and \
                units not in skip_output_units:
                u = Unit(units, registry = self.ds.unit_registry)
                if u.dimensions is not dimensionless:
                    output_units = str(self.ds.unit_system[u.dimensions])
            if (ptype, f) not in self.field_list:
                continue
            self.add_output_field((ptype, f), sampling_type="particle",
                units = units, display_name = dn, 
                output_units = output_units)
            for alias in aliases:
                self.alias((ptype, alias), (ptype, f), units = output_units)
        # We'll either have particle_position or particle_position_[xyz]
        if (ptype, "particle_position") in self.field_list or \
           (ptype, "particle_position") in self.field_aliases:
            particle_scalar_functions(ptype,
                   "particle_position", "particle_velocity",
                   self)
        else:
            # We need to check to make sure that there's a "known field" that
            # overlaps with one of the vector fields.  For instance, if we are
            # in the Stream frontend, and we have a set of scalar position
            # fields, they will overlap with -- and be overridden by -- the
            # "known" vector field that the frontend creates.  So the easiest
            # thing to do is to simply remove the on-disk field (which doesn't
            # exist) and replace it with a derived field.
            if (ptype, "particle_position") in self and \
                 self[ptype, "particle_position"]._function == NullFunc:
                self.pop((ptype, "particle_position"))
            particle_vector_functions(ptype,
                    ["particle_position_%s" % ax for ax in 'xyz'],
                    ["particle_velocity_%s" % ax for ax in 'xyz'],
                    self)
        particle_deposition_functions(ptype, "particle_position",
            "particle_mass", self)
        standard_particle_fields(self, ptype)
        # Now we check for any leftover particle fields
        for field in sorted(self.field_list):
            if field in self: continue
            if not isinstance(field, tuple):
                raise RuntimeError
            if field[0] not in self.ds.particle_types:
                continue
            self.add_output_field(field, sampling_type="particle",
                                  units = self.ds.field_units.get(field, ""))
        self.setup_smoothed_fields(ptype, 
                                   num_neighbors=num_neighbors,
                                   ftype=ftype)
