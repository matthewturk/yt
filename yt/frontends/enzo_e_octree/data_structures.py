from __future__ import annotations  # noqa: I001

import os
from collections.abc import Collection, Iterator
from itertools import accumulate
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

from yt.arraytypes import blankRecordArray
from yt.data_objects.index_subobjects.octree_subset import OctreeSubset
from yt.data_objects.static_output import Dataset
from yt.frontends.enzo.misc import cosmology_get_units
from yt.frontends.enzo_e_octree.fields import EnzoEFieldInfo
from yt.frontends.enzo_e_octree.misc import (block_pos, bname_from_pos,
                                             get_bf_path, get_block_info,
                                             get_listed_subparam,
                                             get_min_level, get_root_blocks,
                                             nested_dict_get, remove_ext)
from yt.funcs import setdefaultattr
from yt.geometry.geometry_handler import YTDataChunk
from yt.geometry.oct_container import EnzoEOctreeContainer
from yt.geometry.oct_geometry_handler import OctreeIndex
from yt.utilities.cosmology import Cosmology
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.on_demand_imports import _libconf as libconf

if TYPE_CHECKING:
    import h5py
else:
    from yt.utilities.on_demand_imports import _h5py as h5py


# TODO: Should any of this be rewritten in cython (prob later for performance)
# TODO: Implement ghost zones
# TODO: Do I need to support blocks with non cubic non divisible nzs? (ie: commit
#       commit my nzone array patch?). All the test datasets work with it.


class EnzoEDomainFile:
    # equivalent to a block list file
    ds: EnzoEOctreeDataset
    index: EnzoEOctreeHierarchy
    nocts: int
    h5fname: str
    min_level: int
    cell_count: int
    levels: np.ndarray
    level_counts: list[int]
    level_inds: list[int]
    oct_handler: EnzoEOctreeContainer

    def __init__(
        self,
        ds: EnzoEOctreeDataset,
        oct_handler: EnzoEOctreeContainer,
        bnames: Collection[str],
        h5fname: str,
        levels: np.ndarray,
        level_counts: list[int],
        domain_id: int,
    ) -> None:
        self.ds = ds
        self.oct_handler = oct_handler

        self.levels = levels
        self.nocts = sum(len(levels) for levels in levels)
        self.cell_count = self.ds.cells_per_oct * self.nocts

        self.level_counts = level_counts
        self.level_inds = list(accumulate(level_counts))
        self.levels = levels

        self.h5fname = h5fname
        self.domain_id = domain_id

    def init_octs(self):
        # The order of the blocks in the levels array doesn't matter
        # but it has to stay consistent between the levels array
        # and when it is added
        offset = 0
        for i, level in enumerate(self.levels):
            # This domain does not contain further refined levels
            if level.size == 0:
                break
            self.oct_handler.add(
                self.domain_id, i, level[:, 3:], level[:, :3], file_ind_offset=offset
            )
            offset += len(level)

    def included(self, selector) -> bool:
        if getattr(selector, "domain_ind", None) is not None:
            raise NotImplementedError
            # return selector.domain_ind == self.domain_id
        domain_ids = self.oct_handler.domain_identify(selector)
        return bool(domain_ids)


class EnzoEOctreeHierarchy(OctreeIndex):
    max_level: int
    min_level: int
    domains: list[EnzoEDomainFile]
    comm: Any
    oct_handler: EnzoEOctreeContainer

    def __init__(self, ds, dataset_type="enzo_e_octree"):
        self.dataset_type = dataset_type
        super().__init__(ds, dataset_type)

    def _detect_output_fields(self) -> None:
        self.field_list = []
        # Do this only on the root processor to save disk work.
        if self.comm.rank in (0, None):
            # Just check the first grid.
            blist: RawBlockList = self.ds.domain_blocks[0]
            field_list, ptypes = self.io._read_field_names(
                blist.bnames[0], blist.h5fname
            )
            mylog.debug("Block %s has: %s", blist.bnames[0], field_list)
            sample_pfields = self.io.sample_pfields
        else:
            field_list = None
            ptypes = None
            sample_pfields = None
        self.field_list = list(self.comm.mpi_bcast(field_list))
        self.dataset.particle_types = list(self.comm.mpi_bcast(ptypes))
        self.dataset.particle_types_raw = self.dataset.particle_types[:]
        self.io.sample_pfields = self.comm.mpi_bcast(sample_pfields)

    def _initialize_oct_handler(self) -> None:
        self.max_level = self.ds.max_level
        self.min_level = self.ds.min_level

        self.oct_handler = EnzoEOctreeContainer(
            self.ds.domain_dimensions,
            self.ds.domain_left_edge,
            self.ds.domain_right_edge,
            self.ds.nz,
        )

        domains: list[EnzoEDomainFile] = []
        self.domains = domains
        # Numpy str methods might be able to be used to speed this up
        for i, dom in enumerate(self.ds.domain_blocks, 1):
            levels: list[list[np.ndarray]] = [[] for _ in range(self.max_level + 1)]
            for bname in dom.bnames:
                pos, level = block_pos(bname, -self.ds.min_level)
                if pos is not None and self.max_level >= level >= 0:
                    levels[level].append(pos)
            np_levels = [np.asarray(lvl) for lvl in levels]

            level_inds = [len(lvl) for lvl in levels]

            domains.append(
                EnzoEDomainFile(
                    self.ds,
                    self.oct_handler,
                    dom.bnames,
                    dom.h5fname,
                    np_levels,
                    level_inds,
                    i,
                )
            )

        nocts = [dom.nocts for dom in domains]
        self.num_grids = sum(nocts)
        self.oct_handler.allocate_domains(nocts, self.ds.domain_dimensions.prod())
        for dom in domains:
            dom.init_octs()

    def _identify_base_chunk(self, dobj) -> None:
        if getattr(dobj, "_chunk_info", None) is None:
            domains = [dom for dom in self.domains if dom.included(dobj.selector)]
            base_region = getattr(dobj, "base_region", dobj)
            if len(domains) > 1:
                mylog.debug("Identified %s intersecting domains", len(domains))
            subsets = [
                EnzoESubset(
                    base_region,
                    domain,
                    self.ds,
                    self.ds.nz,
                    num_ghost_zones=dobj._num_ghost_zones,
                )
                for domain in domains
            ]

            dobj._chunk_info = subsets
        dobj._current_chunk = YTDataChunk(dobj, "all", dobj._chunk_info)

    def _chunk_all(self, dobj) -> Iterator[YTDataChunk]:
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        yield YTDataChunk(dobj, "all", oobjs)

    #     def _chunk_spatial(self, dobj, ngz, sort=None, preload_fields=None):
    #         sobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
    #         for og in sobjs:
    #             if ngz > 0:
    #                 # TODO: Is this right for enzo-e?
    #                 g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
    #             else:
    #                 g = og
    #             yield YTDataChunk(dobj, "spatial", [g], None)

    def _chunk_io(self, dobj, cache=True, local_only=False):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for subset in oobjs:
            yield YTDataChunk(dobj, "io", [subset], None, cache=cache)

    def _initialize_level_stats(self):
        desc = {"names": ["num_blocks", "level", "num_cells"], "formats": ["int64"] * 3}
        max_level = self.dataset.max_level + 1
        self.level_stats = blankRecordArray(desc, max_level)
        self.level_stats["level"] = list(range(max_level))
        self.level_stats["num_blocks"] = [0 for i in range(max_level)]
        self.level_stats["num_cells"] = [0 for i in range(max_level)]
        for level in range(max_level):
            num_blocks = sum(len(dom.levels[level]) for dom in self.domains)
            self.level_stats[level]["num_blocks"] = num_blocks
            self.level_stats[level]["num_cells"] = num_blocks * self.ds.cells_per_oct

    def print_stats(self):
        """
        Prints out (stdout) relevant information about the simulation

        This function prints information based on the fluid on the grids,
        and therefore does not work for DM only runs.
        """

        self._initialize_level_stats()

        header = "{:>3}\t{:>14}\t{:>14}".format("level", "# blocks", "# cells")
        print(header)
        print(f"{len(header.expandtabs()) * '-'}")
        for level in range(self.dataset.max_level + 1):
            print(
                "% 3i\t% 14i\t% 14i"
                % (
                    level,
                    self.level_stats["num_blocks"][level],
                    self.level_stats["num_cells"][level],
                )
            )
        print("-" * 38)
        print(
            "   \t% 14i\t% 5i"
            % (
                self.level_stats["num_blocks"].sum(),
                self.level_stats["num_cells"].sum(),
            )
        )
        print("\n")

        dx = self.get_smallest_dx()
        try:
            print(f"z = {self.dataset.current_redshift:0.8f}")
        except Exception:
            pass
        print(
            "t = {:0.8e} = {:0.8e} s = {:0.8e} years".format(
                self.ds.current_time.in_units("code_time"),
                self.ds.current_time.in_units("s"),
                self.ds.current_time.in_units("yr"),
            )
        )
        print("\nSmallest Block:")
        for item in ("Mpc", "pc", "AU", "cm"):
            print(f"\tWidth: {dx.in_units(item):0.3e}")


# Each subset is only associated with *one* domain
class EnzoESubset(OctreeSubset):
    domain: EnzoEDomainFile
    ds: EnzoEOctreeDataset

    def __init__(
        self,
        base_region,
        domain,
        ds,
        nz,
        num_ghost_zones=0,
    ):
        super().__init__(base_region, domain, ds, nz, num_ghost_zones)
        if num_ghost_zones > 0:
            # TODO: Implement for ghost zones
            raise NotImplementedError
            # if not all(ds.periodicity):
            #     # TODO: For enzo-e, is this accurate?
            #     mylog.warning(
            #         "Ghost zones will wrongly assume the domain to be periodic."
            #     )
            # # Create a base domain *with no self._base_domain.fwidth
            # base_domain = RAMSESDomainSubset(ds.all_data(), domain, ds, num_zones)
            # self._base_domain = base_domain
        elif num_ghost_zones < 0:
            raise RuntimeError(
                f"Cannot initialize a domain subset with a negative number "
                f"of ghost zones, was called with num_ghost_zones={num_ghost_zones}"
            )
        self.particle_count = None

    def fill(
        self, f: h5py.File, fields: Collection[str], selector
    ) -> dict[str, np.float64]:
        # TODO: ghost zones
        # TODO: Rewrite transpose here, accounting for partial selection (non all selectors)
        return self._fill_no_ghostzones(f, fields, selector)

    # From my understanding, the Enzo-E block names are in xyz order,
    # and are sorted in the block list according to c ordering.
    # But the data actually expects to be loaded in fortran order.
    # This leads to axis_order being ('z', 'y', 'x'), the reverses, and the transposes
    def _fill_no_ghostzones(
        self, f: h5py.File, fields: Collection[str], selector
    ) -> dict[str, np.ndarray]:
        # TODO: Rewrite tight loop in cython

        # file_inds: for each cell, the oct they're associated with within that file
        # cell_inds: the index of the cell it actually is in the oct
        # levels: the levels of the cells
        levels, cell_inds, file_inds = self.oct_handler.file_index_octs(
            selector, self.domain_id
        )

        if levels.size == 0:
            return {f: np.empty(0, dtype=np.float64) for f in fields}

        _, oct_inds_i = np.unique(file_inds, return_index=True)

        src: dict[str, np.ndarray] = {
            f: np.zeros((file_inds.max() + 1, self.ds.cells_per_oct), dtype=np.float64)
            for f in fields
        }
        dest: dict[str, np.ndarray] = {
            f: np.zeros(levels.shape[0], dtype=np.float64) for f in fields
        }

        # TODO: Implement nodal fields
        fields = [f for f in fields if not hasattr(f, "nodal_flag")]

        for oii in oct_inds_i:
            lvl = levels[oii]
            fi = file_inds[oii]
            fi_offset = self.domain.level_inds[lvl - 1] if lvl > 0 else 0

            bname = self.ds.bname_from_pos(self.domain.levels[lvl][fi - fi_offset], lvl)
            field_data = f[bname]

            for field in fields:
                fname = f"field_{field[1]}"
                raw_data = field_data[fname]
                data = np.empty(raw_data.shape, dtype="float64")
                raw_data.read_direct(data)
                data = data[self.ds.base_slice]

                if self.ds.dimensionality == 2:
                    data.shape += (1,)
                for ic, nzd in enumerate(data.shape):
                    # the oct selection code currently can't handle arbritary
                    # nz, so we repeat the cells in the smaller axises,
                    # which are factors, to be the same size as the
                    # larger axis
                    if nzd != self.nz:
                        data = np.repeat(data, self.nz // nzd, axis=ic)

                oct_cell_inds = cell_inds[file_inds == fi]
                src[field][fi, oct_cell_inds] = data.T.ravel()[oct_cell_inds]

        for lvl in range(self.ds.max_level + 1):
            self.oct_handler.fill_level(lvl, levels, cell_inds, file_inds, dest, src)
        return dest

    # @property
    # def fwidth(self):
    #     fwidth = super().fwidth
    #     if self._num_ghost_zones > 0:
    #         fwidth = fwidth.reshape(-1, 8, 3)
    #         n_oct = fwidth.shape[0]
    #         # new_fwidth contains the fwidth of the oct+ghost zones
    #         # this is a constant array in each oct, so we simply copy
    #         # the oct value using numpy fancy-indexing
    #         new_fwidth = np.zeros((n_oct, self.nz**3, 3), dtype=fwidth.dtype)
    #         new_fwidth[:, :, :] = fwidth[:, 0:1, :]
    #         fwidth = new_fwidth.reshape(-1, 3)
    #     return fwidth

    # @property
    # def fcoords(self):
    #     num_ghost_zones = self._num_ghost_zones
    #     if num_ghost_zones == 0:
    #         return super().fcoords

    #     oh = self.oct_handler

    #     indices = oh.fill_index(self.selector).reshape(-1, 8)
    #     oct_inds, cell_inds = oh.fill_octcellindex_neighbours(
    #         self.selector, self._num_ghost_zones
    #     )

    #     N_per_oct = self.nz**3
    #     oct_inds = oct_inds.reshape(-1, N_per_oct)
    #     cell_inds = cell_inds.reshape(-1, N_per_oct)

    #     inds = indices[oct_inds, cell_inds]

    #     fcoords = self.ds.arr(oh.fcoords(self.selector)[inds].reshape(-1, 3), "unitary")

    #     return fcoords


class RawBlockList(NamedTuple):
    bnames: list[str]
    h5fname: str


class EnzoEOctreeDataset(Dataset):
    """
    Enzo-E-specific output, set at a fixed time.
    """

    _index_class = EnzoEOctreeHierarchy
    _field_info_class = EnzoEFieldInfo
    refine_by: ClassVar[int] = 2
    _suffixes = (".file_list", ".block_list")
    particle_types: tuple[str, ...] = ()
    particle_types_raw = None
    domain_blocks: list[RawBlockList]
    min_level: int
    max_level: int
    nz: int
    num_domains: int
    cells_per_oct: int
    dimensionality: int
    base_slice: tuple[slice, ...]
    index: EnzoEOctreeHierarchy

    def __init__(
        self,
        filename,
        dataset_type="enzo_e_octree",
        parameter_override=None,
        conversion_override=None,
        storage_filename=None,
        units_override=None,
        unit_system="cgs",
        default_species_fields=None,
    ):
        """
        This class is a stripped down class that simply reads and parses
        *filename* without looking at the index.  *dataset_type* gets passed
        to the index to pre-determine the style of data-output.  However,
        it is not strictly necessary.  Optionally you may specify a
        *parameter_override* dictionary that will override anything in the
        parameter file and a *conversion_override* dictionary that consists
        of {fieldname : conversion_to_cgs} that will override the #DataCGS.
        """
        # Can I read this from the config file?
        basedir = os.path.dirname(filename)
        self.fluid_types += ("enzoe",)
        if parameter_override is None:
            parameter_override = {}
        self._parameter_override = parameter_override
        if conversion_override is None:
            conversion_override = {}
        self._conversion_override = conversion_override
        self.storage_filename = storage_filename

        self.domain_blocks = domain_blocks = []

        with open(get_bf_path(filename, "file_list")) as file_list:
            block_fnames = file_list.readlines()
        if block_fnames[0].endswith(".h5\n"):
            # old style file_list
            with open(get_bf_path(filename, "block_list")) as f:
                bname, data_fname = next(f).strip().split()
                data_fname = f"{basedir}/{data_fname}"
                domain_blocks.append(RawBlockList([bname], data_fname))
                for line in f:
                    bname, dfname = line.split()
                    dfname = f"{basedir}/{dfname}"
                    if data_fname != dfname:
                        domain_blocks.append(RawBlockList([], dfname))
                        data_fname = dfname
                    domain_blocks[-1].bnames.append(bname)
        else:
            # new style file_list
            for fname in block_fnames[1:]:
                blfname = get_bf_path(fname, "block_list")
                h5fname = get_bf_path(fname, "h5")
                with open(blfname) as f:
                    domain_blocks.append(RawBlockList(f.readlines(), h5fname))
        domain_blocks.sort(key=lambda v: int(remove_ext(v.h5fname)[-2:]))
        self.num_domains = len(domain_blocks)

        b0 = self.domain_blocks[0].bnames[0]
        # get dimension from first block name
        level0, left0, right0 = get_block_info(b0, min_dim=0)
        self.dimensionality = left0.size
        self.domain_dimensions = get_root_blocks(b0, min_dim=self.dimensionality)[::-1]
        if self.dimensionality == 2:
            self.domain_dimensions = np.append(self.domain_dimensions, 1)
            axis_order = ("y", "x", "z")
        else:
            axis_order = ("z", "y", "x")

        Dataset.__init__(
            self,
            filename,
            dataset_type,
            units_override=units_override,
            unit_system=unit_system,
            default_species_fields=default_species_fields,
            axis_order=axis_order,
        )

    def _parse_parameter_file(self):
        """
        Parses the parameter file and establishes the various
        dictionaries.
        """
        self._periodicity = tuple(np.ones(self.dimensionality, dtype=bool))
        self.min_level = get_min_level(self.domain_blocks[0].bnames)

        lcfn = get_bf_path(self.filename, "libconfig")
        if os.path.exists(lcfn):
            with open(lcfn) as lf:
                self.parameters = libconf.load(lf)

            # Enzo-E ignores all cosmology param
            # the Physics:list parameter
            physics_list = nested_dict_get(
                self.parameters, ("Physics", "list"), default=[]
            )
            if "cosmology" in physics_list:
                self.cosmological_simulation = 1
                co_pars = [
                    "hubble_constant_now",
                    "omega_matter_now",
                    "omega_lambda_now",
                    "comoving_box_size",
                    "initial_redshift",
                ]
                co_dict = {
                    attr: nested_dict_get(
                        self.parameters, ("Physics", "cosmology", attr)
                    )
                    for attr in co_pars
                }
                for attr in ["hubble_constant", "omega_matter", "omega_lambda"]:
                    setattr(self, attr, co_dict[f"{attr}_now"])

                # Current redshift is not stored, so it's not possible
                # to set all cosmological units yet.
                # Get the time units and use that to figure out redshift.
                k = cosmology_get_units(
                    self.hubble_constant,
                    self.omega_matter,
                    co_dict["comoving_box_size"],
                    co_dict["initial_redshift"],
                    0,
                )
                setdefaultattr(self, "time_unit", self.quan(k["utim"], "s"))
                co = Cosmology(
                    hubble_constant=self.hubble_constant,
                    omega_matter=self.omega_matter,
                    omega_lambda=self.omega_lambda,
                )
            else:
                self.cosmological_simulation = 0
        else:
            self.cosmological_simulation = 0

        fh = h5py.File(self.domain_blocks[0].h5fname, "r")
        self.domain_left_edge = fh.attrs["lower"][::-1]
        self.domain_right_edge = fh.attrs["upper"][::-1]
        if "version" in fh.attrs:
            version = fh.attrs.get("version").tobytes().decode("ascii")
        else:
            version = None  # earliest recorded version is '0.9.0'
        self.parameters["version"] = version

        ablock = next(iter(fh.values()))
        self.current_time = ablock.attrs["time"][0]
        self.parameters["current_cycle"] = ablock.attrs["cycle"][0]
        gsi = ablock.attrs["enzo_GridStartIndex"]
        self.ghost_zones = gsi[0]
        self.base_slice = (
            slice(self.ghost_zones, -self.ghost_zones),
        ) * self.dimensionality
        self.grid_dimensions = ablock.attrs["enzo_GridDimension"]

        field = next(iter(ablock.values()))
        cell_dims = np.full(3, 1, dtype=np.uint8)
        for i, sh in enumerate(field[self.base_slice].shape):
            cell_dims[i] = sh
        max_dim = max(cell_dims)
        if np.any(max_dim % cell_dims != 0):
            raise ValueError(
                f"Cell dimensions per block {self.cell_dims} are not"
                f" divisible by the largest dimension {max_dim}"
            )
        self.nz = max_dim
        self.cells_per_oct = self.nz**3

        fh.close()

        if self.cosmological_simulation:
            self.current_redshift = co.z_from_t(self.current_time * self.time_unit)

        self._periodicity += (False,) * (3 - self.dimensionality)
        self._parse_fluid_prop_params()

        # inclusive
        self.max_level = (
            nested_dict_get(self.parameters, ("Adapt", "max_level"), default=1) - 1
        )

    def _parse_fluid_prop_params(self):
        """
        Parse the fluid properties.
        """

        fp_params = nested_dict_get(
            self.parameters, ("Physics", "fluid_props"), default=None
        )

        if fp_params is not None:
            # in newer versions of enzo-e, this data is specified in a
            # centralized parameter group called Physics:fluid_props
            # -  for internal reasons related to backwards compatibility,
            #    treatment of this physics-group is somewhat special (compared
            #    to the cosmology group). The parameters in this group are
            #    honored even if Physics:list does not include "fluid_props"
            self.gamma = nested_dict_get(fp_params, ("eos", "gamma"))
            de_type = nested_dict_get(
                fp_params, ("dual_energy", "type"), default="disabled"
            )
            uses_de = de_type != "disabled"
        else:
            # in older versions, these parameters were more scattered
            self.gamma = nested_dict_get(self.parameters, ("Field", "gamma"))

            uses_de = False
            for method in ("ppm", "mhd_vlct"):
                subparams = get_listed_subparam(
                    self.parameters, "Method", method, default=None
                )
                if subparams is not None:
                    uses_de = subparams.get("dual_energy", False)
        self.parameters["uses_dual_energy"] = uses_de

    def _set_code_unit_attributes(self):
        if self.cosmological_simulation:
            box_size = self.parameters["Physics"]["cosmology"]["comoving_box_size"]
            k = cosmology_get_units(
                self.hubble_constant,
                self.omega_matter,
                box_size,
                self.parameters["Physics"]["cosmology"]["initial_redshift"],
                self.current_redshift,
            )
            # Now some CGS values
            setdefaultattr(self, "length_unit", self.quan(box_size, "Mpccm/h"))
            setdefaultattr(
                self,
                "mass_unit",
                self.quan(k["urho"], "g/cm**3") * (self.length_unit.in_cgs()) ** 3,
            )
            setdefaultattr(self, "velocity_unit", self.quan(k["uvel"], "cm/s"))
        else:
            p = self.parameters
            for d, u in zip(("length", "time"), ("cm", "s")):
                val = nested_dict_get(p, ("Units", d), default=1)
                setdefaultattr(self, f"{d}_unit", self.quan(val, u))
            mass = nested_dict_get(p, ("Units", "mass"))
            if mass is None:
                density = nested_dict_get(p, ("Units", "density"))
                if density is not None:
                    mass = density * self.length_unit**3
                else:
                    mass = 1
            setdefaultattr(self, "mass_unit", self.quan(mass, "g"))
            setdefaultattr(self, "velocity_unit", self.length_unit / self.time_unit)

        magnetic_unit = np.sqrt(
            4 * np.pi * self.mass_unit / (self.time_unit**2 * self.length_unit)
        )
        magnetic_unit = np.float64(magnetic_unit.in_cgs())
        setdefaultattr(self, "magnetic_unit", self.quan(magnetic_unit, "gauss"))

    def bname_from_pos(self, pos: np.ndarray, lvl: int) -> str:
        return bname_from_pos(
            pos,
            lvl,
            self.min_level,
            self.domain_dimensions,
            self.dimensionality,
        )

    def __str__(self):
        return remove_ext(self.basename)

    @classmethod
    def _is_valid(cls, filename, *args, **kwargs):
        if not any(filename.endswith(sfx) for sfx in cls._suffixes):
            return False
        base_fn = remove_ext(filename)
        if not os.path.isfile(bfn := f"{base_fn}-0.block_list"):
            bfn = f"{base_fn}.block_list"
        try:
            with open(bfn) as f:
                return f.read(1)[0] == "B"
        except Exception:
            return False
