from __future__ import annotations  # noqa: I001

import sys
from collections import defaultdict
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from yt.frontends.enzo_e_octree.misc import (get_particle_mass_correction,
                                             nested_dict_get)
from yt.utilities.exceptions import YTException
from yt.utilities.io_handler import BaseIOHandler
from yt.utilities.on_demand_imports import _h5py as h5py

if TYPE_CHECKING:
    from yt.frontends.enzo_e_octree.data_structures import EnzoESubset

    ParticleFields: TypeAlias = dict[str, list[str]]
    ParticleDataDict: TypeAlias = dict[tuple[str, str], np.ndarray]
    ParticleData: TypeAlias = tuple[tuple[str, str], np.ndarray]

if sys.version_info < (3, 10):
    from yt._maintenance.backports import zip

# TODO: Once I'm done with porting it to octree
# Go through all the copied files and eliminate dead code


class EnzoEIOHandler(BaseIOHandler):
    _dataset_type = "enzo_e_octree"
    _base = slice(None)
    _field_dtype = "float64"
    _sep = "_"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine if particle masses are actually masses or densities.
        if self.ds.parameters["version"] is not None:
            # they're masses for enzo-e versions that record a version string
            mass_flag = True
        else:
            # in earlier versions: query the existence of the "mass_is_mass"
            # particle parameter
            mass_flag = nested_dict_get(
                self.ds.parameters, ("Particle", "mass_is_mass"), default=None
            )
        # the historic approach for initializing the value of "mass_is_mass"
        # was unsound (and could yield a random value). Thus we should only
        # check for the parameter's existence and not its value
        self._particle_mass_is_mass = mass_flag is not None

    def _read_field_names(self, bname: str, h5fname: str):  # type: ignore
        f = h5py.File(h5fname, mode="r")
        try:
            group = f[bname]
        except KeyError as e:
            raise YTException(
                f"Block {bname} is missing from data file {h5fname}."
            ) from e
        fields = []
        ptypes = set()
        dtypes = set()
        # keep one field for each particle type so we can count later
        sample_pfields = {}
        for name, v in group.items():
            if not hasattr(v, "shape") or v.dtype == "O":
                continue
            # mesh fields are "field <name>"
            if name.startswith("field"):
                _, fname = name.split(self._sep, 1)
                fields.append(("enzoe", fname))
                dtypes.add(v.dtype)
            # particle fields are "particle <type> <name>"
            else:
                _, ftype, fname = name.split(self._sep, 2)
                fields.append((ftype, fname))
                ptypes.add(ftype)
                dtypes.add(v.dtype)
                if ftype not in sample_pfields:
                    sample_pfields[ftype] = fname
        self.sample_pfields = sample_pfields

        if len(dtypes) == 1:
            # Now, if everything we saw was the same dtype, we can go ahead and
            # set it here.  We do this because it is a HUGE savings for 32 bit
            # floats, since our numpy copying/casting is way faster than
            # h5py's, for some reason I don't understand.  This does *not* need
            # to be correct -- it will get fixed later -- it just needs to be
            # okay for now.
            self._field_dtype = list(dtypes)[0]
        f.close()
        return fields, ptypes

    def _read_particle_coords(self, chunks, ptf: ParticleFields):
        yield from (
            (ptype, xyz, 0.0)
            for ptype, xyz in self._read_particle_fields(chunks, ptf, None)
        )

    # The output is the same as the original enzo-e frontend, except
    # differently ordered (# TODO: Is this a problem?)
    def _read_particle_fields(self, chunks, ptf: ParticleFields, selector):
        fields: list[tuple[str, str]] = [
            (ptype, fname) for ptype, field_list in ptf.items() for fname in field_list
        ]
        for ptype, field_list in sorted(ptf.items()):
            for ax in "xyz":
                if ax not in field_list:
                    fields.append((ptype, ax))
        for chunk in chunks:
            for subset in chunk.objs:
                rv = self._read_particle_subset(subset, fields)
                for ptype, field_list in ptf.items():
                    x, y, z = (np.asarray(rv[ptype, ax]) for ax in "xyz")
                    if selector is None:
                        # This only ever happens if the call is made from
                        # _read_particle_coords.
                        yield ptype, (x, y, z)
                        continue
                    mask = selector.select_points(x, y, z, 0.0)
                    if mask is None:
                        mask = []
                    for field in field_list:
                        data = np.asarray(rv.pop((ptype, field))[mask])
                        yield (ptype, field), data

    def _read_particle_subset(
        self, subset: EnzoESubset, all_fields: list[tuple[str, str]]
    ) -> ParticleDataDict:
        """Read the particle files."""
        field_data = defaultdict(list)
        with h5py.File(subset.domain.h5fname) as f:
            for block in f.values():
                for field in all_fields:
                    field_data[field].append(block[f"particle_{field[0]}_{field[1]}"])
            return {k: np.concatenate(v) for k, v in field_data.items()}

    def _read_fluid_selection(self, chunks, selector, fields, size):
        chunk_fields = []

        for chunk in chunks:
            # Loop over subsets
            for subset in chunk.objs:
                fname = subset.domain.h5fname

                with h5py.File(fname) as f:
                    chunk_fields.append(subset.fill(f, fields, selector))

        if not chunk_fields:
            return {f: np.empty(shape=0, dtype=np.float64) for f in fields}
        return {
            field: np.concatenate(arrs)
            for field, arrs in zip(
                chunk_fields[0].keys(),
                zip(*(cf.values() for cf in chunk_fields), strict=True),
            )
        }

    def _read_obj_field(self, obj, field, fid_data):
        if fid_data is None:
            fid_data = (None, None)
        fid, rdata = fid_data
        if fid is None:
            close = True
            fid = h5py.h5f.open(obj.filename.encode("latin-1"), h5py.h5f.ACC_RDONLY)
        else:
            close = False
        ftype, fname = field
        node = f"/{obj.block_name}/field{self._sep}{fname}"
        dg = h5py.h5d.open(fid, node.encode("latin-1"))
        if rdata is None:
            rdata = np.empty(
                self.ds.grid_dimensions[: self.ds.dimensionality][::-1],
                dtype=self._field_dtype,
            )
        dg.read(h5py.h5s.ALL, h5py.h5s.ALL, rdata)
        if close:
            fid.close()
        data = rdata[self._base].T
        if self.ds.dimensionality < 3:
            nshape = data.shape + (1,) * (3 - self.ds.dimensionality)
            data = np.reshape(data, nshape)
        return data
