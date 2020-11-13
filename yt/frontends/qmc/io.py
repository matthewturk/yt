import os
from collections import defaultdict

from ase.io import read
import numpy as np

from yt.units.yt_array import uconcatenate
from yt.utilities.io_handler import BaseIOHandler
from yt.utilities.lib.particle_kdtree_tools import generate_smoothing_length
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.on_demand_imports import _h5py as h5py


class IOHandlerQMC(BaseIOHandler):
    """
    Data is read off disk when it is accessed for the first time, e.g.,
    ad["density"], NOT the creation of ad or yt.load() or even the creation
    of the index. 

    Objects involved:
        YTDataContainer: data_objects/data_containers.py
        Index: geometry/geometry_handler.py
        YTSelectionContainer: data_objects/selection_objects/data_selection_objects.py
        BaseIOHandler: utilities/io_handler.py
        IOHandlerFrontendName: frontends/frontendName/io.py

    * Access data: ad["density"]
    * Calls YTDataContainer's __getitem__ method
    * Calls YTSelectionContainer's get_data method
    * Calls Index's _read_particle_fields method
    * Calls BaseIOHandler's _read_particle_selection method
    * Calls IOHandlerFrontendName's _read_particle_fields method
    * Reads data from disk
    """
    _dataset_type = "qmc"
    _vector_fields = (
        ("positions", 3),
    )

    def __init__(self, ds, *args, **kwargs):
        self._vector_fields = dict(self._vector_fields)
        self.data_files = set([])
        super().__init__(ds, *args, **kwargs)

    def _read_fluid_selection(self, chunks, selector, fields, size):
        raise NotImplementedError

    def _read_particle_coords(self, chunks, ptf):
        """
        This is called in _count_particles_chunks in utilities/io_handler.py,
        which, in turn, is called by io handler's _read_particle_selection.
        """
        # This needs to *yield* a series of tuples of (ptype, (x, y, z)).
        # chunks is a list of chunks, and ptf is a dict where the keys are
        # ptypes and the values are lists of fields.
        data_files = set([])
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in sorted(data_files, key=lambda x: (x.filename, x.start)):
            tp = data_file.total_particles
            atoms = read(data_file.filename)
            for ptype in ptf:
                pos = atoms.arrays["positions"]
                yield ptype, (pos[:, 0], pos[:, 1], pos[:, 2])

    def _read_particle_fields(self, chunks, ptf, selector):
        data_files = set([])
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in sorted(data_files, key=lambda x: (x.filename, x.start)):
            tp = data_file.total_particles
            atoms = read(data_file.filename)
            for ptype, field_list in sorted(ptf.items()):
                if tp[ptype] == 0:
                    continue
                if getattr(selector, "is_all_data", False):
                    mask = slice(None, None, None)
                else:
                    pos = atoms.arrays["positions"] 
                    mask = selector.select_points(pos[:, 0], pos[:, 1], pos[:, 2])
                    del pos
                if mask is None:
                    continue
                for field in field_list:
                    data = atoms.arrays[field] 
                    data = data[mask, ...]
                    yield (ptype, field), data

    def _count_particles(self, data_file):
        si, ei = data_file.start, data_file.end
        pcount = np.array(len(read(data_file.filename)))
        if None not in (si, ei):
            np.clip(pcount - si, 0, ei - si, out=pcount)
        npart = {"io" : pcount}
        return npart

    def _identify_fields(self, domain):
        field_list = [
            ("io", "numbers"),
            ("io", "positions"),
        ]
        return field_list, {}

    def _yield_coordinates(self, data_file, needed_ptype=None):
        atoms = read(data_file.filename)
        for ptype, count in data_file.total_particles.items():
            if count == 0:
                continue
            if needed_ptype is not None and ptype != needed_ptype:
                continue
            pp = atoms.arrays["positions"]
            yield ptype, pp

    def _count_particles_chunks(self, psize, chunks, ptf, selector):
        if getattr(selector, "is_all_data", False):
            chunks = list(chunks)
            data_files = set([])
            for chunk in chunks:
                for obj in chunk.objs:
                    data_files.update(obj.data_files)
            data_files = sorted(data_files, key=lambda x: (x.filename, x.start))
            for data_file in data_files:
                for ptype in ptf.keys():
                    psize[ptype] += data_file.total_particles[ptype]
        else:
            for ptype, (x, y, z) in self._read_particle_coords(chunks, ptf):
                psize[ptype] += selector.count_points(x, y, z)
        return dict(psize)

    def _generate_smoothing_length(self, index):
        data_files = index.data_files
        _, extension = os.path.splitext(data_files[0].filename)
        hsml_fn = data_files[0].filename.replace(extension, ".hsml.hdf5")
        if os.path.exists(hsml_fn):
            with h5py.File(hsml_fn, mode="r") as f:
                file_hash = f.attrs["q"]
            if file_hash != self.ds._file_hash:
                mylog.warning("Replacing hsml files.")
                for data_file in data_files:
                    hfn = data_file.filename.replace(extension, ".hsml.hdf5")
                    os.remove(hfn)
            else:
                return
        positions = []
        counts = defaultdict(int)
        for data_file in data_files:
            for _, ppos in self._yield_coordinates(
                data_file, needed_ptype=self.ds._sph_ptypes[0]
            ):
                counts[data_file.filename] += ppos.shape[0]
                positions.append(ppos)
        if not positions:
            return
        offsets = {}
        offset = 0
        for fn, count in counts.items():
            offsets[fn] = offset
            offset += count
        kdtree = index.kdtree
        positions = uconcatenate(positions)[kdtree.idx]
        hsml = generate_smoothing_length(positions, kdtree, self.ds._num_neighbors)
        dtype = positions.dtype
        hsml = hsml[np.argsort(kdtree.idx)].astype(dtype)
        mylog.warning("Writing smoothing lengths to hsml files.")
        for i, data_file in enumerate(data_files):
            si, ei = data_file.start, data_file.end
            fn = data_file.filename
            _, extension = os.path.splitext(fn)
            hsml_fn = data_file.filename.replace(extension, ".hsml.hdf5")
            with h5py.File(hsml_fn, mode="a") as f:
                if i == 0:
                    f.attrs["q"] = self.ds._file_hash
                g = f.require_group(self.ds._sph_ptypes[0])
                d = g.require_dataset(
                    "SmoothingLength", dtype=dtype, shape=(counts[fn],)
                )
                begin = si + offsets[fn]
                end = min(ei, d.size) + offsets[fn]
                d[si:ei] = hsml[begin:end]
        return hsml

    def _get_smoothing_length(self, data_file, position_dtype, position_shape):
        ptype = self.ds._sph_ptypes[0]
        si, ei = data_file.start, data_file.end
        if self.ds.gen_hsmls:
            _, extension = os.path.splitext(data_file.filename)
            fn = data_file.filename.replace(extension, ".hsml.hdf5")
        else:
            fn = data_file.filename
        with h5py.File(fn, mode="r") as f:
            ds = f[ptype]["SmoothingLength"][si:ei, ...]
            dt = ds.dtype.newbyteorder("N")  # Native
            hsml = np.empty(ds.shape, dtype=dt)
            hsml[:] = ds
            return hsml
