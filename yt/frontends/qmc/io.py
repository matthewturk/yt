from ase.io import read
import numpy as np

from yt.utilities.io_handler import BaseIOHandler


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
        ("Coordinates", 3),
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
            poff = data_file.field_offsets
            tp = data_file.total_particles
            f = open(data_file.filename, "rb")
            for ptype in ptf:
                f.seek(poff[ptype, "Coordinates"], os.SEEK_SET)
                pos = self._read_field_from_file(f, tp[ptype], "Coordinates")
                if ptype == self.ds._sph_ptypes[0]:
                    f.seek(poff[ptype, "SmoothingLength"], os.SEEK_SET)
                    hsml = self._read_field_from_file(f, tp[ptype], "SmoothingLength")
                else:
                    hsml = 0.0
                yield ptype, (pos[:, 0], pos[:, 1], pos[:, 2]), hsml
            f.close()

    def _read_particle_fields(self, chunks, ptf, selector):
        raise NotImplementedError

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
            ("io", "momenta"),
        ]
        return field_list, {}

    def _yield_coordinates(self, data_file, needed_ptype=None):
        raise NotImplementedError
