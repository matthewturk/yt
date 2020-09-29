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

    def __init__(self, ds, *args, **kwargs):
        super().__init__(ds, *args, **kwargs)

    def _read_fluid_selection(self, chunks, selector, fields, size):
        raise NotImplementedError

    def _read_particle_coords(self, chunks, ptf):
        """
        This is called in _count_particles_chunks in utilities/io_handler.py,
        which, in turn, is called by io handler's _read_particle_selection.
        """
        raise NotImplementedError

    def _read_particle_fields(self, chunks, ptf, selector):
        raise NotImplementedError
