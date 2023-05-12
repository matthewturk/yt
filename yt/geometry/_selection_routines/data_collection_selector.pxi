cdef class DataCollectionSelector(SelectorObject):
    cdef object obj_ids
    cdef np.int64_t nids

    def __init__(self, dobj):
        self.obj_ids = dobj._obj_ids
        self.nids = self.obj_ids.shape[0]

    cdef int select_bbox(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) nogil:
        # We have to return 1 here, although it does make me uncomfortable to
        # do so.  This will always say we're hitting a certain bbox, but the
        # only grids that will be selected are those that match the grid index.
        return 1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void visit_grid_cells(self, GridVisitor visitor,
                              GridTreeNode *grid, int use_cache,
                              np.uint8_t[:] cached_mask):
        # We override so that we can correctly pick out the grids we want to
        # visit.
        cdef np.float64_t left_edge[3]
        cdef np.float64_t right_edge[3]
        cdef np.float64_t dds[3]
        cdef int dim[3]
        cdef int level, i, selected = 0
        cdef np.float64_t pos[3]
        cdef np.int64_t[:] oids = self.obj_ids
        # This should be done faster somehow, but it's not awful for now.
        for i in range(oids.shape[0]):
            if oids[i] == grid.index:
                selected = 1
                break
        if selected == 0: return
        level = grid.level
        for i in range(3):
            left_edge[i] = grid.left_edge[i]
            right_edge[i] = grid.right_edge[i]
            dds[i] = (right_edge[i] - left_edge[i])/grid.dims[i]
            dim[i] = grid.dims[i]
        with nogil:
            pos[0] = left_edge[0] + dds[0] * 0.5
            visitor.pos[0] = 0
            for i in range(dim[0]):
                pos[1] = left_edge[1] + dds[1] * 0.5
                visitor.pos[1] = 0
                for j in range(dim[1]):
                    pos[2] = left_edge[2] + dds[2] * 0.5
                    visitor.pos[2] = 0
                    for k in range(dim[2]):
                        # Selected is always 1 for this type
                        visitor.visit(grid, 1)
                        visitor.global_index += 1
                        pos[2] += dds[2]
                        visitor.pos[2] += 1
                    pos[1] += dds[1]
                    visitor.pos[1] += 1
                pos[0] += dds[0]
                visitor.pos[0] += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def select_grids(self,
                     np.ndarray[np.float64_t, ndim=2] left_edges,
                     np.ndarray[np.float64_t, ndim=2] right_edges,
                     np.ndarray[np.int32_t, ndim=2] levels):
        cdef int n
        cdef int ng = left_edges.shape[0]
        cdef np.ndarray[np.uint8_t, ndim=1] gridi = np.zeros(ng, dtype='uint8')
        cdef np.ndarray[np.int64_t, ndim=1] oids = self.obj_ids
        with nogil:
            for n in range(self.nids):
                gridi[oids[n]] = 1
        return gridi.astype("bool")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fill_mask_regular_grid(self, gobj):
        cdef np.ndarray[np.uint8_t, ndim=3] mask
        mask = np.ones(gobj.ActiveDimensions, dtype='uint8')
        return mask.astype("bool")

    def _hash_vals(self):
        return (hash(self.obj_ids.tobytes()), self.nids)

data_collection_selector = DataCollectionSelector
