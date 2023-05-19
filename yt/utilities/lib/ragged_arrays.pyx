"""
Some simple operations for operating on ragged arrays



"""


import numpy as np

cimport cython
cimport numpy as np


cdef fused numpy_dt:
    np.float32_t
    np.float64_t
    np.int32_t
    np.int64_t

cdef numpy_dt r_min(numpy_dt a, numpy_dt b):
    if a < b: return a
    return b

cdef numpy_dt r_max(numpy_dt a, numpy_dt b):
    if a > b: return a
    return b

cdef numpy_dt r_add(numpy_dt a, numpy_dt b):
    return a + b

cdef numpy_dt r_subtract(numpy_dt a, numpy_dt b):
    return a - b

cdef numpy_dt r_multiply(numpy_dt a, numpy_dt b):
    return a * b

@cython.cdivision(True)
cdef numpy_dt r_divide(numpy_dt a, numpy_dt b):
    return a / b

def index_unop(np.ndarray[numpy_dt, ndim=1] values,
              np.ndarray[np.int64_t, ndim=1] indices,
              np.ndarray[np.int64_t, ndim=1] sizes,
              operation):
    cdef numpy_dt mi, ma
    if numpy_dt == np.float32_t:
        dt = "float32"
        mi = np.finfo(dt).min
        ma = np.finfo(dt).max
    elif numpy_dt == np.float64_t:
        dt = "float64"
        mi = np.finfo(dt).min
        ma = np.finfo(dt).max
    elif numpy_dt == np.int32_t:
        dt = "int32"
        mi = np.iinfo(dt).min
        ma = np.iinfo(dt).max
    elif numpy_dt == np.int64_t:
        dt = "int64"
        mi = np.iinfo(dt).min
        ma = np.iinfo(dt).max
    cdef np.ndarray[numpy_dt] out_values = np.zeros(sizes.size, dtype=dt)
    cdef numpy_dt (*func)(numpy_dt a, numpy_dt b)
    # Now we figure out our function.  At present, we only allow addition and
    # multiplication, because they are commutative and easy to bootstrap.
    cdef numpy_dt ival, val
    if operation == "sum":
        ival = 0
        func = r_add
    elif operation == "prod":
        ival = 1
        func = r_multiply
    elif operation == "max":
        ival = mi
        func = r_max
    elif operation == "min":
        ival = ma
        func = r_min
    else:
        raise NotImplementedError
    cdef np.int64_t i, ind_ind, ind_arr
    ind_ind = 0
    for i in range(sizes.size):
        # Each entry in sizes is the size of the array
        val = ival
        for _ in range(sizes[i]):
            ind_arr = indices[ind_ind]
            val = func(val, values[ind_arr])
            ind_ind += 1
        out_values[i] = val
    return out_values

def face_barycenters(np.ndarray[np.int64_t, ndim=1] sizes,
                    np.ndarray[np.int64_t, ndim=1] indices,
                    np.ndarray[numpy_dt, ndim=2] coords,
                    int index_offset = 0):
    # Here, sizes is the number of nodes per face, indices is the node index
    # values (should be equal in size to the sum of sizes) and coords are what
    # we are looking up into.  index_offset is what we subtract.
    cdef int i, j, k
    cdef int current_index = 0
    cdef np.ndarray[np.float64_t, ndim=2] barycenters = np.zeros((sizes.shape[0], 3), dtype='f8')
    for i in range(sizes.shape[0]):
        for j in range(sizes[i]):
            for k in range(3):
                barycenters[i, k] += coords[indices[current_index] - index_offset, k]
            current_index += 1
        for k in range(3):
            barycenters[i, k] /= sizes[i]
    return barycenters

def cell_barycenters(np.ndarray[np.int64_t, ndim=1] face_c0,
                     np.ndarray[numpy_dt, ndim=2] face_centers,
                     int index_offset = 0):
    # This accepts the c0 array, which is the 'internal to the face' cell IDs, and the face_centers.
    # We track two arrays on the way out -- to get the number of faces and the center.
    # We make an assumption that our cells are indexed such that face_c0.max() is the number of cells.
    cdef np.uint64_t ncells = face_c0.max() - index_offset + 1 # usually indexed by one
    cdef np.ndarray[np.float64_t, ndim=2] barycenters = np.zeros((ncells, 3), dtype="f8")
    cdef np.ndarray[np.uint16_t, ndim=1] nfaces = np.zeros(ncells, dtype="u2")
    cdef int i, j
    for i in range(face_c0.shape[0]):
        nfaces[face_c0[i] - index_offset] += 1
        for j in range(3):
            barycenters[face_c0[i] - index_offset, j] += face_centers[i, j]
    return barycenters / nfaces[:,None], nfaces
