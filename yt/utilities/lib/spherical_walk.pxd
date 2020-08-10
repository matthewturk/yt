import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from .volume_container cimport VolumeContainer
from .grid_traversal cimport sampler_function

cdef extern from "spherical_volume_rendering_util.h" namespace "svr":
    cdef cppclass SphericalVoxel:
        int radial, polar, azimuthal
        double enter_t
        double exit_t
    cdef cppclass SphereBound:
        double radial, polar, azimuthal

    vector[SphericalVoxel] walkSphericalVolume(double *ray_origin, double *ray_direction,
                                               double *min_bound, double *max_bound,
                                               size_t num_radial_voxels, size_t num_polar_voxels,
                                               size_t num_azimuthal_voxels, double *sphere_center,
                                               double max_t) nogil

cdef int walk_volume_spherical(VolumeContainer *vc,
                          np.float64_t ray_origin[3],
                          np.float64_t ray_direction[3],
                          sampler_function *sample,
                          void *data,
                          np.float64_t *return_t,
                          np.float64_t max_t) nogil
