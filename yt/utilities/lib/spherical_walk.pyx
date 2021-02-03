# distutils: language = c++
# distutils: extra_compile_args=["-std=c++11"]
# distutils: sources = SPHERICAL_VR_SOURCE
# distutils: include_dirs = LIB_DIR_SPHERICAL_VR

import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from .volume_container cimport VolumeContainer
from .grid_traversal cimport sampler_function

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int walk_volume_spherical(VolumeContainer *vc,
                          np.float64_t ray_origin[3],
                          np.float64_t ray_direction[3],
                          sampler_function *sample,
                          void *data,
                          np.float64_t *return_t,
                          np.float64_t max_t) nogil:
    '''
    Spherical Coordinate Voxel Traversal Algorithm
    Cythonized version of the Spherical Coordinate Voxel Traversal Algorithm.
    Arguments:
           ray_origin: The 3-dimensional (x,y,z) origin of the ray.
           ray_direction: The 3-dimensional (x,y,z) unit direction of the ray.
           min_bound: The minimum boundary of the sectored sphere in the form (radial, theta, phi).
           max_bound: The maximum boundary of the sectored sphere in the form (radial, theta, phi).
           num_radial_voxels: The number of radial voxels.
           num_polar_voxels: The number of polar voxels.
           num_azimuthal_voxels: The number of azimuthal voxels.
           sphere_center: The 3-dimensional (x,y,z) center of the sphere.
           max_t: The unitized maximum time of ray traversal. Defaulted to 1.0
    Returns:
           A numpy array of the spherical voxel coordinates.
           The voxel coordinates are as follows:
             For coordinate i in numpy array v:
             v[i,0] = radial_voxel
             v[i,1] = polar_voxel
             v[i,2] = azimuthal_voxel
    Notes:
        - If one wants to traverse the entire sphere, the min_bound = [0, 0, 0]
          and max_bound = [SPHERE_MAX_RADIUS, 2*pi, 2*pi]. Similarly, if one wants to traverse
          the upper hemisphere, max_bound = SPHERE_MAX_RADIUS, 2*pi, pi].
        - Code must be compiled before use:
           > python3 cython_SVR_setup.py build_ext --inplace
    '''

    cdef double min_bound[3], max_bound[3]
    cdef size_t num_radial_voxels, num_polar_voxels, num_azimuthal_voxels

    min_bound[0] = vc.left_edge[0]
    max_bound[0] = vc.right_edge[0]

    min_bound[1] = vc.left_edge[2]
    max_bound[1] = vc.right_edge[2]

    min_bound[2] = vc.left_edge[1]
    max_bound[2] = vc.right_edge[1]

    num_radial_voxels = vc.dims[0]
    num_azimuthal_voxels = vc.dims[1]
    num_polar_voxels = vc.dims[2]
    cdef double sphere_center[3]
    sphere_center[0] = sphere_center[1] = sphere_center[2] = 0.0

    cdef vector[SphericalVoxel] voxels = walkSphericalVolume(ray_origin, ray_direction,
                                                             min_bound, max_bound,
                                                             num_radial_voxels, num_polar_voxels,
                                                             num_azimuthal_voxels, sphere_center,
                                                             max_t)
    cdef int cur_ind[3]
    cdef np.float64_t exit_t, enter_t

    for i in range(voxels.size()):
        cur_ind[0] = voxels[i].radial
        cur_ind[1] = voxels[i].azimuthal
        cur_ind[2] = voxels[i].polar
        enter_t = voxels[i].enter_t
        exit_t = voxels[i].exit_t
        sample(vc, ray_origin, ray_direction, enter_t, exit_t, cur_ind, data)
    return voxels.size()
