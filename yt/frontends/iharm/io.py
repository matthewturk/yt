"""
Skeleton-specific IO functions



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.utilities.io_handler import \
    BaseIOHandler


class IHarmIOHandler(BaseIOHandler):
    _particle_reader = False
    _dataset_type = "iharm"

    def _read_particle_coords(self, chunks, ptf):
        pass

    def _read_particle_fields(self, chunks, ptf, selector):
        pass

    def _read_fluid_selection(self, chunks, selector, fields, size):
        # This needs to allocate a set of arrays inside a dictionary, where the
        # keys are the (ftype, fname) tuples and the values are arrays that
        # have been masked using whatever selector method is appropriate.  The
        # dict gets returned at the end and it should be flat, with selected
        # data.  Note that if you're reading grid data, you might need to
        # special-case a grid selector object.
        # Also note that "chunks" is a generator for multiple chunks, each of
        # which contains a list of grids. The returned numpy arrays should be
        # in 64-bit float and contiguous along the z direction. Therefore, for
        # a C-like input array with the dimension [x][y][z] or a
        # Fortran-like input array with the dimension (z,y,x), a matrix
        # transpose is required (e.g., using np_array.transpose() or
        # np_array.swapaxes(0,2)).

        chunks = list(chunks)
        if any((ftype != "iharm" for ftype,_ in fields)):
            raise NotImplementedError

        rv = {}
        for field in fields:
            rv[field] = np.empty(size, dtype='=f8')
        
        for field in fields:
            t = self.ds._field_map[field[1]]
            ds = self.ds._handle['/' + t[0]]
            ind = 0
            for chunk in chunks:
                for mesh in chunk.objs:
                    data = ds[:,:,:,t[1]]
                    ind += mesh.select(selector, data, rv[field], ind)

        return rv



    def _read_chunk_data(self, chunk, fields):
        # TODO
        
        f = self._handle
        rv = {}
        for g in chunk.objs:
            print(g)
            rv[g.id] = {}

        print("call to chunk data")

        return rv

