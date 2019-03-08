"""
ParticleStore and related classes.




"""

#-----------------------------------------------------------------------------
# Copyright (c) yt Development Team. All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import functools
import numpy as np
import weakref

@functools.total_ordering
class ParticleFile(object):
    _chunksize = 4

    def __init__(self, ds, io, filename, file_id):
        self.ds = ds
        self.io = weakref.proxy(io)
        self.filename = filename
        self.file_id = file_id
        self.total_particles = self._count_particles()
        self._calculate_chunk_indices()

    def _calculate_chunk_indices(self):
        cinds = {}
        for ptype, npart in self.total_particles.items():
            start = np.arange(0, npart, self._chunksize)
            end = np.clip(start + self._chunksize, 0, npart)
            cinds[ptype] = [(si, ei) for si, ei in zip(start, end)]
        self._chunk_indices = cinds

    def select(self, selector):
        pass

    def count(self, selector):
        pass

    def _calculate_offsets(self, fields, pcounts):
        pass

    def __lt__(self, other):
        if self.filename != other.filename:
            return self.filename < other.filename
        return self.start < other.start

    def __eq__(self, other):
        if self.filename != other.filename:
            return False
        return self.start == other.start

    def __hash__(self):
        return hash((self.filename, self.file_id))

    def _read_particle_positions(self, ptype, f=None):
        raise NotImplementedError

    def _get_particle_positions(self, ptype, f=None):
        pcount = self.total_particles[ptype]
        if pcount == 0:
            return None

        # Correct for periodicity.
        dle = self.ds.domain_left_edge.to('code_length').v
        dw = self.ds.domain_width.to('code_length').v
        pos = self._read_particle_positions(ptype, f=f)
        pos.convert_to_units('code_length')
        pos = pos.v
        np.subtract(pos, dle, out=pos)
        np.mod(pos, dw, out=pos)
        np.add(pos, dle, out=pos)

        for si, ei in self._chunk_indices[ptype]:
            yield pos[si:ei]

    def _read_particle_fields(self, ptype, field_list, frange=None):
        raise NotImplementedError

    def _get_particle_fields(self, ptf, selector):
        for ptype, field_list in sorted(ptf.items()):
            pcount = self.total_particles[ptype]
            if pcount == 0:
                continue

            for frange, coords in zip(self._chunk_indices[ptype],
                                      self._get_particle_positions(ptype)):
                x = coords[:, 0]
                y = coords[:, 1]
                z = coords[:, 2]
                mask = selector.select_points(x, y, z, 0.0)
                del x, y, z
                if mask is None:
                    continue

                for field, data in \
                  self._read_particle_fields(ptype, field_list, frange):
                    data = data[mask]
                    yield (ptype, field), data
