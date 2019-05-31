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

import abc
import functools
import numpy as np
import weakref
from yt.funcs import iterable
from collections import defaultdict

@functools.total_ordering
class ParticleFile(metaclass = abc.ABCMeta):
    # Temporary small chunk size for testing!
    # Change this to 64**3 or something.

    _fields_in_file = None
    _num_chunks = None

    def __init__(self, ds, io, filename, file_id, chunk_size = 32**3):
        self.ds = ds
        self.io = weakref.proxy(io)
        self.filename = filename
        self.file_id = file_id
        self.total_particles = self._count_particles()
        self.chunk_size = chunk_size
        self._calculate_chunk_indices()

    @abc.abstractmethod
    def _count_particles(self):
        # Return a dict of (ptype, nparticles)
        pass

    @abc.abstractmethod
    def _identify_fields(self):
        # Needs to return (fields, units) where (fields) is [(ptype,
        # pfield), ...]
        pass

    def keys(self):
        if self._fields_in_file is None:
            _, self._fields_in_file = self._identify_fields()
        yield from self._fields_in_file


    def _calculate_chunk_indices(self):
        cinds = {}
        for ptype, npart in self.total_particles.items():
            start = np.arange(0, npart, self.chunk_size)
            end = np.clip(start + self.chunk_size, 0, npart)
            cinds[ptype] = np.array([(si, ei) for si, ei in zip(start, end)])
        self._chunk_indices = cinds
        # We have chunks for each
        self._num_chunks = sum(len(_) for _ in cinds.values())

    def _calculate_offsets(self, fields, pcounts):
        pass

    def iter_coordinates(self, ptypes, chunk_slice = None):
        if chunk_slice is None:
            chunk_slice = defaultdict(lambda: None)
        for ptype in ptypes:
            field_spec = [(ptype, [])]
            for ci, (pos, _) in self.iter_chunks(
                    field_spec, return_positions = True,
                    chunk_slice = chunk_slice[ptype]):
                yield ci, (ptype, pos)
                _ = [__ for __ in _] # exhaust our iterator


    def iter_chunks(self, field_spec, return_positions = False,
                    chunk_slice = None):
        # field_spec here is a list of (field_type, (field0, field1))
        # tuples
        if chunk_slice is None:
            chunk_slice = defaultdict(lambda: slice(None))
        elif not iterable(chunk_slice): # single number
            pass
        elif isinstance(chunk_slice, slice): # already a slice
            pass
        with self._open_file() as f:
            for ptype, fields in field_spec:
                if isinstance(chunk_slice, dict):
                    sl = chunk_slice[ptype]
                else:
                    sl = chunk_slice
                for ci, (si, ei) in enumerate(
                        self._chunk_indices[ptype][sl]):
                    # Note: we aren't using "yield from" here, but instead
                    # yielding a tuple of a set of positions and a
                    # generator that will return all the fields.
                    if return_positions:
                        yield (ci, (
                            self._read_particle_positions(ptype,
                                                         (f, (si, ei))),
                            self._iter_fields(ptype, fields, (f, (si, ei)))))
                    else:
                        yield from ((ci, _) for _ in
                                    self._iter_fields(ptype, fields,
                                                     (f, (si, ei))))

    def __getitem__(self, key):
        # We assume here that "key" is one or multiple field-tuples.
        # Validating this is out of scope for the present time.
        if not isinstance(key, tuple):
            raise KeyError
        if not len(key) == 2:
            raise KeyError
        if not all(isinstance(_, str) for _ in key):
            raise KeyError
        return np.concatenate([v for (_, v) in 
                               self._iter_fields(key[0], [key[1]])])

    def __lt__(self, other):
        return self.filename < other.filename

    def __eq__(self, other):
        return self.filename == other.filename

    def __hash__(self):
        return hash((self.filename, self.file_id))

    @abc.abstractmethod
    def _read_particle_positions(self, ptype, f=None):
        raise NotImplementedError

    def _get_particle_positions(self, ptype, f=None):
        pcount = self.total_particles[ptype]
        if pcount == 0:
            return None

        # Correct for periodicity.
        dle = self.ds.domain_left_edge.to('code_length').v
        dw = self.ds.domain_width.to('code_length').v
        pos = self._read_particle_positions(ptype)
        pos.convert_to_units('code_length')
        pos = pos.v
        np.subtract(pos, dle, out=pos)
        np.mod(pos, dw, out=pos)
        np.add(pos, dle, out=pos)

        for si, ei in self._chunk_indices[ptype]:
            yield pos[si:ei]

    def _iter_fields(self, ptype, field_list, state=None):
        raise NotImplementedError

    def _get_particle_fields(self, ptf, selector, chunk_slice = None):
        for ptype, field_list in sorted(ptf.items()):
            pcount = self.total_particles[ptype]
            if pcount == 0:
                continue

            field_spec = [(ptype, field_list)]
            for ci, (coords, read_iter) in self.iter_chunks(
                    field_spec, return_positions = True,
                    chunk_slice = chunk_slice):
                x = coords[:, 0]
                y = coords[:, 1]
                z = coords[:, 2]
                mask = selector.select_points(x, y, z, 0.0)
                del x, y, z
                if mask is None:
                    continue

                for (_, field), data in read_iter:
                    data = data[mask]
                    yield (ptype, field), data
