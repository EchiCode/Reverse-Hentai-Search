"""Indexed reads over original FP16 files, without another corpus on disk."""

import mmap
import operator
from pathlib import Path

import numpy as np

from balanced_l2.common import record_dtype


class _Field:
    def __init__(self, records, name):
        self.records, self.name = records, name
        dtype = records.dtype.fields[name][0]
        self.dtype, dimensions = dtype.subdtype or (dtype, ())
        self.shape = (len(records), *dimensions)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, rows):
        return self.records._take(rows, self.name)


class SourceRecords:
    """Read concatenated source rows in file order; returned selections are copies."""

    def __init__(self, files, dim):
        if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 65535:
            raise ValueError('Source dimensions must fit uint16')
        self.files = tuple(Path(path).resolve() for path in files)
        if not self.files or len(set(self.files)) != len(self.files):
            raise ValueError('Expected distinct original source files')
        self.dtype = record_dtype(dim)
        self._arrays, self._maps, ends = [], [], []
        self._count, self._closed = 0, False
        try:
            for path in self.files:
                with path.open('rb') as stream:
                    size = stream.seek(0, 2)
                    if size % self.dtype.itemsize:
                        raise ValueError(f'Truncated original source records: {path}')
                    if not size:
                        continue
                    # Python 3.13 can retain a mapping without retaining a file descriptor.
                    mapping = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ, trackfd=False)
                self._maps.append(mapping)
                self._arrays.append(np.ndarray(size // self.dtype.itemsize, dtype=self.dtype, buffer=mapping))
                self._count += len(self._arrays[-1])
                ends.append(self._count)
        except BaseException:
            self.close()
            raise
        self._ends = np.asarray(ends, dtype=np.int64)
        self._starts = np.r_[0, self._ends[:-1]]
        self.shape = (self._count,)

    def __len__(self):
        return self._count

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        self._closed = True
        self._arrays.clear()
        for mapping in self._maps:
            mapping.close()
        self._maps.clear()

    def __getitem__(self, rows):
        if isinstance(rows, str):
            if rows not in self.dtype.names:
                raise KeyError(rows)
            return _Field(self, rows)
        return self._take(rows)

    def _take(self, rows, field=None):
        if self._closed:
            raise ValueError('Original source records are closed')
        dtype = self.dtype if field is None else self.dtype.fields[field][0]
        extra_shape = dtype.subdtype[1] if dtype.subdtype else ()
        if isinstance(rows, slice):
            first, stop, step = rows.indices(self._count)
            if step != 1:
                return self._take(np.arange(first, stop, step, dtype=np.int64), field)
            output = np.empty(max(0, stop - first), dtype=dtype)
            cursor, target = first, 0
            file_id = int(np.searchsorted(self._ends, cursor, side='right'))
            while cursor < stop:
                count = min(stop, int(self._ends[file_id])) - cursor
                source = self._arrays[file_id]
                if field is not None:
                    source = source[field]
                local = cursor - int(self._starts[file_id])
                output[target:target + count] = source[local:local + count]
                cursor, target, file_id = cursor + count, target + count, file_id + 1
            return output
        if isinstance(rows, (int, np.integer)) and not isinstance(rows, (bool, np.bool_)):
            indices = np.asarray(operator.index(rows), dtype=np.int64)
        else:
            indices = np.asarray(rows)
            if indices.dtype.kind == 'b':
                if indices.shape != (self._count,):
                    raise IndexError('Source boolean mask must cover every row')
                indices = np.flatnonzero(indices)
            elif indices.dtype.kind not in 'iu':
                if indices.size or isinstance(rows, np.ndarray):
                    raise IndexError('Source row indices must be integers')
                indices = indices.astype(np.int64)
        shape = indices.shape
        if indices.dtype.kind == 'u' and (indices >= self._count).any():
            raise IndexError('Source row index is out of bounds')
        indices = indices.astype(np.int64, copy=True).reshape(-1)
        indices[indices < 0] += self._count
        if ((indices < 0) | (indices >= self._count)).any():
            raise IndexError('Source row index is out of bounds')
        output = np.empty(len(indices), dtype=dtype)
        if len(indices):
            file_ids = np.searchsorted(self._ends, indices, side='right')
            order = np.argsort(file_ids, kind='stable')
            boundaries = np.r_[0, np.flatnonzero(np.diff(file_ids[order])) + 1, len(order)]
            for first, stop in zip(boundaries[:-1], boundaries[1:]):
                positions = order[first:stop]
                file_id = int(file_ids[positions[0]])
                source = self._arrays[file_id]
                if field is not None:
                    source = source[field]
                output[positions] = source[indices[positions] - self._starts[file_id]]
        return output.reshape((*shape, *extra_shape))[()] if not shape else output.reshape((*shape, *extra_shape))
