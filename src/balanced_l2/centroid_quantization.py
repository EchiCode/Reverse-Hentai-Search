"""FP16 and LVQ centroid wire formats used by the joint quantization sweep."""

import importlib.util
from pathlib import Path
import struct

import numpy as np

from balanced_l2.common import source_path


METHODS = ('fp16', 'lvq8', 'lvq6', 'lvq5')
_ENCODER_PATH = source_path(Path(__file__).resolve().parents[1], 'client/scripts/encode-centroids.py')
_SPEC = importlib.util.spec_from_file_location('joint_centroid_encoder', _ENCODER_PATH)
_ENCODER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ENCODER)
_BATCH_ROWS = 4096


def _shape(heads, dimensions):
    if (isinstance(heads, bool) or not isinstance(heads, int) or not 1 <= heads <= 2**32 - 1
            or isinstance(dimensions, bool) or not isinstance(dimensions, int)
            or not 1 <= dimensions <= 65535):
        raise ValueError('Centroid count and dimensions must fit uint32 and uint16')


def encode(values, method):
    """Encode a common FP16 source; return uncompressed bytes, without changing norms."""
    if method not in METHODS:
        raise ValueError(f'Unknown centroid codec: {method}')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.asarray(values, dtype=np.float32)
        rounded = values.astype('<f2')
    if values.ndim != 2:
        raise ValueError('Expected a two-dimensional centroid matrix')
    _shape(len(values), values.shape[1])
    if not np.isfinite(values).all() or not np.isfinite(rounded).all():
        raise ValueError('Centroids must fit finite FP16')
    if method == 'fp16':
        return rounded.tobytes()
    return _ENCODER.encode_values(rounded.astype(np.float32), int(method[3:]), metric='l2')


def decode(payload, method, heads, dimensions):
    """Decode literal wire arithmetic, rounding once to Float32 after restoring the mean."""
    if method not in METHODS:
        raise ValueError(f'Unknown centroid codec: {method}')
    _shape(heads, dimensions)
    if method == 'fp16':
        if len(payload) != heads * dimensions * 2:
            raise ValueError('Invalid FP16 centroid size')
        values = np.frombuffer(payload, '<f2').astype(np.float32).reshape(heads, dimensions)
        if not np.isfinite(values).all():
            raise ValueError('FP16 centroids must be finite')
        return values
    bits = int(method[3:])
    prefix = 16 + dimensions * 2 + heads * 4
    row_bytes = (dimensions * bits + 7) // 8
    if (len(payload) != prefix + heads * row_bytes or len(payload) < 16
            or struct.unpack_from('<8sHHI', payload) != (f'LVQ{bits}IDX\0'.encode(), 2, dimensions, heads)):
        raise ValueError('Invalid LVQ L2 centroid header or size')
    metadata = np.frombuffer(payload, '<f2', count=dimensions + 2 * heads, offset=16).astype(np.float64)
    mean, lower, delta = np.split(metadata, [dimensions, dimensions + heads])
    if not np.isfinite(metadata).all() or (delta <= 0).any():
        raise ValueError('Centroid metadata must be finite with positive scales')
    packed = np.frombuffer(payload, np.uint8, offset=prefix).reshape(heads, row_bytes)
    used_bits = dimensions * bits % 8
    if used_bits and (packed[:, -1] >> used_bits).any():
        raise ValueError('Centroid row padding must be zero')
    values = np.empty((heads, dimensions), np.float32)
    for first in range(0, heads, _BATCH_ROWS):
        stop = min(first + _BATCH_ROWS, heads)
        if bits == 8:
            codes = packed[first:stop]
        else:
            planes = np.unpackbits(packed[first:stop], axis=1, count=dimensions * bits, bitorder='little')
            codes = np.sum(planes.reshape(stop - first, dimensions, bits) << np.arange(bits, dtype=np.uint8),
                           axis=2, dtype=np.uint8)
        values[first:stop] = (lower[first:stop, None] + codes * delta[first:stop, None] + mean).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Centroid reconstruction must be finite')
    return values
