"""Local vector quantization of posting records, with a bounded packing buffer."""

import struct

import numpy as np

from balanced_l2.common import record_dtype


_HEADER = struct.Struct('<8sHBBI')
_MAGIC = b'LVQPOST\0'
_BATCH_ROWS = 4096


def _validate_dimensions(dimensions):
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 65535:
        raise ValueError('Posting dimensions must fit uint16')


def _quantized_dtype(dimensions, bits):
    return np.dtype([('code', '<i4'), ('page', '<i2'), ('lower', '<f2'), ('delta', '<f2'),
                     ('packed', 'u1', ((dimensions * bits + 7) // 8,))])


def _validate_vectors(vectors):
    if not np.isfinite(vectors).all() or not np.any(vectors != 0, axis=1).all():
        raise ValueError('Source posting vectors must be finite and nonzero')


def decode_fp16(raw: bytes, dimensions: int = 384):
    """Read original int32 code, int16 page, and float16 vectors in source order."""
    _validate_dimensions(dimensions)
    dtype = record_dtype(dimensions)
    if len(raw) % dtype.itemsize:
        raise ValueError('Truncated float16 posting record')
    records = np.frombuffer(raw, dtype=dtype)
    vectors = records['vec'].astype(np.float32)
    _validate_vectors(vectors)
    return records['code'].copy(), records['page'].copy(), vectors


def encode_posting(raw: bytes, dimensions: int = 384, bits: int = 6) -> bytes:
    """Encode one posting using an FP16 local mean and per-row affine codes."""
    _validate_dimensions(dimensions)
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError('Posting quantization must use between 2 and 8 bits')
    source_dtype = record_dtype(dimensions)
    if len(raw) % source_dtype.itemsize:
        raise ValueError('Truncated float16 posting record')
    source = np.frombuffer(raw, dtype=source_dtype)
    count = len(source)
    if count > np.iinfo(np.uint32).max:
        raise ValueError('Posting record count must fit uint32')
    with np.errstate(over='ignore', invalid='ignore'):
        mean = source['vec'].mean(axis=0, dtype=np.float32).astype('<f2') if count else np.zeros(dimensions, '<f2')
    if not np.isfinite(mean).all():
        raise ValueError('Posting mean cannot be represented as finite float16')
    output = bytearray(_HEADER.pack(_MAGIC, dimensions, bits, 1, count))
    output.extend(mean.tobytes())
    mean = mean.astype(np.float32)
    dtype, levels = _quantized_dtype(dimensions, bits), (1 << bits) - 1
    for first in range(0, count, _BATCH_ROWS):
        block = source[first:first + _BATCH_ROWS]
        values = block['vec'].astype(np.float32)
        _validate_vectors(values)
        centered = values - mean
        with np.errstate(over='ignore'):
            lower = centered.min(axis=1).astype('<f2')
            upper = centered.max(axis=1).astype('<f2')
            delta = np.maximum((upper.astype(np.float32) - lower.astype(np.float32)) / levels,
                               np.finfo(np.float16).tiny).astype('<f2')
        if not np.isfinite(lower).all() or not np.isfinite(delta).all() or (delta <= 0).any():
            raise ValueError('Posting ranges cannot be represented as finite float16')
        codes = np.clip(np.rint((centered - lower.astype(np.float32)[:, None])
                               / delta.astype(np.float32)[:, None]), 0, levels).astype(np.uint8)
        encoded = np.empty(len(block), dtype=dtype)
        encoded['code'], encoded['page'] = block['code'], block['page']
        encoded['lower'], encoded['delta'] = lower, delta
        if bits == 8:
            encoded['packed'] = codes
        else:
            planes = (codes[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1
            encoded['packed'] = np.packbits(planes.reshape(len(block), -1), axis=1, bitorder='little')
        output.extend(encoded.tobytes())
    return bytes(output)


def decode_posting(payload: bytes, dimensions: int = 384):
    """Decode literal wire values to float32; callers apply cosine normalization."""
    _validate_dimensions(dimensions)
    if len(payload) < _HEADER.size:
        raise ValueError('Truncated quantized posting header')
    magic, stored_dimensions, bits, version, count = _HEADER.unpack_from(payload)
    if magic != _MAGIC or stored_dimensions != dimensions or not 2 <= bits <= 8 or version != 1:
        raise ValueError('Unsupported quantized posting header')
    dtype = _quantized_dtype(dimensions, bits)
    prefix = _HEADER.size + 2 * dimensions
    if len(payload) != prefix + count * dtype.itemsize:
        raise ValueError('Invalid quantized posting size')
    mean = np.frombuffer(payload, '<f2', count=dimensions, offset=_HEADER.size).astype(np.float64)
    if not np.isfinite(mean).all():
        raise ValueError('Quantized posting mean must be finite')
    records = np.frombuffer(payload, dtype=dtype, count=count, offset=prefix)
    vectors = np.empty((count, dimensions), dtype=np.float32)
    used_bits = dimensions * bits % 8
    for first in range(0, count, _BATCH_ROWS):
        block = records[first:first + _BATCH_ROWS]
        lower, delta = block['lower'].astype(np.float64), block['delta'].astype(np.float64)
        if not np.isfinite(lower).all() or not np.isfinite(delta).all() or (delta <= 0).any():
            raise ValueError('Quantized posting ranges must be finite with positive scales')
        packed = block['packed']
        if used_bits and (packed[:, -1] >> used_bits).any():
            raise ValueError('Quantized posting row padding must be zero')
        if bits == 8:
            codes = packed
        else:
            planes = np.unpackbits(packed, axis=1, count=dimensions * bits, bitorder='little')
            codes = np.sum(planes.reshape(len(block), dimensions, bits) << np.arange(bits, dtype=np.uint8),
                           axis=2, dtype=np.uint8)
        vectors[first:first + len(block)] = (lower[:, None] + codes * delta[:, None] + mean).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError('Quantized posting reconstruction must be finite')
    return records['code'].copy(), records['page'].copy(), vectors
