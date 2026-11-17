"""GGUF block quantization and native PyTorch FP8 encodings for vector trials."""

from contextlib import contextmanager
import struct

import numpy as np
import torch


METHODS = (
    'gguf_q4_0', 'gguf_q4_1', 'gguf_q5_0', 'gguf_q5_1', 'gguf_q8_0',
) + tuple(f'fp8_{kind}_{scale}' for kind in ('e4m3', 'e5m2')
          for scale in ('none', 'row', 'b32', 'b64', 'b128'))
_HEADER = struct.Struct('<8sBBIH')
_MAGIC = b'EXTPOST\0'


def _batch_rows(dimensions):
    return min(4096, max(1, 16 * 1024 ** 2 // (4 * dimensions)))


@contextmanager
def _torch_threads():
    previous = torch.get_num_threads()
    if previous > 8:
        torch.set_num_threads(8)
    try:
        yield
    finally:
        if torch.get_num_threads() != previous:
            torch.set_num_threads(previous)


def _gguf_spec(method, dimensions):
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    qtype = GGMLQuantizationType[method[5:].upper()]
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if dimensions % block_size:
        raise ValueError(f'{method} dimensions must be a multiple of {block_size}')
    return qtype, dimensions // block_size * type_size


def _fp8_spec(method, dimensions):
    _, kind, scale = method.split('_')
    dtype = torch.float8_e4m3fn if kind == 'e4m3' else torch.float8_e5m2
    block_size = 0 if scale == 'none' else dimensions if scale == 'row' else int(scale[1:])
    groups = (dimensions + block_size - 1) // block_size if block_size else 0
    row_dtype = np.dtype([('scales', '<f4', (groups,)), ('codes', 'u1', (dimensions,))])
    return dtype, block_size, groups, row_dtype


def _finite(values):
    if not np.isfinite(values).all():
        raise ValueError('Posting vectors and reconstructions must be finite')


def encode(vectors, method: str) -> bytes:
    """Encode float32 vectors; IDs/pages are handled separately by the caller."""
    values = np.asarray(vectors)
    if (method not in METHODS or values.ndim != 2 or values.dtype.kind not in 'fiu'
            or not 1 <= values.shape[1] <= 65535 or len(values) > np.iinfo(np.uint32).max):
        raise ValueError('Expected a known method and a numeric vector matrix with uint16 dimensions')
    count, dimensions = values.shape
    output = bytearray(_HEADER.pack(_MAGIC, 1, METHODS.index(method), count, dimensions))
    if method.startswith('gguf_'):
        from gguf.quants import quantize

        qtype, _ = _gguf_spec(method, dimensions)
        for first in range(0, count, _batch_rows(dimensions)):
            block = np.ascontiguousarray(values[first:first + _batch_rows(dimensions)], dtype=np.float32)
            _finite(block)
            if (np.abs(block) > np.finfo(np.float16).max).any():
                raise ValueError('GGUF input exceeds the original float16 source range')
            output.extend(quantize(block, qtype).tobytes())
        return bytes(output)

    dtype, block_size, groups, row_dtype = _fp8_spec(method, dimensions)
    maximum = torch.finfo(dtype).max
    with _torch_threads(), torch.inference_mode():
        for first in range(0, count, _batch_rows(dimensions)):
            block = np.ascontiguousarray(values[first:first + _batch_rows(dimensions)], dtype=np.float32)
            _finite(block)
            encoded = np.empty(len(block), dtype=row_dtype)
            if groups:
                if groups * block_size == dimensions:
                    grouped = block.reshape(len(block), groups, block_size)
                else:
                    padded = np.zeros((len(block), groups * block_size), np.float32)
                    padded[:, :dimensions] = block
                    grouped = padded.reshape(len(block), groups, block_size)
                magnitude = np.max(np.abs(grouped), axis=2)
                scales = np.where(magnitude == 0, 1, magnitude / maximum).astype('<f4')
                if not np.isfinite(scales).all() or (scales <= 0).any():
                    raise ValueError('FP8 scales must be finite and positive')
                encoded['scales'] = scales
                block = np.ascontiguousarray((grouped / scales[:, :, None]).reshape(len(block), -1)[:, :dimensions])
            quantized = torch.from_numpy(block).to(dtype)
            _finite(quantized.to(torch.float32).numpy())
            encoded['codes'] = quantized.view(torch.uint8).numpy()
            output.extend(encoded.tobytes())
    return bytes(output)


def decode(payload: bytes) -> np.ndarray:
    """Restore original-scale float32 vectors using the exact stored method."""
    if len(payload) < _HEADER.size:
        raise ValueError('Truncated external posting header')
    magic, version, method_id, count, dimensions = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != 1 or method_id >= len(METHODS) or dimensions == 0:
        raise ValueError('Invalid external posting header')
    method = METHODS[method_id]
    if method.startswith('gguf_'):
        from gguf.quants import dequantize

        qtype, row_bytes = _gguf_spec(method, dimensions)
        if len(payload) != _HEADER.size + count * row_bytes:
            raise ValueError('Invalid GGUF posting size')
        vectors = np.empty((count, dimensions), np.float32)
        for first in range(0, count, _batch_rows(dimensions)):
            length = min(_batch_rows(dimensions), count - first)
            packed = np.frombuffer(payload, np.uint8, count=length * row_bytes,
                                   offset=_HEADER.size + first * row_bytes).reshape(length, row_bytes)
            decoded = dequantize(packed, qtype)
            _finite(decoded)
            vectors[first:first + length] = decoded
        return vectors

    dtype, block_size, groups, row_dtype = _fp8_spec(method, dimensions)
    if len(payload) != _HEADER.size + count * row_dtype.itemsize:
        raise ValueError('Invalid FP8 posting size')
    records = np.frombuffer(payload, row_dtype, count=count, offset=_HEADER.size)
    vectors = np.empty((count, dimensions), np.float32)
    with _torch_threads(), torch.inference_mode():
        for first in range(0, count, _batch_rows(dimensions)):
            block = records[first:first + _batch_rows(dimensions)]
            decoded = torch.from_numpy(block['codes'].copy()).view(dtype).to(torch.float32).numpy()
            if groups:
                scales = block['scales']
                if not np.isfinite(scales).all() or (scales <= 0).any():
                    raise ValueError('FP8 scales must be finite and positive')
                decoded *= np.repeat(scales, block_size, axis=1)[:, :dimensions]
            _finite(decoded)
            vectors[first:first + len(block)] = decoded
    return vectors
