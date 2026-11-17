#!/usr/bin/env python3
"""Encode float16 centroids as LVQ6/LVQ8 with an explicit routing metric."""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np


DEFAULT_DIMENSIONS = 384


def encode_values(centroids, bits: int = 8, metric: str = 'ip') -> bytes:
    """Encode in-memory centroids; the benchmark also uses five-bit rows."""
    with np.errstate(over='ignore', invalid='ignore'):
        centroids = np.asarray(centroids, dtype=np.float32)
    if (centroids.ndim != 2 or not 1 <= centroids.shape[1] <= 65535
            or not 1 <= len(centroids) <= np.iinfo(np.uint32).max
            or isinstance(bits, bool) or not isinstance(bits, int) or bits not in (5, 6, 8)
            or not np.isfinite(centroids).all()):
        raise ValueError('Expected finite centroid rows, uint16 dimensions and a bit depth of 5, 6 or 8')
    if metric not in ('ip', 'l2'):
        raise ValueError('Routing metric must be ip or l2')
    dimensions = centroids.shape[1]
    mean = centroids.mean(axis=0, keepdims=True).astype(np.float32)
    centered = centroids - mean
    levels = (1 << bits) - 1
    with np.errstate(over='ignore'):
        lower = centered.min(axis=1, keepdims=True).astype(np.float16)
        upper = centered.max(axis=1, keepdims=True).astype(np.float16)
        delta = np.maximum(
            (upper.astype(np.float32) - lower.astype(np.float32)) / levels,
            np.finfo(np.float16).tiny,
        ).astype(np.float16)
    if not np.isfinite(lower).all() or not np.isfinite(delta).all() or (delta <= 0).any():
        raise ValueError('The centroid ranges cannot be represented as finite float16')
    codes = np.clip(
        np.rint((centered - lower.astype(np.float32)) / delta.astype(np.float32)),
        0,
        levels,
    ).astype(np.uint8)
    if bits != 8:
        # Rows start at byte boundaries; unused high bits in the final byte are zero.
        planes = (codes[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1
        packed = np.packbits(planes.reshape(len(codes), -1), axis=1, bitorder='little').tobytes()
    else:
        packed = codes.tobytes()

    header = struct.pack('<8sHHI', f'LVQ{bits}IDX\0'.encode(), 2 if metric == 'l2' else 1, dimensions, len(centroids))
    with np.errstate(over='ignore', invalid='ignore'):
        stored_mean = mean.astype('<f2')
    if not np.isfinite(stored_mean).all():
        raise ValueError('The centroid mean cannot be represented as finite float16')
    payload = b''.join((
        header,
        stored_mean.tobytes(),
        lower.astype('<f2').tobytes(),
        delta.astype('<f2').tobytes(),
        packed,
    ))
    return payload


def encode(source: Path, destination: Path, dimensions: int, bits: int = 8,
           metric: str = 'ip') -> tuple[int, int, int]:
    if (isinstance(dimensions, bool) or not isinstance(dimensions, int)
            or not 1 <= dimensions <= 65535 or bits not in (6, 8)):
        raise ValueError('Dimensions must fit uint16 and the bit depth must be 6 or 8')
    if metric not in ('ip', 'l2'):
        raise ValueError('Routing metric must be ip or l2')
    raw = gzip.decompress(source.read_bytes())
    if not raw or len(raw) % (2 * dimensions):
        raise ValueError('The float16 source must contain complete, nonempty vectors')
    centroids = np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(-1, dimensions)
    payload = encode_values(centroids, bits, metric)
    encoded = gzip.compress(payload, compresslevel=9, mtime=0)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    return len(centroids), len(payload), len(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path, nargs='?', default=Path('centroids.f16.bin.gz'))
    parser.add_argument('destination', type=Path, nargs='?')
    parser.add_argument('--dimensions', type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument('--bits', type=int, choices=(6, 8), default=6)
    parser.add_argument('--metric', choices=('ip', 'l2'), default='l2')
    args = parser.parse_args()
    destination = args.destination or Path(f'centroids.lvq{args.bits}.bin.gz')
    count, raw_bytes, gzip_bytes = encode(args.source, destination, args.dimensions, args.bits, args.metric)
    print(f'Encoded {count} LVQ{args.bits} {args.metric} centroids: {raw_bytes} bytes raw, {gzip_bytes} bytes gzip')


if __name__ == '__main__':
    main()
