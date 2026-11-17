#!/usr/bin/env python3
"""Split a gzip centroid index into eight byte slices or native HTTP Brotli parts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import zlib


PART_COUNT = 8
MAX_PART_BYTES = 25 * 1024 ** 2


def split(source, output, http_brotli=False):
    source, output = Path(source), Path(output)
    size = source.stat().st_size
    if not PART_COUNT <= size <= PART_COUNT * MAX_PART_BYTES:
        raise ValueError('Index must fit eight nonempty parts of at most 25 MiB each')
    compressed = source.read_bytes()
    if len(compressed) != size:
        raise ValueError('Source index changed while reading')
    try:
        payload = gzip.decompress(compressed)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError('Source must be a complete gzip stream') from error
    if len(payload) < 16:
        raise ValueError('Truncated centroid header')
    magic, version, dimensions, count = struct.unpack_from('<8sHHI', payload)
    bits = {b'LVQ6IDX\0': 6, b'LVQ8IDX\0': 8}.get(magic)
    if bits is None or version not in (1, 2) or not dimensions or not count:
        raise ValueError('Expected an LVQ6/LVQ8 version 1 or 2 centroid index')
    expected = 16 + dimensions * 2 + count * 4 + count * ((dimensions * bits + 7) // 8)
    if len(payload) != expected:
        raise ValueError('Centroid payload size does not match its header')
    if output.is_symlink() or output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory; existing releases are never replaced')
    if http_brotli:
        if len(payload) > PART_COUNT * MAX_PART_BYTES:
            raise ValueError('Raw index must fit eight parts of at most 25 MiB each')
        import brotli

    output.mkdir(parents=True, exist_ok=True)
    if http_brotli:
        manifest = {'schema_version': 2, 'compression': 'http', 'raw_bytes': len(payload),
                    'raw_sha256': hashlib.sha256(payload).hexdigest(), 'compressed_bytes': 0,
                    'source_gzip_sha256': hashlib.sha256(compressed).hexdigest(), 'parts': []}
    else:
        manifest = {'schema_version': 1, 'gzip_bytes': size,
                    'sha256': hashlib.sha256(compressed).hexdigest(), 'parts': []}
    content = payload if http_brotli else compressed
    width, extra = divmod(len(content), PART_COUNT)
    offset = 0
    for index in range(PART_COUNT):
        chunk = content[offset:offset + width + (index < extra)]
        data = brotli.compress(chunk, quality=6) if http_brotli else chunk
        if len(data) > MAX_PART_BYTES:
            raise ValueError('Compressed part exceeds the 25 MiB asset limit')
        name = f'part-{index}.bin' if http_brotli else f'centroids.lvq{bits}.bin.gz.part-{index}'
        with (output / name).open('xb') as target:
            target.write(data)
        if http_brotli:
            manifest['parts'].append({'file': name, 'raw_bytes': len(chunk),
                                      'sha256': hashlib.sha256(chunk).hexdigest(), 'compressed_bytes': len(data)})
            manifest['compressed_bytes'] += len(data)
        else:
            manifest['parts'].append({'file': name, 'bytes': len(data),
                                      'sha256': hashlib.sha256(data).hexdigest()})
        offset += len(chunk)
    if http_brotli:
        (output / '_headers').write_text(
            '/*\n'
            '  Access-Control-Allow-Origin: *\n'
            '  Timing-Allow-Origin: *\n'
            '  Access-Control-Expose-Headers: Content-Encoding\n'
            '  Cross-Origin-Resource-Policy: cross-origin\n'
            '\n/*.bin\n'
            '  Content-Encoding: br\n'
            '  Content-Type: application/octet-stream\n'
            '  Cache-Control: public, max-age=31536000, immutable, no-transform\n')
    temporary = output / 'centroid-parts.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(output / 'centroid-parts.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--http-brotli', action='store_true',
                        help='Compress raw slices independently for native HTTP decoding on static Pages')
    args = parser.parse_args()
    manifest = split(args.source, args.output, http_brotli=args.http_brotli)
    if args.http_brotli:
        print(f"Split {manifest['raw_bytes']} raw bytes into eight HTTP Brotli parts "
              f"({manifest['compressed_bytes']} compressed bytes) in {args.output}")
    else:
        print(f"Split {manifest['gzip_bytes']} gzip bytes into eight parts in {args.output}")


if __name__ == '__main__':
    main()
