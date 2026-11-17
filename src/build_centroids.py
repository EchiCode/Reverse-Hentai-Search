"""Train only centroid indexes and compare lossless whole-file compression methods."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zlib

import brotli
import numpy as np

from balanced_l2.build import source_rows, train
from balanced_l2.common import artifact, input_files, iter_record_batches, require_disk_space, snapshot, write_json
from balanced_l2.export import export_centroids


PAGES_LIMIT_BYTES = 25 * 1024 ** 2


def build(paths='vectors', output='centroid_download_comparison', heads=(80000, 96000), *,
          sample_size=2500000, dim=384, max_iter=20):
    heads = list(heads)
    if (not heads or len(set(heads)) != len(heads)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in [sample_size, dim, max_iter, *heads]) or dim > 65535):
        raise ValueError('Expected distinct positive head counts and positive training parameters')
    files = input_files(paths)
    inputs = [snapshot(path) for path in files]
    output = Path(output)
    if output.is_symlink():
        raise ValueError('Output must not be a symlink')
    output = output.resolve()

    def unchanged():
        if inputs != [snapshot(path) for path in files]:
            raise ValueError('Original vector files changed during centroid preparation')

    total_rows = source_rows(files, dim)
    rows = min(total_rows, sample_size)
    if any(not 2 <= count <= rows for count in heads):
        raise ValueError('Head counts must be between two and the training sample rows')
    output.mkdir(parents=True, exist_ok=False)
    require_disk_space(output, rows * dim * 4 + sum(heads) * dim * 4 + 16)
    state = {'schema_version': 1, 'status': 'preparing_sample', 'inputs': inputs,
             'source_rows': total_rows, 'dimensions': dim, 'head_counts': heads,
             'sample_rows': rows, 'sample_seed': 42, 'max_iter': max_iter,
             'pages_limit_bytes': PAGES_LIMIT_BYTES,
             'timing_note': 'Local CPU codec timings; browser delivery and decoder support are not tested.',
             'completed': []}
    write_json(output / 'results.json', state)
    selected = np.sort(np.random.default_rng(42).choice(total_rows, rows, replace=False))
    sample = output / 'sample.f32'
    temporary = sample.with_suffix('.tmp')
    print(f'Preparing one shared sample: {rows:,} rows, {rows * dim * 4 + 8:,} bytes', flush=True)
    offset = 0
    with temporary.open('xb') as stream:
        stream.write(struct.pack('<ii', rows, dim))
        # Read bounded batches and release the source-file cache instead of mapping the whole corpus.
        for records in iter_record_batches(files, dim, 65536):
            left, right = np.searchsorted(selected, [offset, offset + len(records)])
            if left < right:
                values = records['vec'][selected[left:right] - offset].astype(np.float32)
                norms = np.linalg.norm(values, axis=-1)
                if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
                    raise ValueError('Source vectors must be finite and nonzero')
                (values / norms[..., None]).astype('<f4', copy=False).tofile(stream)
            offset += len(records)
    unchanged()
    if offset != total_rows:
        raise ValueError('Original vector row count changed during sampling')
    temporary.replace(sample)
    del selected, records, values, norms
    state.update(status='training', sample=artifact(sample))
    write_json(output / 'results.json', state)
    zstd = shutil.which('zstd')
    state['skipped_compressions'] = {}
    if zstd is None:
        state['skipped_compressions']['zstd'] = 'zstd executable is not installed'
    lz4_frame = None
    if importlib.util.find_spec('lz4') is not None:
        import lz4.frame as lz4_frame
    else:
        state['skipped_compressions']['lz4'] = 'Optional python-lz4 package is not installed'
    compressors = {
        'brotli': lambda raw, level: brotli.compress(raw, quality=level),
        'zstd': lambda raw, level: subprocess.run([zstd, '-q', '-c', f'-{level}', '-T1'],
                                                  input=raw, stdout=subprocess.PIPE, check=True).stdout,
        'xz': lambda raw, level: lzma.compress(raw, format=lzma.FORMAT_XZ, preset=level),
        'zlib': lambda raw, level: zlib.compress(raw, level=level),
    }
    decompressors = {'brotli': brotli.decompress, 'xz': lzma.decompress, 'zlib': zlib.decompress,
                     'zstd': lambda data: subprocess.run([zstd, '-d', '-q', '-c'], input=data,
                                                        stdout=subprocess.PIPE, check=True).stdout}
    if lz4_frame:
        compressors['lz4'] = lambda raw, level: lz4_frame.compress(raw, compression_level=level)
        decompressors['lz4'] = lz4_frame.decompress
    http_encodings = {'gzip': 'gzip', 'brotli': 'br', 'zstd': 'zstd', 'zlib': 'deflate'}
    for count in heads:
        unchanged()
        directory = output / f'h{count}'
        native = directory / 'native'
        print(f'Training {count:,} centroids only; no assignment or posting export', flush=True)
        train(sample, native, count, max_iter)
        started = time.perf_counter()
        export_centroids(directory, native / 'heads.f32')
        export_seconds = time.perf_counter() - started
        gzip_path = directory / 'centroids.lvq6.bin.gz'
        compressed_gzip = gzip_path.read_bytes()
        started = time.perf_counter()
        raw = gzip.decompress(compressed_gzip)
        gzip_decompression_seconds = time.perf_counter() - started
        row = {'head_count': count, 'centroid_method': 'lvq6', 'metric': 'l2',
               'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
               'compressions': [{'codec': 'gzip', 'level': 9, **artifact(gzip_path),
                                 'fits_pages': gzip_path.stat().st_size <= PAGES_LIMIT_BYTES,
                                 'compression_seconds': None, 'export_seconds': export_seconds,
                                 'decompression_seconds': gzip_decompression_seconds,
                                 'http_content_encoding': 'gzip',
                                 'extra_browser_decoder_required': False,
                                 'timing_scope': 'FP16 and LVQ6 encoding plus gzip export'}]}
        configurations = [('brotli', 6, 'br6'), ('brotli', 11, 'br11')]
        if zstd:
            configurations.extend([('zstd', 3, 'zst3'), ('zstd', 19, 'zst19')])
        configurations.extend([('xz', 9, 'xz9'), ('zlib', 9, 'zlib')])
        if lz4_frame:
            configurations.append(('lz4', 12, 'lz4'))
        for codec, level, suffix in configurations:
            started = time.perf_counter()
            compressed = compressors[codec](raw, level)
            seconds = time.perf_counter() - started
            started = time.perf_counter()
            restored = decompressors[codec](compressed)
            decompression_seconds = time.perf_counter() - started
            if restored != raw:
                raise ValueError(f'{codec} round-trip changed the centroid payload')
            destination = directory / f'centroids.lvq6.bin.{suffix}'
            temporary = destination.with_suffix('.tmp')
            temporary.write_bytes(compressed)
            temporary.replace(destination)
            row['compressions'].append({'codec': codec, 'level': level, **artifact(destination),
                                        'fits_pages': len(compressed) <= PAGES_LIMIT_BYTES,
                                        'compression_seconds': seconds,
                                        'decompression_seconds': decompression_seconds,
                                        'http_content_encoding': http_encodings.get(codec),
                                        'extra_browser_decoder_required': codec not in http_encodings})
        unchanged()
        state['completed'].append(row)
        write_json(output / 'results.json', state)
        print(f'{count:,}: ' + ', '.join(
            f"{item['codec']}{item['level']} {item['bytes']:,} bytes (fits: {item['fits_pages']})"
            for item in row['compressions']), flush=True)
    state['status'] = 'complete'
    write_json(output / 'results.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', default=['vectors'])
    parser.add_argument('--output', default='centroid_download_comparison')
    parser.add_argument('--heads', nargs='+', type=int, default=[80000, 96000])
    parser.add_argument('--sample-size', type=int, default=2500000)
    parser.add_argument('--max-iter', type=int, default=20)
    args = parser.parse_args()
    result = build(args.input, args.output, args.heads, sample_size=args.sample_size, max_iter=args.max_iter)
    print(json.dumps({'status': result['status'], 'results': str(Path(args.output).resolve() / 'results.json')}, indent=2))
