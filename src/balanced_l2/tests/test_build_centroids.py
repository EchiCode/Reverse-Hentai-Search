"""Centroid-only preparation checks with a tiny CPU fixture and mocked GPU training."""

import gzip
import json
import lzma
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib
from unittest.mock import patch

import brotli
import numpy as np

import build_centroids
from balanced_l2.common import artifact, read_matrix, record_dtype, write_json


class BuildCentroidsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='centroid-only-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'vectors'
        self.source.mkdir()
        self.records = np.zeros(15, dtype=record_dtype(3))
        self.records['vec'] = np.random.default_rng(3).normal(size=(15, 3)).astype('<f2')
        self.records[:6].tofile(self.source / '2.bin')
        self.records[6:].tofile(self.source / '10.bin')

    def test_one_shared_sample_trains_requested_counts_and_saves_only_indexes(self):
        output = self.root / 'comparison'
        calls = []

        def fake_train(sample, native, count, max_iter):
            values = read_matrix(sample)
            calls.append((count, max_iter, values.copy()))
            native.mkdir(parents=True)
            heads = native / 'heads.f32'
            heads.write_bytes(struct.pack('<ii', count, 3) + values[:count].astype('<f4').tobytes())
            write_json(native / 'training.json', {'head_count': count, 'sample': artifact(sample)})

        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        original_batches = build_centroids.iter_record_batches
        with patch.object(build_centroids, 'train', side_effect=fake_train), patch.object(
                build_centroids, 'iter_record_batches',
                side_effect=lambda files, dim, batch_size: original_batches(files, dim, 4)):
            result = build_centroids.build(self.source, output, [2, 3], sample_size=11, dim=3, max_iter=4)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual([call[:2] for call in calls], [(2, 4), (3, 4)])
        np.testing.assert_array_equal(calls[0][2], calls[1][2])
        selected = np.sort(np.random.default_rng(42).choice(15, 11, replace=False))
        expected = self.records['vec'][selected].astype(np.float32)
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_array_equal(read_matrix(output / 'sample.f32'), expected)
        self.assertEqual((output / 'sample.f32').stat().st_size, 8 + 11 * 3 * 4)
        self.assertEqual(json.loads((output / 'results.json').read_text()), result)
        for row in result['completed']:
            directory = output / f"h{row['head_count']}"
            raw = gzip.decompress((directory / 'centroids.lvq6.bin.gz').read_bytes())
            self.assertEqual(struct.unpack_from('<8sHHI', raw), (b'LVQ6IDX\0', 2, 3, row['head_count']))
            self.assertEqual([(item['codec'], item['level']) for item in row['compressions'][:3]],
                             [('gzip', 9), ('brotli', 6), ('brotli', 11)])
            for item in row['compressions']:
                compressed = Path(item['path']).read_bytes()
                if item['codec'] == 'gzip':
                    decoded = gzip.decompress(compressed)
                    self.assertIsNone(item['compression_seconds'])
                    self.assertIn('LVQ6 encoding', item['timing_scope'])
                elif item['codec'] == 'brotli':
                    decoded = brotli.decompress(compressed)
                elif item['codec'] == 'zstd':
                    decoded = subprocess.run([build_centroids.shutil.which('zstd'), '-d', '-q', '-c'],
                                             input=compressed, stdout=subprocess.PIPE, check=True).stdout
                elif item['codec'] == 'xz':
                    decoded = lzma.decompress(compressed)
                elif item['codec'] == 'zlib':
                    decoded = zlib.decompress(compressed)
                else:
                    import lz4.frame
                    decoded = lz4.frame.decompress(compressed)
                self.assertEqual(decoded, raw)
                self.assertTrue(item['fits_pages'])
                self.assertGreaterEqual(item['decompression_seconds'], 0)
            self.assertTrue({'xz', 'zlib'} <= {item['codec'] for item in row['compressions']})
            self.assertNotIn('bzip2', {item['codec'] for item in row['compressions']})
            for item in row['compressions']:
                if item['codec'] in ('xz', 'lz4'):
                    self.assertTrue(item['extra_browser_decoder_required'])
                    self.assertIsNone(item['http_content_encoding'])
            self.assertTrue((directory / 'native/training.json').exists())
            self.assertTrue((directory / 'native/heads.f32').exists())
        self.assertFalse(list(output.rglob('assignments*')))
        self.assertFalse(list(output.rglob('clusters')))
        self.assertFalse(list(output.rglob('vectors.f32')))
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.source.iterdir()})

    def test_invalid_counts_or_existing_output_never_call_training(self):
        output = self.root / 'comparison'
        with patch.object(build_centroids, 'train', side_effect=AssertionError('GPU training was called')):
            for heads in ([2, 2], [0], [16], [], [True]):
                with self.subTest(heads=heads), self.assertRaises(ValueError):
                    build_centroids.build(self.source, output, heads, sample_size=11, dim=3)
                self.assertFalse(output.exists())
            for dim in (0, 65536, '3', True):
                with self.subTest(dim=dim), self.assertRaises(ValueError):
                    build_centroids.build(self.source, output, [2], sample_size=11, dim=dim)
                self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                build_centroids.build(self.source, output, [2], sample_size=11, dim=3)


if __name__ == '__main__':
    unittest.main()
