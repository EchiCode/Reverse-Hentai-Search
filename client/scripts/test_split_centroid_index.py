"""Byte-preserving and optional native HTTP Brotli centroid release splitting."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name('split-centroid-index.py')
spec = importlib.util.spec_from_file_location('split_centroid_index', SCRIPT)
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


def payload(bits=6, version=2):
    dimensions, count = 4, 32
    return (struct.pack('<8sHHI', f'LVQ{bits}IDX\0'.encode(), version, dimensions, count)
            + bytes(dimensions * 2 + count * 2) + struct.pack('<e', 1) * count
            + random.Random(42).randbytes(count * dimensions * bits // 8))


class SplitCentroidIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='centroid-parts-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source.gz'

    def test_eight_parts_reassemble_original_gzip_and_have_verified_manifest(self):
        for bits in (6, 8):
            for version in (1, 2):
                with self.subTest(bits=bits, version=version):
                    raw = payload(bits, version)
                    compressed = gzip.compress(raw, mtime=0)
                    self.source.write_bytes(compressed)
                    output = self.root / f'parts-{bits}-{version}'
                    manifest = splitter.split(self.source, output)
                    self.assertEqual(json.loads((output / 'centroid-parts.json').read_text()), manifest)
                    self.assertEqual(manifest['schema_version'], 1)
                    self.assertEqual(manifest['gzip_bytes'], len(compressed))
                    self.assertEqual(manifest['sha256'], hashlib.sha256(compressed).hexdigest())
                    self.assertEqual(len(manifest['parts']), 8)
                    joined = bytearray()
                    sizes = []
                    for index, part in enumerate(manifest['parts']):
                        self.assertEqual(part['file'], f'centroids.lvq{bits}.bin.gz.part-{index}')
                        data = (output / part['file']).read_bytes()
                        self.assertEqual(part['bytes'], len(data))
                        self.assertEqual(part['sha256'], hashlib.sha256(data).hexdigest())
                        self.assertGreater(len(data), 0)
                        self.assertLessEqual(len(data), splitter.MAX_PART_BYTES)
                        sizes.append(len(data))
                        joined.extend(data)
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                    self.assertEqual(joined, compressed)
                    self.assertEqual(gzip.decompress(joined), raw)
                    self.assertEqual(self.source.read_bytes(), compressed)
                    self.assertEqual(len(list(output.iterdir())), 9)

    def test_rejects_bad_source_before_creating_output(self):
        raw = payload()
        invalid = [b'bad', b'not gzip' * 4, gzip.compress(raw)[:-1], gzip.compress(raw[:-1]),
                   gzip.compress(raw + b'\0'), gzip.compress(b'short'),
                   gzip.compress(b'BADMAGIC' + raw[8:]), gzip.compress(payload(version=3)),
                   gzip.compress(raw[:10] + b'\0\0' + raw[12:]),
                   gzip.compress(raw[:12] + b'\0\0\0\0' + raw[16:])]
        output = self.root / 'invalid'
        for data in invalid:
            with self.subTest(data=data[:16]):
                self.source.write_bytes(data)
                with self.assertRaises(ValueError):
                    splitter.split(self.source, output)
                self.assertFalse(output.exists())
        self.source.write_bytes(gzip.compress(raw))
        with patch.object(splitter, 'MAX_PART_BYTES', 1), self.assertRaises(ValueError):
            splitter.split(self.source, output)
        self.assertFalse(output.exists())

    def test_existing_release_is_unchanged_and_empty_output_is_allowed(self):
        self.source.write_bytes(gzip.compress(payload()))
        output = self.root / 'existing'
        output.mkdir()
        (output / 'centroid-parts.json').write_text('existing release')
        with self.assertRaises(ValueError):
            splitter.split(self.source, output)
        self.assertEqual((output / 'centroid-parts.json').read_text(), 'existing release')
        self.assertEqual(len(list(output.iterdir())), 1)
        (output / 'centroid-parts.json').unlink()
        splitter.split(self.source, output)
        with self.assertRaises(ValueError):
            splitter.split(self.source, output)

    def test_http_brotli_parts_decode_to_original_payload_and_publish_headers(self):
        try:
            import brotli
        except ModuleNotFoundError:
            self.skipTest('Optional HTTP Brotli export needs the installed brotli package')
        for bits in (6, 8):
            with self.subTest(bits=bits):
                raw = payload(bits)
                compressed = gzip.compress(raw, mtime=0)
                self.source.write_bytes(compressed)
                output = self.root / f'brotli-{bits}'
                manifest = splitter.split(self.source, output, http_brotli=True)
                self.assertEqual(manifest['schema_version'], 2)
                self.assertEqual(manifest['compression'], 'http')
                self.assertEqual(manifest['raw_bytes'], len(raw))
                self.assertEqual(manifest['raw_sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(manifest['source_gzip_sha256'], hashlib.sha256(compressed).hexdigest())
                self.assertEqual(len(manifest['parts']), 8)
                restored, compressed_bytes = bytearray(), 0
                for index, part in enumerate(manifest['parts']):
                    self.assertEqual(part['file'], f'part-{index}.bin')
                    data = (output / part['file']).read_bytes()
                    decoded = brotli.decompress(data)
                    self.assertEqual(part['raw_bytes'], len(decoded))
                    self.assertEqual(part['compressed_bytes'], len(data))
                    self.assertEqual(part['sha256'], hashlib.sha256(decoded).hexdigest())
                    self.assertLessEqual(len(data), splitter.MAX_PART_BYTES)
                    self.assertLessEqual(len(decoded), splitter.MAX_PART_BYTES)
                    restored.extend(decoded)
                    compressed_bytes += len(data)
                self.assertEqual(restored, raw)
                self.assertEqual(manifest['compressed_bytes'], compressed_bytes)
                self.assertEqual(self.source.read_bytes(), compressed)
                self.assertEqual(json.loads((output / 'centroid-parts.json').read_text()), manifest)
                headers = (output / '_headers').read_text()
                global_headers, binary_headers = headers.split('\n/*.bin\n')
                self.assertIn('Access-Control-Allow-Origin: *', global_headers)
                self.assertIn('Timing-Allow-Origin: *', global_headers)
                self.assertIn('Cross-Origin-Resource-Policy: cross-origin', global_headers)
                self.assertNotIn('Content-Encoding', global_headers.split('Expose-Headers:')[0])
                self.assertIn('Content-Encoding: br', binary_headers)
                self.assertIn('Content-Type: application/octet-stream', binary_headers)
                self.assertIn('max-age=31536000, immutable, no-transform', binary_headers)
                with self.assertRaises(ValueError):
                    splitter.split(self.source, output, http_brotli=True)
        cli_output = self.root / 'cli-brotli'
        completed = subprocess.run([sys.executable, str(SCRIPT), str(self.source), str(cli_output),
                                    '--http-brotli'], check=True, text=True, capture_output=True)
        self.assertIn('HTTP Brotli parts', completed.stdout)
        self.assertEqual(json.loads((cli_output / 'centroid-parts.json').read_text())['schema_version'], 2)

    def test_http_brotli_size_limits_do_not_publish_a_manifest(self):
        try:
            import brotli
        except ModuleNotFoundError:
            self.skipTest('Optional HTTP Brotli export needs the installed brotli package')
        raw = payload()
        compressed = gzip.compress(raw, mtime=0)
        self.source.write_bytes(compressed)
        output = self.root / 'raw-too-large'
        raw_limit = (len(compressed) + 7) // 8
        self.assertGreater(len(raw), raw_limit * 8)
        with patch.object(splitter, 'MAX_PART_BYTES', raw_limit), self.assertRaises(ValueError):
            splitter.split(self.source, output, http_brotli=True)
        self.assertFalse(output.exists())
        compressed_limit = (len(raw) + 7) // 8
        output = self.root / 'compressed-too-large'
        with patch.object(splitter, 'MAX_PART_BYTES', compressed_limit), \
                patch.object(brotli, 'compress', return_value=b'x' * (compressed_limit + 1)), \
                self.assertRaises(ValueError):
            splitter.split(self.source, output, http_brotli=True)
        self.assertFalse((output / 'centroid-parts.json').exists())


if __name__ == '__main__':
    unittest.main()
