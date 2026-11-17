"""Standalone centroid wire-format checks and legacy compatibility."""

import gzip
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name('encode-centroids.py')
spec = importlib.util.spec_from_file_location('client_centroid_encoder', SCRIPT)
encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder)


class ClientCentroidEncoderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='client-lvq-export-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'centroids.f16.bin.gz'

    def test_l2_metric_changes_only_the_version_and_preserves_row_padding(self):
        rng = np.random.default_rng(72)
        for dimensions in (1, 3, 5, 7, 384):
            values = rng.normal(size=(13, dimensions)).astype('<f2')
            self.source.write_bytes(gzip.compress(values.tobytes(), mtime=0))
            for bits in (6, 8):
                with self.subTest(dimensions=dimensions, bits=bits):
                    actual, legacy = self.root / 'actual.gz', self.root / 'legacy.gz'
                    count, raw_size, gzip_size = encoder.encode(self.source, actual, dimensions, bits, metric='l2')
                    encoder.encode(self.source, legacy, dimensions, bits, metric='ip')
                    payload = gzip.decompress(actual.read_bytes())
                    self.assertEqual((count, raw_size, gzip_size), (13, len(payload), actual.stat().st_size))
                    self.assertEqual(struct.unpack_from('<8sHHI', payload), (f'LVQ{bits}IDX\0'.encode(), 2, dimensions, 13))
                    old_payload = gzip.decompress(legacy.read_bytes())
                    self.assertEqual(payload[:8] + payload[10:], old_payload[:8] + old_payload[10:])
                    if bits == 6 and dimensions * bits % 8:
                        row_bytes = (dimensions * bits + 7) // 8
                        codes = np.frombuffer(payload[16 + 2 * dimensions + 4 * count:], dtype=np.uint8).reshape(count, row_bytes)
                        self.assertTrue((codes[:, -1] >> (dimensions * bits % 8) == 0).all())

    def test_explicit_ip_and_default_function_preserve_golden_payloads(self):
        values = np.array([[0, .25, -.25, .75, -.5], [.5, -.25, .125, -.75, .5]], dtype='<f2')
        self.source.write_bytes(gzip.compress(values.tobytes(), mtime=0))
        output = self.root / 'legacy.gz'
        encoder.encode(self.source, output, 5)
        self.assertEqual(gzip.decompress(output.read_bytes()).hex(),
                         '4c5651384944580001000500020000000034000000ac0000000000b800ba051d051d339940ff00cc66bf00ff')
        encoder.encode(self.source, output, 5, bits=6, metric='ip')
        self.assertEqual(gzip.decompress(output.read_bytes()).hex(),
                         '4c5651364944580001000500020000000034000000ac0000000000b800ba142514258d09fd0072f6023f')

    def test_cli_defaults_to_l2_lvq6_and_can_explicitly_write_legacy_ip_lvq8(self):
        self.source.write_bytes(gzip.compress(np.eye(3, dtype='<f2').tobytes(), mtime=0))
        for arguments, bits, version in (([], 6, 2), (['--bits', '8', '--metric', 'ip'], 8, 1)):
            subprocess.run([sys.executable, str(SCRIPT.resolve()), '--dimensions', '3', *arguments],
                           cwd=self.root, check=True, capture_output=True, text=True)
            output = self.root / f'centroids.lvq{bits}.bin.gz'
            self.assertEqual(struct.unpack_from('<8sH', gzip.decompress(output.read_bytes())), (f'LVQ{bits}IDX\0'.encode(), version))

    def test_invalid_input_preserves_existing_output(self):
        output = self.root / 'good.gz'
        output.write_bytes(b'previous valid output')
        cases = [(b'', 3, 6), (b'\0', 3, 6), (np.array([np.nan], dtype='<f2').tobytes(), 1, 6),
                 (np.array([np.inf], dtype='<f2').tobytes(), 1, 8),
                 (np.zeros(3, dtype='<f2').tobytes(), 0, 6),
                 (np.zeros(3, dtype='<f2').tobytes(), 3, 4),
                 (np.array([[-65504, 65504], [65504, -65504], [65504, -65504]], dtype='<f2').tobytes(), 2, 6)]
        for raw, dimensions, bits in cases:
            self.source.write_bytes(gzip.compress(raw, mtime=0))
            with self.subTest(dimensions=dimensions, bits=bits), self.assertRaises(ValueError):
                encoder.encode(self.source, output, dimensions, bits)
            self.assertEqual(output.read_bytes(), b'previous valid output')
        self.assertFalse(output.with_name(output.name + '.tmp').exists())
        with self.assertRaises(ValueError):
            encoder.encode(self.source, output, 3, metric='unknown')


if __name__ == '__main__':
    unittest.main()
