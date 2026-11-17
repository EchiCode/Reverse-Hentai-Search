"""Shared-source centroid codec and literal L2 wire-format checks."""

import gzip
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from balanced_l2 import centroid_quantization as codec
from balanced_l2.export import decode_centroids


class CentroidQuantizationTests(unittest.TestCase):
    def test_all_four_formats_share_fp16_source_and_preserve_nonunit_means(self):
        values = np.random.default_rng(9).normal(size=(13, 7)).astype(np.float32) + 2
        rounded = values.astype(np.float16).astype(np.float32)
        self.assertEqual(codec.METHODS, ('fp16', 'lvq8', 'lvq6', 'lvq5'))
        errors = []
        for method in codec.METHODS:
            with self.subTest(method=method):
                wire = codec.encode(values, method)
                self.assertEqual(wire, codec.encode(rounded, method))
                actual = codec.decode(wire, method, 13, 7)
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.shape, values.shape)
                self.assertTrue(np.isfinite(actual).all())
                self.assertGreater(float(np.linalg.norm(actual, axis=1).min()), 1)
                if method == 'fp16':
                    self.assertEqual(wire, values.astype('<f2').tobytes())
                    np.testing.assert_array_equal(actual, rounded)
                else:
                    errors.append(float(np.mean((actual - rounded) ** 2)))
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])

    def test_existing_lvq6_lvq8_encoder_wires_remain_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.random.default_rng(17).normal(size=(11, 7)).astype('<f2')
            source = root / 'source.gz'
            source.write_bytes(gzip.compress(values.tobytes(), mtime=0))
            for bits in (6, 8):
                path = root / f'lvq{bits}.gz'
                codec._ENCODER.encode(source, path, 7, bits, metric='l2')
                payload = codec.encode(values, f'lvq{bits}')
                self.assertEqual(payload, gzip.decompress(path.read_bytes()))
                if bits == 6:
                    np.testing.assert_array_equal(codec.decode(payload, 'lvq6', 11, 7),
                                                  decode_centroids(payload, 11, 7))

    def test_all_lvq_bit_depths_decode_literal_packed_values_and_zero_row_padding(self):
        values = np.random.default_rng(8).normal(size=(5, 7)).astype(np.float32)
        for bits in (5, 6, 8):
            with self.subTest(bits=bits), patch.object(codec, '_BATCH_ROWS', 2):
                payload = codec.encode(values, f'lvq{bits}')
                self.assertEqual(struct.unpack_from('<8sHHI', payload),
                                 (f'LVQ{bits}IDX\0'.encode(), 2, 7, 5))
                mean = np.frombuffer(payload, '<f2', 7, offset=16).astype(np.float64)
                lower = np.frombuffer(payload, '<f2', 5, offset=30).astype(np.float64)
                delta = np.frombuffer(payload, '<f2', 5, offset=40).astype(np.float64)
                row_bytes, prefix = (7 * bits + 7) // 8, 50
                expected = np.empty_like(values)
                for row in range(5):
                    packed = int.from_bytes(payload[prefix + row * row_bytes:prefix + (row + 1) * row_bytes], 'little')
                    self.assertEqual(packed >> (7 * bits), 0)
                    for dimension in range(7):
                        value = packed >> (dimension * bits) & ((1 << bits) - 1)
                        expected[row, dimension] = lower[row] + value * delta[row] + mean[dimension]
                np.testing.assert_array_equal(codec.decode(payload, f'lvq{bits}', 5, 7), expected)

    def test_mean_restoration_occurs_before_float32_rounding(self):
        for bits in (5, 6, 8):
            payload = struct.pack('<8sHHI3eB', f'LVQ{bits}IDX\0'.encode(), 2, 1, 1,
                                  -65504, 65504, 2**-14, 1)
            decoded = codec.decode(payload, f'lvq{bits}', 1, 1)
            self.assertEqual(decoded[0, 0], 2**-14)

    def test_invalid_input_shapes_counts_and_nonfinite_values_are_rejected(self):
        for values in (np.empty((0, 3)), np.empty((1, 0)), [1, 2], [[np.nan]], [[np.inf]], [[1e9]]):
            for method in codec.METHODS:
                with self.subTest(values=values, method=method), self.assertRaises(ValueError):
                    codec.encode(values, method)
        for method in ('unknown', 'lvq4', None):
            with self.assertRaises(ValueError):
                codec.encode([[1]], method)
            with self.assertRaises(ValueError):
                codec.decode(b'\0\0', method, 1, 1)
        for heads, dimensions in ((0, 1), (True, 1), (1, False), (1, 65536), (1.0, 1), (1, 1.0)):
            with self.subTest(heads=heads, dimensions=dimensions), self.assertRaises(ValueError):
                codec.decode(b'\0\0', 'fp16', heads, dimensions)
        for raw in (b'', b'\0', b'\0\0\0', np.array([np.nan], '<f2').tobytes()):
            with self.assertRaises(ValueError):
                codec.decode(raw, 'fp16', 1, 1)

    def test_rejects_wrong_method_header_version_ranges_lengths_and_padding(self):
        for bits in (5, 6, 8):
            valid = codec.encode([[1, 2, 3], [3, 2, 1]], f'lvq{bits}')
            bad_values = [valid[:15], valid[:-1], valid + b'\0']
            for offset, value in ((0, 0), (8, 1), (10, 4), (12, 3)):
                bad = bytearray(valid)
                bad[offset] = value
                bad_values.append(bad)
            for offset, value in ((16, 0x7e00), (22, 0x7c00), (26, 0), (26, 0xbc00)):
                bad = bytearray(valid)
                struct.pack_into('<H', bad, offset, value)
                bad_values.append(bad)
            if 3 * bits % 8:
                bad = bytearray(valid)
                bad[-1] |= 0x80
                bad_values.append(bad)
            for payload in bad_values:
                with self.subTest(bits=bits, payload=payload), self.assertRaises(ValueError):
                    codec.decode(payload, f'lvq{bits}', 2, 3)
            with self.assertRaises(ValueError):
                codec.decode(valid, 'lvq6' if bits != 6 else 'lvq8', 2, 3)


if __name__ == '__main__':
    unittest.main()
