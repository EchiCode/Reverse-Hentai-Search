"""Check real GGUF/FP8 encodings, bounded batches, and malformed payloads."""

import struct
import unittest
from unittest.mock import patch

import numpy as np
import torch

from balanced_l2 import posting_external_codecs as codec


class PostingExternalCodecTests(unittest.TestCase):
    def test_native_fp8_bytes_are_literal_float8_values(self):
        values = np.array([[0, 1, -1, .5, 2]], np.float32)
        for method, expected in (('fp8_e4m3_none', '0038b83040'), ('fp8_e5m2_none', '003cbc3840')):
            with self.subTest(method=method):
                payload = codec.encode(values, method)
                self.assertEqual(payload[16:].hex(), expected)
                np.testing.assert_array_equal(codec.decode(payload), values)

    def test_all_methods_preserve_empty_zero_and_normalized_matrix_shapes(self):
        source = np.random.default_rng(3).normal(size=(7, 384)).astype(np.float32)
        source /= np.linalg.norm(source, axis=1)[:, None]
        source = source.astype(np.float16).astype(np.float32)
        for method in codec.METHODS:
            with self.subTest(method=method):
                self.assertEqual(codec.decode(codec.encode(np.empty((0, 384), np.float32), method)).shape, (0, 384))
                zeros = np.zeros((3, 384), np.float32)
                np.testing.assert_array_equal(codec.decode(codec.encode(zeros, method)), zeros)
                decoded = codec.decode(codec.encode(source, method))
                self.assertEqual(decoded.dtype, np.dtype('float32'))
                self.assertEqual(decoded.shape, source.shape)
                self.assertTrue(np.isfinite(decoded).all())
                self.assertLess(float(np.max(np.abs(decoded - source))), .04)

    def test_gguf_payloads_and_reconstructions_match_the_library(self):
        from gguf.constants import GGMLQuantizationType
        from gguf.quants import dequantize, quantize

        source = np.random.default_rng(4).normal(size=(3, 64)).astype(np.float16).astype(np.float32)
        for method in codec.METHODS[:5]:
            with self.subTest(method=method):
                qtype = GGMLQuantizationType[method[5:].upper()]
                packed = quantize(source, qtype)
                payload = codec.encode(source, method)
                self.assertEqual(payload[16:], packed.tobytes())
                np.testing.assert_array_equal(codec.decode(payload), dequantize(packed, qtype))
        ones = np.ones((1, 32), np.float32)
        self.assertEqual(codec.encode(ones, 'gguf_q4_0')[16:], bytes.fromhex('00b0') + bytes(16))
        np.testing.assert_array_equal(codec.decode(codec.encode(ones, 'gguf_q4_0')), ones)

    def test_fp8_scales_are_per_row_or_block_and_zero_groups_use_one(self):
        for kind, dtype in (('e4m3', torch.float8_e4m3fn), ('e5m2', torch.float8_e5m2)):
            maximum = torch.finfo(dtype).max
            for mode, block_size in (('row', 35), ('b32', 32), ('b64', 64), ('b128', 128)):
                dimensions = 35 if mode == 'row' else block_size + 3
                source = np.zeros((2, dimensions), np.float32)
                source[0] = .125
                source[0, -3:] = [1, -2, 3]
                method = f'fp8_{kind}_{mode}'
                payload = codec.encode(source, method)
                groups = (dimensions + block_size - 1) // block_size
                row_bytes = 4 * groups + dimensions
                with self.subTest(method=method):
                    self.assertEqual(len(payload), 16 + 2 * row_bytes)
                    first = np.frombuffer(payload, '<f4', count=groups, offset=16)
                    expected = np.array([3 / maximum] if mode == 'row' else [.125 / maximum, 3 / maximum], np.float32)
                    np.testing.assert_array_equal(first, expected)
                    second = np.frombuffer(payload, '<f4', count=groups, offset=16 + row_bytes)
                    np.testing.assert_array_equal(second, np.ones(groups, np.float32))
                    np.testing.assert_array_equal(codec.decode(payload)[1], source[1])
                    self.assertGreater(float(np.linalg.norm(codec.decode(payload)[0])), 3)

    def test_torch_thread_limit_is_restored_after_success_and_failure(self):
        previous = torch.get_num_threads()
        finite = codec._finite
        observed_threads = []

        def check(values):
            observed_threads.append(torch.get_num_threads())
            finite(values)

        try:
            torch.set_num_threads(9)
            with patch.object(codec, '_finite', side_effect=check):
                payload = codec.encode(np.ones((2, 3), np.float32), 'fp8_e4m3_row')
                self.assertEqual(torch.get_num_threads(), 9)
                codec.decode(payload)
                self.assertEqual(torch.get_num_threads(), 9)
                with self.assertRaises(ValueError):
                    codec.encode(np.full((1, 3), np.nan, np.float32), 'fp8_e4m3_row')
                self.assertEqual(torch.get_num_threads(), 9)
            self.assertTrue(observed_threads)
            self.assertEqual(set(observed_threads), {8})
        finally:
            torch.set_num_threads(previous)

    def test_fp8_conversion_buffers_do_not_include_the_entire_posting(self):
        source = np.full((4099, 7), .25, np.float32)
        finite = codec._finite
        observed_rows = []

        def check(values):
            observed_rows.append(len(values))
            finite(values)

        with patch.object(codec, '_finite', side_effect=check):
            decoded = codec.decode(codec.encode(source, 'fp8_e4m3_none'))
        self.assertEqual(max(observed_rows), 4096)
        self.assertIn(3, observed_rows)
        np.testing.assert_array_equal(decoded, source)

    def test_rejects_invalid_source_shapes_and_nonfinite_or_overflow_values(self):
        invalid = [np.empty((1, 0)), np.ones(3), np.ones((1, 2, 3)),
                   np.array([['bad']]), np.ones((1, 3), np.complex64),
                   np.full((1, 3), np.nan), np.full((1, 3), np.inf)]
        for source in invalid:
            with self.subTest(shape=source.shape, dtype=source.dtype), self.assertRaises(ValueError):
                codec.encode(source, 'fp8_e4m3_none')
        with self.assertRaises(ValueError):
            codec.encode(np.ones((1, 3)), 'unknown')
        for method in codec.METHODS[:5]:
            with self.subTest(method=method), self.assertRaises(ValueError):
                codec.encode(np.ones((1, 33)), method)
        with self.assertRaises(ValueError):
            codec.encode(np.full((1, 32), 1e6, np.float32), 'gguf_q4_0')
        with self.assertRaises(ValueError):
            codec.encode(np.full((1, 3), 1e6, np.float32), 'fp8_e4m3_none')

    def test_rejects_corrupt_headers_scales_and_nonfinite_code_values(self):
        valid = codec.encode(np.ones((1, 3), np.float32), 'fp8_e4m3_row')
        invalid = [valid[:15], valid[:-1], valid + b'\0']
        for offset, value in ((0, 0), (8, 2), (9, len(codec.METHODS)), (10, 2), (14, 0)):
            broken = bytearray(valid)
            broken[offset] = value
            invalid.append(broken)
        for scale in (0, -1, np.inf, np.nan):
            broken = bytearray(valid)
            struct.pack_into('<f', broken, 16, scale)
            invalid.append(broken)
        for method, code in (('fp8_e4m3_none', 0x7f), ('fp8_e5m2_none', 0x7c)):
            broken = bytearray(codec.encode(np.ones((1, 1), np.float32), method))
            broken[16] = code
            invalid.append(broken)
        for payload in invalid:
            with self.subTest(payload=bytes(payload).hex()), self.assertRaises(ValueError):
                codec.decode(payload)
        broken = bytearray(codec.encode(np.ones((1, 32), np.float32), 'gguf_q4_0'))
        struct.pack_into('<H', broken, 16, 0x7e00)
        with self.assertRaises(ValueError):
            codec.decode(broken)


if __name__ == '__main__':
    unittest.main()
