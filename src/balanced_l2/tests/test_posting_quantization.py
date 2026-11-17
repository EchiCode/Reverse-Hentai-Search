"""Wire-format, corruption, and bounded-buffer checks for quantized postings."""

import struct
import unittest
from unittest.mock import patch

import numpy as np

from balanced_l2.common import record_dtype
from balanced_l2 import posting_quantization as codec


def records(vectors):
    vectors = np.asarray(vectors, dtype=np.float16)
    result = np.empty(len(vectors), dtype=record_dtype(vectors.shape[1]))
    result['code'] = np.arange(len(vectors)) * 3 - 100
    result['page'] = np.arange(len(vectors)) % 2000 + 1
    result['vec'] = vectors
    return result


class PostingQuantizationTests(unittest.TestCase):
    def test_literal_four_bit_wire_values_preserve_metadata_and_padding(self):
        source = records([[1, 2, 3], [3, 2, 1]])
        source['code'], source['page'] = [42, -5], [1, 2]
        payload = codec.encode_posting(source.tobytes(), 3, 4)
        # Mean=(2,2,2), lower=-1, FP16 delta=0.13330078125; codes=(0,8,15)/(15,8,0).
        self.assertEqual(payload.hex(),
                         '4c5651504f5354000300040102000000004000400040'
                         '2a000000010000bc4430800ffbffffff020000bc44308f00')
        codes, pages, vectors = codec.decode_posting(payload, 3)
        np.testing.assert_array_equal(codes, [42, -5])
        np.testing.assert_array_equal(pages, [1, 2])
        np.testing.assert_array_equal(vectors, np.array([
            [1, 2.06640625, 2.99951171875], [2.99951171875, 2.06640625, 1],
        ], np.float32))

    def test_empty_and_constant_postings_work_at_every_bit_depth(self):
        for dimensions in (1, 7, 384):
            empty = np.empty(0, dtype=record_dtype(dimensions)).tobytes()
            fp_codes, fp_pages, fp_vectors = codec.decode_fp16(empty, dimensions)
            self.assertEqual((len(fp_codes), len(fp_pages), fp_vectors.shape), (0, 0, (0, dimensions)))
            source = records(np.full((4, dimensions), .5))
            for bits in range(2, 9):
                with self.subTest(dimensions=dimensions, bits=bits):
                    payload = codec.encode_posting(empty, dimensions, bits)
                    self.assertEqual(len(payload), 16 + 2 * dimensions)
                    decoded = codec.decode_posting(payload, dimensions)
                    self.assertEqual(decoded[2].shape, (0, dimensions))
                    decoded = codec.decode_posting(codec.encode_posting(source.tobytes(), dimensions, bits), dimensions)
                    np.testing.assert_array_equal(decoded[0], source['code'])
                    np.testing.assert_array_equal(decoded[1], source['page'])
                    np.testing.assert_array_equal(decoded[2], source['vec'].astype(np.float32))

    def test_quantization_error_decreases_with_bit_depth_and_fp16_is_exact(self):
        values = np.random.default_rng(31).normal(size=(31, 384)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1)[:, None]
        source = records(values)
        fp_codes, fp_pages, original = codec.decode_fp16(source.tobytes())
        np.testing.assert_array_equal(fp_codes, source['code'])
        np.testing.assert_array_equal(fp_pages, source['page'])
        np.testing.assert_array_equal(original, source['vec'].astype(np.float32))
        errors = []
        for bits in (4, 6, 8):
            codes, pages, decoded = codec.decode_posting(codec.encode_posting(source.tobytes(), bits=bits))
            np.testing.assert_array_equal(codes, source['code'])
            np.testing.assert_array_equal(pages, source['page'])
            errors.append(float(np.mean((decoded.astype(np.float64) - original) ** 2)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[0], 1e-4)

    def test_three_five_and_seven_bit_rows_match_literal_wire_arithmetic(self):
        dimensions = 7
        source = records(np.random.default_rng(19).normal(size=(3, dimensions)))
        for bits in (3, 5, 7):
            with self.subTest(bits=bits):
                payload = codec.encode_posting(source.tobytes(), dimensions, bits)
                codes, pages, actual = codec.decode_posting(payload, dimensions)
                mean = np.frombuffer(payload, '<f2', dimensions, offset=16).astype(np.float64)
                row_bytes = 10 + (dimensions * bits + 7) // 8
                expected = np.empty_like(actual)
                for row in range(len(source)):
                    offset = 16 + 2 * dimensions + row * row_bytes
                    code, page, lower, delta = struct.unpack_from('<ih ee', payload, offset)
                    self.assertEqual((code, page), (codes[row], pages[row]))
                    packed = int.from_bytes(payload[offset + 10:offset + row_bytes], 'little')
                    self.assertEqual(packed >> (dimensions * bits), 0)
                    for dimension in range(dimensions):
                        value = (packed >> (dimension * bits)) & ((1 << bits) - 1)
                        expected[row, dimension] = lower + value * delta + mean[dimension]
                np.testing.assert_array_equal(actual, expected)

    def test_pack_and_unpack_buffers_are_bounded_and_cross_batch_rows_survive(self):
        source = records(np.random.default_rng(2).normal(size=(4099, 7)))
        packed_rows, unpacked_rows = [], []
        packbits, unpackbits = np.packbits, np.unpackbits

        def pack(values, **kwargs):
            packed_rows.append(len(values))
            return packbits(values, **kwargs)

        def unpack(values, **kwargs):
            unpacked_rows.append(len(values))
            return unpackbits(values, **kwargs)

        with patch.object(codec.np, 'packbits', side_effect=pack), patch.object(codec.np, 'unpackbits', side_effect=unpack):
            decoded = codec.decode_posting(codec.encode_posting(source.tobytes(), 7, 6), 7)
        self.assertEqual(packed_rows, [4096, 3])
        self.assertEqual(unpacked_rows, [4096, 3])
        np.testing.assert_array_equal(decoded[0], source['code'])
        np.testing.assert_array_equal(decoded[1], source['page'])
        self.assertTrue(np.isfinite(decoded[2]).all())

    def test_rejects_invalid_source_dimensions_bits_and_ranges(self):
        source = records([[1, 2, 3], [3, 2, 1]])
        raw = source.tobytes()
        for dimensions in (0, -1, 65536, True, 3.0):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                codec.encode_posting(raw, dimensions)
        for bits in (0, 1, 9, 16, True, 4.0):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                codec.encode_posting(raw, 3, bits)
        for vectors in ([[0, 0, 0]], [[np.nan, 1, 2]], [[np.inf, 1, 2]]):
            invalid = records(vectors).tobytes()
            for decode in (True, False):
                with self.subTest(vectors=vectors, decode=decode), self.assertRaises(ValueError):
                    (codec.decode_fp16 if decode else codec.encode_posting)(invalid, 3)
        for function in (codec.decode_fp16, codec.encode_posting):
            with self.assertRaises(ValueError):
                function(raw[:-1], 3)
        extreme = records([[-65504, 65504], [65504, 65504], [65504, 65504]])
        with self.assertRaises(ValueError):
            codec.encode_posting(extreme.tobytes(), 2)

    def test_rejects_corrupt_quantized_payloads(self):
        valid = codec.encode_posting(records([[1, 2, 3], [3, 2, 1]]).tobytes(), 3, 4)
        invalid = [valid[:15], valid[:-1], valid + b'\0']
        for offset, value in ((0, 0), (8, 4), (10, 1), (11, 2), (12, 3)):
            broken = bytearray(valid)
            broken[offset] = value
            invalid.append(broken)
        for offset, half in ((16, 0x7e00), (22 + 6, 0x7c00), (22 + 8, 0), (22 + 8, 0xbc00)):
            broken = bytearray(valid)
            struct.pack_into('<H', broken, offset, half)
            invalid.append(broken)
        broken = bytearray(valid)
        broken[22 + 11] |= 0x80
        invalid.append(broken)
        for payload in invalid:
            with self.subTest(payload=bytes(payload).hex()), self.assertRaises(ValueError):
                codec.decode_posting(payload, 3)


if __name__ == '__main__':
    unittest.main()
