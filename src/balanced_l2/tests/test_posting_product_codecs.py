import os
import tempfile
import unittest

import faiss
import numpy as np

from balanced_l2.posting_product_codecs import (FaissCodec, METHODS, PRODUCT_METHODS,
                                               SCALAR_METHODS, load, train)


class PostingProductCodecTests(unittest.TestCase):
    def setUp(self):
        self.sample = np.random.default_rng(42).normal(size=(256, 32)).astype(np.float32)

    def train_small(self, name):
        # Small roundtrip fixtures intentionally fall below Faiss's quality recommendation.
        with tempfile.TemporaryFile() as ignored:
            previous = os.dup(2)
            try:
                os.dup2(ignored.fileno(), 2)
                return train(name, self.sample, threads=2)
            finally:
                os.dup2(previous, 2)
                os.close(previous)

    def assert_roundtrip(self, name):
        codec = self.train_small(name)
        encoded = codec.encode(self.sample[:7])
        reconstructed = codec.decode(encoded)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(encoded.shape, (7, codec.code_size))
        self.assertEqual(reconstructed.shape, (7, 32))
        self.assertEqual(reconstructed.dtype, np.float32)
        self.assertTrue(np.isfinite(reconstructed).all())
        self.assertLess(float(np.square(reconstructed - self.sample[:7]).mean()), 2.)
        restored = load(codec.serialize())
        self.assertEqual(restored.name, name)
        self.assertEqual(restored.dimensions, 32)
        self.assertEqual(restored.code_size, codec.code_size)
        np.testing.assert_array_equal(restored.encode(self.sample[:7]), encoded)
        np.testing.assert_array_equal(restored.decode(encoded), reconstructed)
        self.assertEqual(restored.encode(np.empty((0, 32))).shape, (0, codec.code_size))
        self.assertEqual(restored.decode(np.empty((0, codec.code_size), np.uint8)).shape, (0, 32))
        return codec

    def test_complete_faiss_method_grid(self):
        self.assertEqual(len(SCALAR_METHODS), 26)
        self.assertEqual(sum(name.startswith('pq') for name in PRODUCT_METHODS), 57)
        self.assertEqual(len(PRODUCT_METHODS), 64)
        self.assertEqual(len(set(METHODS)), 91)
        for name in ('pq16x2', 'pq128x7', 'pq8x4', 'pq384x8', 'opq96x8', 'rq8x6', 'lsq8x6'):
            self.assertIn(name, METHODS)

    def test_every_scalar_range_and_bf16_are_serialized(self):
        for name in (*SCALAR_METHODS, 'bf16'):
            with self.subTest(name=name):
                codec = self.assert_roundtrip(name)
                if '_meanstd' in name:
                    self.assertEqual(codec.index.sq.rangestat, faiss.ScalarQuantizer.RS_meanstd)
                elif '_quantile' in name:
                    self.assertEqual(codec.index.sq.rangestat, faiss.ScalarQuantizer.RS_quantiles)
                elif '_optim' in name:
                    self.assertEqual(codec.index.sq.rangestat, faiss.ScalarQuantizer.RS_optim)

    def test_native_product_rotation_residual_and_local_search_roundtrip(self):
        for name, width in (('pq16x2', 4), ('opq32x8', 32), ('rq4x6', 3), ('lsq4x6', 3)):
            with self.subTest(name=name):
                codec = self.assert_roundtrip(name)
                self.assertEqual(codec.code_size, width)

    def test_invalid_inputs_and_corrupted_models_fail(self):
        for values in (np.ones(32), np.zeros((0, 32)), np.ones((3, 0)), np.full((4, 32), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                train('sq4', values)
        with self.assertRaises(ValueError):
            train('pq16x8', self.sample[:32])
        with self.assertRaises(ValueError):
            train('pq96x4', self.sample)
        with self.assertRaises(ValueError):
            train('unknown', self.sample)
        codec = self.train_small('sq4')
        with self.assertRaises(ValueError):
            codec.encode(np.ones((2, 31)))
        with self.assertRaises(ValueError):
            codec.decode(np.zeros((2, codec.code_size), np.int32))
        with self.assertRaises(ValueError):
            codec.decode(np.zeros((2, codec.code_size + 1), np.uint8))
        payload = codec.serialize()
        corrupted = payload[:-1] + bytes([payload[-1] ^ 1])
        for invalid in (b'', payload[:11], payload[:-1], corrupted):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                load(invalid)
        original = faiss.vector_to_array(codec.index.sq.trained)
        original[0] = np.nan
        faiss.copy_array_to_vector(original, codec.index.sq.trained)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            codec.serialize()


if __name__ == '__main__':
    unittest.main()
