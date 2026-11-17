import base64
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from balanced_l2 import posting_scoring as scoring
from balanced_l2.common import record_dtype


def literal_winners(queries, vectors):
    scores = np.empty((len(queries), len(vectors)), dtype=np.float64)
    for row, query in enumerate(queries):
        for column, vector in enumerate(vectors):
            squared_norm = 0.
            for value in vector:
                squared_norm += float(value) * float(value)
            dot, dimension = 0., 0
            while dimension + 3 < len(query):
                group = float(query[dimension]) * float(vector[dimension])
                for offset in (1, 2, 3):
                    group += float(query[dimension + offset]) * float(vector[dimension + offset])
                dot += group
                dimension += 4
            for dimension in range(dimension, len(query)):
                dot += float(query[dimension]) * float(vector[dimension])
            scores[row, column] = dot * float(np.float32(1 / math.sqrt(squared_norm)))
    ids = np.argmax(scores, axis=1)
    return ids.astype(np.int64), scores[np.arange(len(queries)), ids]


class PostingScoringTests(unittest.TestCase):
    def check_backend(self, queries, vectors, gpu):
        expected_ids, expected_scores = literal_winners(queries, vectors)
        ids, scores = scoring.best_matches(queries, vectors, gpu=gpu, query_batch=3, vector_batch=4)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(scores.view(np.uint64), expected_scores.view(np.uint64))

    def test_cpu_matches_client_dot_groups_tails_and_batch_ties(self):
        rng = np.random.default_rng(42)
        for dimensions in (1, 3, 4, 5, 37, 384):
            with self.subTest(dimensions=dimensions):
                vectors = rng.normal(size=(17, dimensions)).astype(np.float32)
                vectors[-1] = vectors[0]
                queries = np.r_[vectors[:1], rng.normal(size=(6, dimensions)).astype(np.float32)]
                self.check_backend(queries, vectors, False)
        queries = np.ones((1, 8), np.float32)
        vectors = np.array([[1.e20, 0, 0, 0, -1.e20, 1, 1, 1]], np.float32)
        _, scores = scoring.best_matches(queries, vectors, gpu=False)
        self.assertEqual(scores[0], 0.)

    def test_query_scale_is_preserved_and_only_vector_norms_are_removed(self):
        vectors = np.array([[2, 0], [1, 1], [4, 0]], np.float32)
        queries = np.array([[1, 0], [.25, 0]], np.float32)
        ids, scores = scoring.best_matches(queries, vectors, gpu=False, vector_batch=1)
        np.testing.assert_array_equal(ids, [0, 0])
        np.testing.assert_array_equal(scores, [1., .25])

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the actual client parity check')
    def test_cpu_matches_actual_client_decoder_and_scoring(self):
        rng = np.random.default_rng(17)
        records = np.zeros(17, record_dtype(7))
        records['vec'] = rng.normal(size=(17, 7)).astype(np.float16)
        records['vec'][-1] = records['vec'][0]
        vectors = records['vec'].astype(np.float32)
        queries = np.r_[vectors[:1], rng.normal(size=(4, 7)).astype(np.float32)]
        root = Path(__file__).resolve().parents[2]
        client = root / 'client'
        if not client.is_dir():
            client = root.parent / 'client'
        source = client / 'search.js'
        script = r'''
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({ performance });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8')
    + '\nglobalThis.api = {decodeSearchCluster, scoreSearchCluster};', context);
const bytes = Buffer.from(input.raw, 'base64');
const cluster = context.api.decodeSearchCluster(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), 7);
const winners = input.queries.map(query => context.api.scoreSearchCluster(new Float32Array(query), cluster, -Infinity)[0]);
console.log(JSON.stringify({ids: winners.map(item => item.index),
    scoreHex: Buffer.from(Float64Array.from(winners, item => item.similarity).buffer).toString('hex')}));
'''
        result = subprocess.run([shutil.which('node'), '-e', script, str(source)], check=True,
                                input=json.dumps({'raw': base64.b64encode(records.tobytes()).decode(),
                                                  'queries': queries.tolist()}),
                                capture_output=True, text=True)
        expected = json.loads(result.stdout)
        backends = (False, True) if os.environ.get('BALANCED_L2_TEST_GPU') == '1' else (False,)
        for gpu in backends:
            with self.subTest(gpu=gpu):
                ids, scores = scoring.best_matches(queries, vectors, gpu=gpu, vector_batch=4)
                np.testing.assert_array_equal(ids, expected['ids'])
                self.assertEqual(scores.astype('<f8').tobytes().hex(), expected['scoreHex'])

    def test_score_tiles_obey_budget(self):
        rng = np.random.default_rng(3)
        queries, vectors = rng.normal(size=(5, 7)).astype(np.float32), rng.normal(size=(13, 7)).astype(np.float32)
        original, shapes = scoring._cpu_winners, []

        def record(query, vector, inverse):
            shapes.append((len(query), len(vector)))
            return original(query, vector, inverse)

        with patch.object(scoring, 'SCORE_BYTES', 64 * 6), patch.object(scoring, '_cpu_winners', side_effect=record):
            actual = scoring.best_matches(queries, vectors, gpu=False, query_batch=100, vector_batch=100)
        self.assertGreater(len(shapes), 1)
        self.assertTrue(all(rows * columns <= 6 for rows, columns in shapes))
        expected = literal_winners(queries, vectors)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1].view(np.uint64), expected[1].view(np.uint64))

    def test_empty_and_invalid_inputs(self):
        rows, scores = scoring.best_matches(np.empty((0, 3)), np.eye(3), gpu=False)
        self.assertEqual(rows.shape, (0,))
        self.assertEqual(scores.dtype, np.float64)
        rows, scores = scoring.best_matches(np.ones((2, 3)), np.empty((0, 3)), gpu=False)
        np.testing.assert_array_equal(rows, [-1, -1])
        self.assertTrue(np.isneginf(scores).all())
        for queries, vectors in ((np.ones(3), np.eye(3)), (np.eye(3), np.ones((2, 4))),
                                 (np.full((1, 3), np.nan), np.eye(3)),
                                 (np.eye(3), np.zeros((1, 3))), (np.eye(3), np.full((1, 3), np.inf))):
            with self.subTest(shape=np.shape(queries)), self.assertRaises(ValueError):
                scoring.best_matches(queries, vectors, gpu=False)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                scoring.best_matches(np.eye(3), np.eye(3), gpu=False, query_batch=size)

    @unittest.skipUnless(os.environ.get('BALANCED_L2_TEST_GPU') == '1', 'GPU opt-in')
    def test_gpu_matches_literal_bits_for_near_ties_subnormals_and_overflow(self):
        rng = np.random.default_rng(91)
        values = np.array([0., 1.e-44, -1.e-44, 1.e-30, -1.e-30, .5, -1., 2.], np.float32)
        self.check_backend(rng.choice(values, (7, 37)), rng.choice(values, (19, 37)), True)
        self.check_backend(np.ones((1, 4), np.float32),
                           np.array([[1.e8, 1, -1.e8, 0], [1.e8, 2, -1.e8, 0]], np.float32), True)
        self.check_backend(np.array([[3.e38, 3.e38]], np.float32),
                           np.array([[3.e38, 0], [0, 3.e38], [1.e-38, 1.e-38]], np.float32), True)
        self.check_backend(np.ones((1, 8), np.float32),
                           np.array([[1.e20, 0, 0, 0, -1.e20, 1, 1, 1]], np.float32), True)

    @unittest.skipUnless(os.environ.get('BALANCED_L2_TEST_GPU') == '1', 'GPU opt-in')
    def test_gpu_restores_cublas_modes_after_failure(self):
        import cupy as cp
        from cupy_backends.cuda.libs import cublas
        handle = cp.cuda.device.get_cublas_handle()
        before = cublas.getMathMode(handle), cublas.getPointerMode(handle)
        with patch.object(scoring, '_gpu_winners', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                scoring.best_matches(np.eye(3), np.eye(3))
        self.assertEqual((cublas.getMathMode(handle), cublas.getPointerMode(handle)), before)


if __name__ == '__main__':
    unittest.main()
