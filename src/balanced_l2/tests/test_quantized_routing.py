"""Small exact-routing checks; GPU checks require BALANCED_L2_TEST_GPU=1."""

import os
import unittest

import numpy as np

from balanced_l2.quantized_routing import route_queries


def literal_routes(heads, queries, probes, metric="l2"):
    result = []
    for query in queries:
        scores = []
        for head in heads:
            score = 0.
            for a, b in zip(query, head):
                score += float(a) * float(b)
            if metric == "l2":
                norm = 0.
                for b in head:
                    norm += float(b) * float(b)
                score += -.5 * norm
            scores.append(score)
        result.append(sorted(range(len(heads)), key=lambda i: (-scores[i], i))[:probes])
    return np.asarray(result, dtype=np.int32)


class QuantizedRoutingTests(unittest.TestCase):
    def cases(self):
        rng = np.random.default_rng(17)
        heads = rng.normal(size=(73, 32)).astype(np.float32)
        heads[11] = heads[3]
        heads[5] = np.nextafter(heads[3], np.inf)
        queries = rng.normal(size=(19, 32)).astype(np.float32)
        queries[0] = 0
        yield heads, queries, 10
        yield np.ones((17, 13), np.float32), rng.normal(size=(5, 13)).astype(np.float32), 10
        yield np.zeros((13, 7), np.float32), np.ones((9, 7), np.float32), 10
        prototypes = rng.normal(size=(4, 32)).astype(np.float32)
        repeated = prototypes[np.arange(39) % 4]
        yield repeated, queries, 10
        tiny = np.array([0., 1.e-44, -1.e-44, 1.e-30, -1.e-30, 1., -1.], dtype=np.float32)
        yield rng.choice(tiny, (19, 32)), rng.choice(tiny, (7, 32)), 10
        # Float32 norms collapse these to a tie; literal float64 L2 must pick ID 1.
        yield np.array([[1., 2. ** -13], [1., 0.]], np.float32), np.array([[1., 0.]], np.float32), 1
        yield heads[:5], queries, 5

    def test_cpu_matches_literal_scores_and_ties(self):
        for heads, queries, probes in self.cases():
            for block in (1, 7, 256):
                np.testing.assert_array_equal(route_queries(heads, queries, probes, block, gpu=False),
                                              literal_routes(heads, queries, probes))

    def test_validation_and_empty_queries(self):
        heads = np.eye(3, dtype=np.float32)
        self.assertEqual(route_queries(heads, np.empty((0, 3)), gpu=False).shape, (0, 3))
        for kwargs in ({"probes": 0}, {"probes": True}, {"query_block": 0}, {"metric": "cosine"}):
            with self.assertRaises(ValueError):
                route_queries(heads, heads, gpu=False, **kwargs)
        for bad in (np.full((3, 3), np.nan), np.ones((3, 4)), np.array([1])):
            with self.assertRaises(ValueError):
                route_queries(heads, bad, gpu=False)

    def test_l2_uses_float64_norm_bias_and_preserves_ties(self):
        for heads, queries, probes in self.cases():
            np.testing.assert_array_equal(route_queries(heads, queries, probes, 7, gpu=False, metric="l2"),
                                          literal_routes(heads, queries, probes, "l2"))
        heads = np.array([[10., 0.], [1., 0.], [0., 1.]], np.float32)
        query = np.array([[.9, 0.]], np.float32)
        self.assertEqual(route_queries(heads, query, 1, gpu=False, metric="ip")[0, 0], 0)
        self.assertEqual(route_queries(heads, query, 1, gpu=False)[0, 0], 1)

    @unittest.skipUnless(os.environ.get("BALANCED_L2_TEST_GPU") == "1", "GPU opt-in")
    def test_gpu_matches_literal_scores_and_ties(self):
        for heads, queries, probes in self.cases():
            for metric in ("ip", "l2"):
                np.testing.assert_array_equal(route_queries(heads, queries, probes, 7, metric=metric),
                                              literal_routes(heads, queries, probes, metric))

    @unittest.skipUnless(os.environ.get("BALANCED_L2_TEST_GPU") == "1", "GPU opt-in")
    def test_gpu_supports_128k_head_ids(self):
        rng = np.random.default_rng(128)
        heads = rng.normal(size=(128000, 5)).astype(np.float32)
        queries = rng.normal(size=(3, 5)).astype(np.float32)
        scores = np.zeros((len(queries), len(heads)), np.float64)
        for dimension in range(heads.shape[1]):
            scores += queries[:, dimension, None].astype(np.float64) * heads[None, :, dimension].astype(np.float64)
        norms = np.zeros(len(heads), np.float64)
        for dimension in range(heads.shape[1]):
            norms += heads[:, dimension].astype(np.float64) ** 2
        scores += -.5 * norms
        expected = np.argsort(-scores, axis=1, kind="stable")[:, :12]
        actual = route_queries(heads, queries, probes=12, query_block=4096)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue((actual >= 65536).any())

    @unittest.skipUnless(os.environ.get("BALANCED_L2_TEST_GPU") == "1", "GPU opt-in")
    def test_gpu_score_matrix_obeys_budget_across_multiple_batches(self):
        import cupy as cp
        from unittest.mock import patch
        rng = np.random.default_rng(51)
        heads = rng.normal(size=(73, 5)).astype(np.float32)
        queries = rng.normal(size=(9, 5)).astype(np.float32)
        allocations = []
        original_empty = cp.empty

        def allocate(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2:
                allocations.append(shape)
            return original_empty(shape, *args, **kwargs)

        with patch("balanced_l2.quantized_routing.GPU_SCORE_BYTES", 73 * 4 * 2), \
                patch.object(cp, "empty", side_effect=allocate):
            actual = route_queries(heads, queries, probes=5, query_block=4096)
        self.assertGreater(len(allocations), 1)
        self.assertTrue(all(rows <= 2 and columns == 73 for rows, columns in allocations))
        np.testing.assert_array_equal(actual, literal_routes(heads, queries, 5))

    @unittest.skipUnless(os.environ.get("BALANCED_L2_TEST_GPU") == "1", "GPU opt-in")
    def test_gpu_literal_score_bits_match_cpu(self):
        import cupy as cp
        from balanced_l2.quantized_routing import _LITERAL_SOURCE
        rng = np.random.default_rng(91)
        values = np.array([0., 1.e-44, -1.e-44, 1.e-30, -1.e-30, .5, -1., 2.], dtype=np.float32)
        heads = rng.choice(values, (19, 37))
        queries = rng.choice(values, (7, 37))
        rows, ids = np.indices((len(queries), len(heads)))
        rows, ids = rows.ravel().astype(np.int64), ids.ravel().astype(np.int64)
        expected = np.zeros(len(ids), np.float64)
        for pair, (row, head) in enumerate(zip(rows, ids)):
            score = 0.
            for a, b in zip(queries[row], heads[head]):
                score += float(a) * float(b)
            expected[pair] = score
        kernel = cp.RawKernel(_LITERAL_SOURCE, "literal_scores", options=("--fmad=false", "--ftz=false"))
        output = cp.empty(len(ids), cp.float64)
        kernel(((len(ids) + 127) // 128,), (128,),
               (cp.asarray(queries), cp.asarray(heads), cp.asarray(rows), cp.asarray(ids),
                np.int32(heads.shape[1]), np.uint64(len(ids)), output))
        np.testing.assert_array_equal(cp.asnumpy(output).view(np.uint64), expected.view(np.uint64))


if __name__ == "__main__":
    unittest.main()
