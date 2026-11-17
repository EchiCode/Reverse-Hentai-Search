"""Small real CPU/export/replay checks for the complete sequential benchmark."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import benchmark as b
from balanced_l2.benchmark_ground_truth import benchmark_noises, input_digest, query_sets
from balanced_l2.build import adopt_build, assign
from balanced_l2.common import artifact, prepare_vectors, read_matrix, record_dtype, sha256, snapshot, write_json
from balanced_l2.export import export_postings
from balanced_l2.quantized_routing import route_queries


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="balanced-l2-benchmark-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "vectors"
        self.data.mkdir()
        records = np.zeros(40, record_dtype(4))
        records["code"] = np.arange(40)
        records["vec"] = np.random.default_rng(3).normal(size=(40, 4)).astype(np.float16)
        self.raw = self.data / "1.bin"
        records.tofile(self.raw)
        self.inputs = [snapshot(self.raw)]
        self.baseline = self.root / "control"
        self.cache_root = self.root / "bench"
        self.cache_root.mkdir()
        write_json(self.cache_root / "client_assets.json", {"fixed_bytes": {"gzip": 50000}})
        source, sample = self.root / "vectors.f32", self.root / "sample.f32"
        selected = np.sort(np.random.default_rng(42).choice(40, 32, replace=False))
        prepare_vectors([self.raw], source, 4, sample_rows=selected, sample_path=sample)
        matrix = read_matrix(source)
        heads = self.root / "heads.f32"
        heads.write_bytes(struct.pack("<ii", 5, 4) + matrix[:5].tobytes())
        labels = self.root / "labels.i32"
        assign(source, heads, labels, gpu=False)
        adopt_build(self.baseline, [self.raw], heads, labels,
                    {"sample_sha256": sha256(sample)}, {"metric": "l2"}, dim=4, sample_size=32)
        export_postings([self.raw], self.baseline / "native/assignments.i32", 5, self.baseline,
                        dim=4, threads=1, batch_size=16)
        truth_dir = self.cache_root / "truth/q8"
        truth_dir.mkdir(parents=True)
        sampled, queries = query_sets(matrix, 8, (.01, .025))
        for noise, values in queries.items():
            truth = np.argmax(values @ matrix.T, axis=1).astype(np.int64)[:, None]
            settings = {"count": 8, "seed": 42, "noise_std": noise,
                        "ground_truth": "exact nearest neighbor over normalized source vectors"}
            np.savez(truth_dir / f"query_truth_noise_{noise:g}.npz", queries=values, truth=truth,
                     sampled_rows=sampled, settings=json.dumps(settings), normalized_corpus_sha256=sha256(source),
                     source_inputs_sha256=input_digest(self.inputs),
                     query_sha256=hashlib.sha256(values.tobytes()).hexdigest())
        self.source = source
        stack = ExitStack()
        self.addCleanup(stack.close)
        for key, value in {"CONTROL": self.baseline, "CONTROL_TARGET": 4,
                           "BENCHMARK_ROOT": self.cache_root, "PROBES": (1, 2, 4)}.items():
            stack.enter_context(patch.object(b, key, value))
        stack.enter_context(patch.object(b, "train_worker", side_effect=self.train))
        stack.enter_context(patch.object(b, "assign", side_effect=lambda *a, **k: assign(*a, **{**k, "gpu": False})))
        stack.enter_context(patch.object(b, "route_queries", side_effect=lambda *a, **k: route_queries(*a, **{**k, "gpu": False})))

    @staticmethod
    def train(sample, native, target):
        values = read_matrix(sample)[:target]
        heads = native / "heads.f32"
        heads.write_bytes(struct.pack("<ii", *values.shape) + values.tobytes())
        write_json(native / "training.json", {"metric": "l2", "sample": artifact(sample), "heads": artifact(heads),
                   "params": {"n_clusters": target, "hierarchical_n_iters": 20}})

    def run_benchmark(self, name="result"):
        return b.run(self.data, self.root / name, head_targets=(4, 6), noises=(.01, .025),
                     query_count=8, dim=4, sample_size=32, threads=1)

    def test_complete_sweep_reuses_control_and_cleans_only_own_outputs(self):
        result = self.run_benchmark()
        output = self.root / "result"
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cases"]["4"]["actual_heads"], 5)
        self.assertTrue(result["cases"]["4"]["control_reused"])
        self.assertEqual(result["cases"]["6"]["actual_heads"], 6)
        self.assertFalse(result["cases"]["6"]["control_reused"])
        points = json.loads((output / "points.json").read_text())["points"]
        self.assertEqual(len(points), 12)
        self.assertEqual(len((output / "points.csv").read_text().splitlines()), 13)
        self.assertFalse((output / "_staging").exists())
        self.assertFalse((output / "h6/clusters").exists())
        self.assertTrue((self.baseline / "clusters/cluster_0.bin.gz").exists())
        self.assertTrue((output / "h6/native/assignments.i32").exists())
        self.assertTrue((output / "plots/noise_0.025.png").exists())
        self.assertEqual(self.run_benchmark()["status"], "complete")

    def test_resume_preserves_the_first_complete_sweep(self):
        with patch.object(b, "train_worker", side_effect=RuntimeError("interrupt")):
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                self.run_benchmark("partial")
        output = self.root / "partial"
        saved = json.loads((output / "comparison.json").read_text())
        self.assertEqual(saved["cases"]["4"]["status"], "complete")
        self.assertEqual(len(json.loads((output / "points.json").read_text())["points"]), 6)
        native = output / "h6/native"
        (native / "assignments.i32").write_bytes(b"orphan")
        (native / "assignments.i32.tmp").write_bytes(b"partial")
        self.assertEqual(self.run_benchmark("partial")["status"], "complete")

    def test_complete_reuse_rejects_changed_truth(self):
        result = self.run_benchmark()
        path = Path(result["truth"]["0.01"]["path"])
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.run_benchmark()

    def test_truth_reuse_rejects_changed_query_bits(self):
        output = self.root / "truth-check"
        caches = benchmark_noises(self.source, output, 8, (.025,), inputs=self.inputs,
                                 reuse_dir=self.cache_root / "truth/q8")
        path = caches[.025]
        with np.load(path, allow_pickle=False) as old:
            values = {key: old[key] for key in old.files}
        values["queries"][0, 0] += .001
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "Saved truth differs"):
            benchmark_noises(self.source, output, 8, (.025,), inputs=self.inputs)

    def test_changed_original_inputs_are_rejected(self):
        self.run_benchmark()
        self.raw.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.run_benchmark()

    def test_resume_refuses_case_symlink_before_cleanup(self):
        with patch.object(b, "train_worker", side_effect=RuntimeError("interrupt")):
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                self.run_benchmark("linked")
        case = self.root / "linked/h6"
        (case / "native").rmdir()
        case.rmdir()
        case.symlink_to(self.baseline, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "case must not be a symlink"):
            self.run_benchmark("linked")
        self.assertTrue((self.baseline / "clusters/cluster_0.bin.gz").exists())


if __name__ == "__main__":
    unittest.main()
