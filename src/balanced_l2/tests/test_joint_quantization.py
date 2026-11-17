"""Small real-codec sweeps verify the joint grid, source order, and cleanup."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import benchmark_joint_quantization as benchmark
from balanced_l2.common import read_matrix, record_dtype, snapshot


class JointQuantizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="joint-quantization-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "vectors"
        self.source.mkdir()
        self.output = self.root / "comparison"
        rng = np.random.default_rng(19)
        records = np.zeros(96, record_dtype(7))
        records["code"] = np.arange(96) + 100
        records["page"] = np.arange(96) % 8 + 1
        records["vec"] = rng.normal(size=(96, 7))
        self.file = self.source / "1.bin"
        records.tofile(self.file)
        self.before = snapshot(self.file)
        self.render = patch.object(benchmark, "_render")
        self.render.start()
        self.addCleanup(self.render.stop)

    def run_small(self, heads=(8, 12)):
        return benchmark.run(self.source, self.output, head_targets=heads, query_count=8,
                             probes=(1, 2, 4, 8), dim=7, sample_size=72, threads=1, gpu=False)

    def test_requested_default_grid_is_64_pairs_and_320_points(self):
        self.assertEqual(benchmark.HEADS, (64000, 80000, 96000, 128000))
        self.assertEqual(benchmark.CODECS, ("fp16", "lvq8", "lvq6", "lvq5"))
        self.assertEqual(benchmark.PROBES, (4, 8, 16, 32, 64))
        self.assertEqual(benchmark.NOISE, .025)
        listed = subprocess.run([sys.executable, str(benchmark.ROOT / "benchmark_joint_quantization.py"),
                                 "--list-cases"], capture_output=True, text=True, check=True)
        self.assertEqual(len(listed.stdout.splitlines()), 64)
        self.assertTrue(all("probes=4,8,16,32,64" in line for line in listed.stdout.splitlines()))

    def test_full_small_grid_scores_all_codec_pairs_and_cleans_each_count_before_next(self):
        original = benchmark._train
        trained = []

        def train(sample, native, target, gpu, iterations):
            self.assertEqual({path.name for path in (self.output / "_staging").iterdir()}, {"sample.f32"})
            source = np.fromfile(self.file, record_dtype(7))["vec"].astype(np.float32)
            source /= np.linalg.norm(source, axis=1)[:, None]
            sampled = np.sort(np.random.default_rng(42).choice(96, 72, replace=False))
            np.testing.assert_array_equal(read_matrix(sample).view(np.uint32), source[sampled].view(np.uint32))
            if target == 12:
                self.assertFalse((self.output / "_cases/h8").exists())
                self.assertTrue((self.output / "metrics_h8.json").exists())
            trained.append(target)
            return original(sample, native, target, gpu, iterations)

        posting_case = benchmark._posting_case

        def evaluate(case, method, records, counts, posting_rows, *args):
            self.assertFalse((case / "_export_spool").exists())
            self.assertEqual(len(list((case / "clusters").glob("cluster_*.bin.gz"))), len(counts))
            offsets = np.r_[0, np.cumsum(counts)]
            for cid in (0, len(counts) - 1):
                saved = gzip.decompress((case / "clusters" / f"cluster_{cid}.bin.gz").read_bytes())
                self.assertEqual(saved, records[posting_rows[offsets[cid]:offsets[cid + 1]]].tobytes())
            return posting_case(case, method, records, counts, posting_rows, *args)

        with patch.object(benchmark, "_train", side_effect=train), \
                patch.object(benchmark, "_posting_case", side_effect=evaluate):
            result = self.run_small()
        self.assertEqual(trained, [8, 12])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["points"]), 2 * 16 * 4)
        self.assertEqual(snapshot(self.file), self.before)
        self.assertFalse((self.output / "_cases").exists())
        self.assertFalse((self.output / "_staging").exists())
        self.assertTrue(result["staging_removed"])
        for head in (8, 12):
            points = [point for point in result["points"] if point["head_count"] == head]
            expected = {(centroid, posting, probe) for centroid in benchmark.CODECS
                        for posting in benchmark.CODECS for probe in (1, 2, 4, 8)}
            self.assertEqual({(p["centroid_method"], p["posting_method"], p["probes"]) for p in points}, expected)
            self.assertTrue(all(point["queries"] == 8 and point["noise_std"] == .025 for point in points))
            self.assertTrue(all(0 <= point["recall_percent"] <= point["tie_aware_recall_percent"] <= 100 for point in points))
            self.assertTrue(all(point["cold_gzip_MB_mean"] > point["posting_gzip_MB_mean"] for point in points))
            self.assertTrue(all(point["recall_vs_fp16_percent"] == 100
                                for point in points if point["posting_method"] == "fp16"))
            for centroid in benchmark.CODECS:
                for posting in benchmark.CODECS:
                    subset = [point for point in points if point["centroid_method"] == centroid and point["posting_method"] == posting]
                    self.assertEqual([point["posting_gzip_MB_mean"] for point in subset],
                                     sorted(point["posting_gzip_MB_mean"] for point in subset))
        with patch.object(benchmark, "_stage", side_effect=AssertionError("Restaged complete run")), \
                patch.object(benchmark, "_train", side_effect=AssertionError("Retrained complete run")):
            self.assertEqual(self.run_small()["points"], result["points"])

    def test_resume_uses_posting_checkpoint_without_retraining(self):
        original = benchmark._checkpoint
        interrupted = False

        def checkpoint(*args):
            nonlocal interrupted
            original(*args)
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt("test durable posting checkpoint")

        with patch.object(benchmark, "_checkpoint", side_effect=checkpoint):
            with self.assertRaises(KeyboardInterrupt):
                self.run_small(heads=(8,))
        self.assertTrue((self.output / "_cases/h8/progress_fp16.json").exists())
        with patch.object(benchmark, "_train", side_effect=AssertionError("Retrained resumed count")):
            result = self.run_small(heads=(8,))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["points"]), 64)
        self.assertFalse((self.output / "_cases").exists())

    def test_checkpoint_marker_failure_preserves_previous_durable_arrays(self):
        original = benchmark.write_json
        failed = False

        def write(path, value):
            nonlocal failed
            if path.name == "progress_fp16.json" and value["done"] == 2 and not failed:
                failed = True
                raise KeyboardInterrupt("test after new arrays but before marker commit")
            return original(path, value)

        with patch.object(benchmark, "write_json", side_effect=write):
            with self.assertRaises(KeyboardInterrupt):
                self.run_small(heads=(8,))
        case = self.output / "_cases/h8"
        self.assertTrue((case / "progress_fp16_1.npz").exists())
        self.assertTrue((case / "progress_fp16_2.npz").exists())
        state = json.loads((case / "progress_fp16.json").read_text())
        self.assertEqual(state["done"], 1)
        benchmark.check_artifact(state["arrays"])
        with patch.object(benchmark, "_train", side_effect=AssertionError("Retrained durable model")):
            self.assertEqual(self.run_small(heads=(8,))["status"], "complete")

    def test_saved_posting_export_reuses_matching_orphans_and_rejects_corruption(self):
        original = benchmark._write_checkpoint
        failed = False

        def checkpoint(case, name, done, arrays, **metadata):
            nonlocal failed
            if name == "postings" and done == 2 and not failed:
                failed = True
                raise KeyboardInterrupt("test saved posting before checkpoint")
            return original(case, name, done, arrays, **metadata)

        with patch.object(benchmark, "_write_checkpoint", side_effect=checkpoint):
            with self.assertRaises(KeyboardInterrupt):
                self.run_small(heads=(8,))
        orphan = self.output / "_cases/h8/clusters/cluster_1.bin.gz"
        before = snapshot(orphan)
        posting_case = benchmark._posting_case

        def evaluate(*args):
            self.assertEqual(snapshot(orphan), before)
            return posting_case(*args)

        with patch.object(benchmark, "_posting_case", side_effect=evaluate):
            self.assertEqual(self.run_small(heads=(8,))["status"], "complete")

        # Corruption with valid gzip bytes must be detected before it enters scoring.
        other = self.root / "corrupt"
        with patch.object(benchmark, "_checkpoint", side_effect=KeyboardInterrupt("stop after export")):
            with self.assertRaises(KeyboardInterrupt):
                benchmark.run(self.source, other, head_targets=(8,), query_count=8,
                              probes=(1, 2, 4, 8), dim=7, sample_size=72, threads=1, gpu=False)
        path = other / "_cases/h8/clusters/cluster_0.bin.gz"
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()) + b"corrupt", mtime=0))
        with self.assertRaisesRegex(ValueError, "Saved clustered posting changed"):
            benchmark.run(self.source, other, head_targets=(8,), query_count=8,
                          probes=(1, 2, 4, 8), dim=7, sample_size=72, threads=1, gpu=False)

    def test_completed_metrics_survive_cleanup_interruption_and_then_skip_deleted_build(self):
        with patch.object(benchmark, "_clean_case", side_effect=RuntimeError("test before cleanup")):
            with self.assertRaisesRegex(RuntimeError, "before cleanup"):
                self.run_small(heads=(8,))
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["cases"]["8"]["status"], "complete")
        self.assertTrue((self.output / "_cases/h8").exists())
        with patch.object(benchmark, "_model", side_effect=AssertionError("Rebuilt completed model")):
            result = self.run_small(heads=(8,))
        self.assertEqual(result["status"], "complete")
        self.assertFalse((self.output / "_cases").exists())

    def test_rejects_changed_sources_metrics_and_foreign_case_cleanup(self):
        self.run_small(heads=(8,))
        metrics = self.output / "metrics_h8.json"
        original = metrics.read_bytes()
        metrics.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.run_small(heads=(8,))
        metrics.write_bytes(original)
        # Restoring bytes alone does not restore the fingerprint's timestamp.
        ledger = json.loads((self.output / "results.json").read_text())
        ledger["cases"]["8"]["metrics"] = benchmark.artifact(metrics)
        (self.output / "results.json").write_text(json.dumps(ledger))
        changed = deepcopy(ledger)
        changed["points"][0]["recall_percent"] = -1
        (self.output / "results.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "Retained metrics differ"):
            self.run_small(heads=(8,))
        (self.output / "results.json").write_text(json.dumps(ledger))
        case_parent = self.output / "_cases"
        case_parent.mkdir()
        (case_parent / "h8").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_small(heads=(8,))
        self.assertTrue(self.file.exists())
        (case_parent / "h8").unlink()
        case_parent.rmdir()
        self.file.write_bytes(self.file.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            self.run_small(heads=(8,))

    def test_metric_separates_centroid_coverage_from_posting_reranking(self):
        records = np.zeros(2, record_dtype(2))
        records["code"], records["page"] = [1, 2], 1
        records["vec"] = [[1, 0], [0, 1]]
        arguments = dict(records=records, labels=np.array([0, 1]), counts=np.array([1, 1]),
                         truth=np.array([0]), queries=np.array([[1., 0.]], np.float32),
                         routes=np.array([[0, 1]]), winners=np.array([[1]]), scores=np.array([[.9]]),
                         sizes=np.array([100, 50]), baseline=np.array([[0]]), head_count=2,
                         centroid_method="lvq5", posting_method="lvq5", centroid_bytes=30,
                         config={"probes": [2], "noise_std": .025},
                         timings=dict(encode_seconds=0., decode_seconds=0., score_seconds=0.))
        point = benchmark._measure(**arguments)[0]
        self.assertEqual(point["routing_coverage_percent"], 100)
        self.assertEqual(point["recall_percent"], 0)
        self.assertEqual(point["recall_vs_fp16_percent"], 0)
        self.assertEqual(point["tie_aware_recall_percent"], 0)
        self.assertAlmostEqual(point["cold_gzip_MB_mean"], 180 / 1e6)


if __name__ == "__main__":
    unittest.main()
