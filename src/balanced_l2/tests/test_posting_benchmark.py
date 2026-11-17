import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import benchmark_cluster_quantization as benchmark
from balanced_l2.common import record_dtype, snapshot
from balanced_l2.export import export_centroids


class PostingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "data"
        self.source.mkdir()
        (self.source / "clusters").mkdir()
        self.output = self.root / "comparison"
        rng = np.random.default_rng(7)
        self.heads = rng.normal(size=(32, 32)).astype(np.float32)
        self.heads /= np.linalg.norm(self.heads, axis=1)[:, None]
        native = self.source / "heads.f32"
        with native.open("wb") as stream:
            stream.write(struct.pack("<ii", 32, 32))
            self.heads.tofile(stream)
        export_centroids(self.source, native)
        for cid, head in enumerate(self.heads):
            records = np.zeros(32, record_dtype(32))
            records["code"] = np.arange(cid * 32 + 1, cid * 32 + 33)
            records["page"] = 1
            values = head + rng.normal(0, .005, size=(32, 32))
            values /= np.linalg.norm(values, axis=1)[:, None]
            records["vec"] = values
            (self.source / "clusters" / f"cluster_{cid}.bin.gz").write_bytes(
                gzip.compress(records.tobytes(), compresslevel=9, mtime=0))

    def run_small(self, methods=("lvq6",), **kwargs):
        return benchmark.run(self.source, self.output, query_count=16, methods=methods,
                             train_size=1024, gpu=False, **kwargs)

    def test_grid_contains_the_original_114_configurations(self):
        from collections import Counter
        methods = benchmark.all_methods()
        self.assertEqual(len(set(methods)), 114)
        self.assertEqual(Counter(map(benchmark.family, methods)),
                         dict(reference=2, lvq=7, sq=26, pq=57, opq=3, rq=2, lsq=2, gguf=5, fp8=10))
        self.assertEqual(benchmark.PROBES, (1, 2, 4, 8, 16, 32))
        self.assertEqual(benchmark.NOISE, .025)

    def test_default_shortlist_and_cli_selection(self):
        from collections import Counter
        methods = benchmark.DEFAULT_METHODS
        self.assertEqual(len(set(methods)), 20)
        self.assertTrue(set(methods) <= set(benchmark.all_methods()))
        self.assertEqual(Counter(map(benchmark.family, methods)),
                         dict(reference=1, lvq=4, sq=4, pq=4, opq=1, rq=1, lsq=1, gguf=3, fp8=1))
        command = [sys.executable, str(benchmark.ROOT / "benchmark_cluster_quantization.py")]
        for arguments, expected in (([], methods), (["--all-methods"], benchmark.all_methods()),
                                    (["--methods", "lvq6", "lvq6"], ("fp16", "lvq6"))):
            listed = subprocess.run(command + ["--list-methods", *arguments], check=True,
                                    capture_output=True, text=True)
            self.assertEqual(listed.stdout.splitlines(), list(expected))
        invalid = subprocess.run(command + ["--all-methods", "--methods", "lvq6"],
                                 capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)

    def test_run_uses_shortlist_when_methods_are_omitted(self):
        with patch.object(benchmark, "DEFAULT_METHODS", ("fp16", "lvq6")), \
                patch("balanced_l2.plot_posting_quantization.render"):
            saved = self.run_small(methods=None)
        self.assertEqual(saved["manifest"]["methods"], ["fp16", "lvq6"])
        self.assertEqual(len(saved["points"]), 2 * len(benchmark.PROBES))

    def test_prefix_winners_preserve_centroid_rank_ties(self):
        rows = np.array([[8, 4, 3, 2], [2, 3, 4, 5]])
        scores = np.array([[.7, .7, .8, .8], [-np.inf, .1, .2, .3]])
        winners, values = benchmark.prefix_winners(rows, scores, (1, 2, 4))
        np.testing.assert_array_equal(winners, [[8, 8, 3], [-1, 3, 5]])
        np.testing.assert_array_equal(values, [[.7, .7, .8], [-np.inf, .1, .3]])

    def test_checkpoint_cleanup_cannot_reach_another_directory(self):
        foreign = self.root / "checkpoint_32.npz"
        foreign.write_bytes(b"preserve this file")
        progress = {"done": 32, "arrays": {"path": str(foreign)}}
        with self.assertRaisesRegex(ValueError, "another case"):
            benchmark._checkpoint_path(self.output / "cases/fp16", progress)
        self.assertEqual(foreign.read_bytes(), b"preserve this file")

    def test_reranking_failure_is_not_counted_as_posting_coverage(self):
        records = np.zeros(2, record_dtype(2))
        records["code"], records["page"] = [1, 2], 1
        records["vec"] = [[1, 0], [0, 1]]
        points = benchmark._measure("lvq2", records, np.array([0]), np.array([[1., 0.]], np.float32),
                                    np.array([[0, 1]]), np.array([0, 1, 2]), np.array([100, 50]),
                                    np.array([[1]]), np.array([[.9]]), np.array([[0]]), (2,), 20, 30, 0.,
                                    dict(encode_seconds=0., decode_seconds=0., score_seconds=0.))
        self.assertEqual(points[0]["routing_coverage_percent"], 100)
        self.assertEqual(points[0]["recall_percent"], 0)
        self.assertEqual(points[0]["recall_vs_fp16_percent"], 0)
        self.assertEqual(points[0]["tie_aware_recall_percent"], 0)
        self.assertAlmostEqual(points[0]["cold_gzip_MB_mean"], 200 / 1e6)

    def test_real_codec_grid_resume_and_source_change_rejection(self):
        before = [snapshot(path) for path in sorted((self.source / "clusters").iterdir())]
        methods = ("bf16", "lvq3", "sq6", "pq8x4", "gguf_q4_0", "fp8_e4m3_row")
        saved = self.run_small(methods)
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(len(saved["points"]), 7 * 6)
        self.assertTrue(all(case["status"] == "complete" for case in saved["cases"].values()))
        self.assertTrue(saved["staging_removed"])
        self.assertFalse((self.output / "_staging").exists())
        self.assertTrue((self.output / "plots/index.html").is_file())
        for method in saved["manifest"]["methods"]:
            rows = [point for point in saved["points"] if point["method"] == method]
            self.assertEqual([point["probes"] for point in rows], list(benchmark.PROBES))
            self.assertEqual([point["posting_gzip_MB_mean"] for point in rows],
                             sorted(point["posting_gzip_MB_mean"] for point in rows))
            self.assertTrue(all(point["noise_std"] == .025 for point in rows))
        self.assertTrue(all(point["recall_vs_fp16_percent"] == 100 for point in saved["points"] if point["method"] == "fp16"))
        self.assertEqual(before, [snapshot(path) for path in sorted((self.source / "clusters").iterdir())])
        with patch.object(benchmark, "_stage", side_effect=AssertionError("restaged complete run")), \
                patch("balanced_l2.plot_posting_quantization.render"):
            self.assertEqual(self.run_small(methods)["points"], saved["points"])
        # Recover a kill after durable results but before the large staging file was removed.
        stage = self.output / "_staging/records.bin"
        stage.parent.mkdir()
        stage.write_bytes(b"owned temporary records")
        saved["staging_removed"] = False
        (self.output / "results.json").write_text(json.dumps(saved))
        with patch("balanced_l2.plot_posting_quantization.render"):
            self.assertTrue(self.run_small(methods)["staging_removed"])
        self.assertFalse(stage.exists())
        path = self.source / "clusters/cluster_0.bin.gz"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "Sources, settings"):
            self.run_small(methods)

    def test_completed_posting_checkpoint_survives_interruption(self):
        original = benchmark._checkpoint

        def interrupted(*args):
            original(*args)
            raise KeyboardInterrupt("test interruption after durable checkpoint")

        with patch.object(benchmark, "_checkpoint", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_small(methods=("fp16",))
        self.assertEqual(json.loads((self.output / "results.json").read_text())["status"], "running")
        with patch.object(benchmark, "_decode", side_effect=AssertionError("decoded completed postings")), \
                patch("balanced_l2.plot_posting_quantization.render"):
            saved = self.run_small(methods=("fp16",))
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(len(saved["points"]), 6)

    def test_invalid_codec_is_recorded_and_next_codec_runs(self):
        original = benchmark._encode

        def fail_one(raw, method, dimensions, model):
            if method == "lvq2":
                raise ValueError("invalid test reconstruction")
            return original(raw, method, dimensions, model)

        with patch.object(benchmark, "_encode", side_effect=fail_one), \
                patch("balanced_l2.plot_posting_quantization.render"):
            saved = self.run_small(methods=("lvq2", "lvq6"))
        self.assertEqual(saved["cases"]["lvq2"]["status"], "rejected")
        self.assertEqual(saved["cases"]["lvq6"]["status"], "complete")
        self.assertEqual(len(saved["points"]), 12)

    def test_complete_reuse_checks_shared_truth_integrity(self):
        with patch("balanced_l2.plot_posting_quantization.render"):
            self.run_small(methods=("fp16",))
            with (self.output / "truth.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Saved build artifact changed"):
                self.run_small(methods=("fp16",))


if __name__ == "__main__":
    unittest.main()
