"""Small format, resume, memory-bound, and exact-record export checks."""

import gzip
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from balanced_l2 import build as pipeline, common, export


def matrix(path, values):
    values = np.asarray(values, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<ii", *values.shape) + values.tobytes())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="balanced-l2-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = np.zeros(23, dtype=common.record_dtype(3))
        self.records["code"] = np.arange(len(self.records)) + 100
        self.records["page"] = np.arange(len(self.records)) % 7
        self.records["vec"] = np.random.default_rng(17).normal(size=(len(self.records), 3)).astype(np.float16)
        self.files = [self.root / "2.bin", self.root / "10.bin"]
        self.records[:7].tofile(self.files[0])
        self.records[7:].tofile(self.files[1])

    def comparison_heads(self, name="comparison"):
        root = self.root / name
        heads = root / "h3/native/heads.f32"
        matrix(heads, [[.8, 0, 0], [0, .7, 0], [0, 0, .6]])
        sample = {"path": str(root / "deleted-sample.f32"), "bytes": 248,
                  "mtime_ns": 123, "sha256": "f" * 64}
        training = {"method": "balanced_l2", "metric": "l2", "head_count": 3,
                    "dimensions": 3, "sample_rows": 20, "sample": sample,
                    "heads": common.artifact(heads),
                    "params": {"n_clusters": 3, "metric": "sqeuclidean", "hierarchical": True,
                               "hierarchical_n_iters": 20}}
        common.write_json(heads.with_name("training.json"), training)
        comparison = {"schema_version": 1, "status": "complete",
                      "inputs": [common.snapshot(path) for path in self.files],
                      "source_rows": len(self.records), "dimensions": 3, "sample_rows": 20,
                      "sample_seed": 42, "max_iter": 20, "sample": sample,
                      "completed": [{"head_count": 3, "centroid_method": "lvq6", "metric": "l2"}]}
        common.write_json(root / "results.json", comparison)
        return heads

    def test_input_order_normalization_and_common_sample(self):
        self.assertEqual(common.input_files(self.root), [path.resolve() for path in self.files])
        selected = np.array([0, 6, 7, 22])
        source, sampled = self.root / "vectors.f32", self.root / "sample.f32"
        rows = common.prepare_vectors(self.files, source, 3, 4, selected, sampled)
        self.assertEqual(rows, len(self.records))
        expected = self.records["vec"].astype(np.float32)
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_array_equal(common.read_matrix(source), expected)
        np.testing.assert_array_equal(common.read_matrix(sampled), expected[selected])
        with self.assertRaises(FileExistsError):
            common.prepare_vectors(self.files, source, 3)

    def test_assignment_keeps_nonunit_l2_means_and_is_atomic(self):
        source, heads, labels = (self.root / name for name in ("vectors.f32", "heads.f32", "labels.i32"))
        matrix(source, [[.9, 0], [0, 0], [0, 1], [9, 0]])
        matrix(heads, [[10, 0], [1, 0], [0, 1]])
        with patch.object(pipeline, "GPU_SCORE_BYTES", 12):
            result = pipeline.assign(source, heads, labels, gpu=False)
        self.assertEqual(result["query_batch_size"], 1)
        np.testing.assert_array_equal(np.fromfile(labels, "<i4"), [1, 1, 2, 0])
        with self.assertRaises(FileExistsError):
            pipeline.assign(source, heads, labels, gpu=False)
        matrix(source, [[np.nan, 0]])
        bad = self.root / "bad.i32"
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            pipeline.assign(source, heads, bad, gpu=False)
        self.assertFalse(bad.exists())
        self.assertFalse(bad.with_name(bad.name + ".tmp").exists())

    def check_postings(self, output, labels, heads):
        for cid in range(heads):
            path = output / "clusters" / f"cluster_{cid}.bin.gz"
            self.assertEqual(gzip.decompress(path.read_bytes()), self.records[labels == cid].tobytes())
        np.testing.assert_array_equal(np.load(output / "posting_counts.npy"), np.bincount(labels, minlength=heads))
        self.assertFalse((output / "_export_spool").exists())
        self.assertFalse(list(output.rglob("*.tmp")))

    def test_sharded_export_preserves_original_record_order_and_empty_clusters(self):
        labels = np.arange(len(self.records), dtype=np.int32) % 5
        path = self.root / "labels.i32"
        labels.tofile(path)
        output = self.root / "export"
        kwargs = dict(dim=3, batch_size=4, threads=2, heads_per_shard=2)
        result = export.export_postings(self.files, path, 6, output, **kwargs)
        self.assertEqual(result["assignments_per_vector"], 1)
        self.assertEqual(result["posting_records"], len(self.records))
        self.assertEqual(result["nonempty_clusters"], 5)
        self.check_postings(output, labels, 6)
        self.assertEqual(result, export.export_postings(self.files, path, 6, output, **kwargs))
        with self.files[0].open("ab") as source:
            source.write(b"changed")
        with self.assertRaises(ValueError):
            export.export_postings(self.files, path, 6, output, **kwargs)

    def test_export_resumes_after_completed_shard_checkpoint(self):
        labels = np.arange(len(self.records), dtype=np.int32) % 5
        path, output = self.root / "labels.i32", self.root / "export"
        labels.tofile(path)
        original = export.write_json

        def interrupt(checkpoint, value):
            original(checkpoint, value)
            if value.get("completed_shards") == [0]:
                raise RuntimeError("interrupt after shard")

        kwargs = dict(dim=3, batch_size=4, threads=2, heads_per_shard=2)
        with patch.object(export, "write_json", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupt after shard"):
                export.export_postings(self.files, path, 6, output, **kwargs)
        before = (output / "clusters/cluster_0.bin.gz").read_bytes()
        export.export_postings(self.files, path, 6, output, **kwargs)
        self.assertEqual((output / "clusters/cluster_0.bin.gz").read_bytes(), before)
        self.check_postings(output, labels, 6)

    def test_centroid_export_marks_l2_and_keeps_nonunit_means(self):
        heads = self.root / "heads.f32"
        values = np.array([[.2, .3, .4], [1., 2., 3.], [0, 0, 0]], np.float32)
        matrix(heads, values)
        result = export.export_centroids(self.root, heads)
        self.assertEqual(result["centroid_metric"], "l2")
        raw = gzip.decompress((self.root / "centroids.lvq6.bin.gz").read_bytes())
        self.assertEqual(struct.unpack("<8sHHI", raw[:16]), (b"LVQ6IDX\0", 2, 3, 3))
        np.testing.assert_array_equal(np.frombuffer(gzip.decompress((self.root / "centroids.bin.gz").read_bytes()), "<f2").reshape(3, 3), values.astype(np.float16))

    def test_lvq6_decoder_matches_literal_wire_values_and_rejects_bad_payloads(self):
        for dimensions in (1, 3, 7, 384):
            values = np.random.default_rng(dimensions).normal(size=(5, dimensions)).astype(np.float32) + 2
            heads = self.root / "heads.f32"
            matrix(heads, values)
            export.export_centroids(self.root, heads)
            raw = gzip.decompress((self.root / "centroids.lvq6.bin.gz").read_bytes())
            decoded = export.decode_centroids(raw, 5, dimensions)
            metadata = np.frombuffer(raw, "<f2", dimensions + 10, offset=16).astype(np.float64)
            mean, lower, delta = np.split(metadata, [dimensions, dimensions + 5])
            row_bytes, prefix = (dimensions * 6 + 7) // 8, 16 + 2 * dimensions + 20
            expected = np.empty_like(values)
            for row in range(5):
                codes = int.from_bytes(raw[prefix + row * row_bytes:prefix + (row + 1) * row_bytes], "little")
                for column in range(dimensions):
                    expected[row, column] = lower[row] + ((codes >> (column * 6)) & 63) * delta[row] + mean[column]
            np.testing.assert_array_equal(decoded, expected)
            self.assertGreater(float(decoded.mean()), 1)
            for bad in (raw[:-1], raw + b"\0", raw[:8] + b"\1\0" + raw[10:]):
                with self.assertRaises(ValueError):
                    export.decode_centroids(bad, 5, dimensions)
            bad = bytearray(raw)
            bad[16 + 2 * dimensions + 10:18 + 2 * dimensions + 10] = b"\0\0"
            with self.assertRaisesRegex(ValueError, "positive scales"):
                export.decode_centroids(bad, 5, dimensions)
            if dimensions * 6 % 8:
                bad = bytearray(raw)
                bad[-1] |= 128
                with self.assertRaisesRegex(ValueError, "padding"):
                    export.decode_centroids(bad, 5, dimensions)

    def test_adopted_build_exports_without_staging_or_retraining(self):
        heads, labels_path = self.root / "heads.f32", self.root / "labels.i32"
        matrix(heads, [[1, 0, 0], [0, 1, 0], [0, 0, .5]])
        labels = np.arange(len(self.records), dtype=np.int32) % 3
        labels.tofile(labels_path)
        output = self.root / "adopted"
        config = dict(dim=3, batch_size=4, threads=2, sample_size=20, gpu=False)
        state = pipeline.adopt_build(output, self.files, heads, labels_path, {"metric": "l2"}, {"backend": "numpy"}, **config)
        self.assertEqual(state["completed"], ["vectors", "heads", "assignments"])
        with patch.object(pipeline, "train", side_effect=AssertionError("retrained")), \
                patch.object(pipeline, "prepare_vectors", side_effect=AssertionError("restaged")):
            result = pipeline.build(self.files, output, 3, resume=True, export_only=True, **config)
        self.assertEqual(result["method"], "balanced_l2")
        self.check_postings(output, labels, 3)
        self.assertFalse((output / "_work").exists())
        self.assertEqual(pipeline.build(self.files, output, 3, resume=True, export_only=True, **config), result)
        with self.assertRaisesRegex(ValueError, "parameters differ"):
            pipeline.build(self.files, output, 3, resume=True, export_only=True, **{**config, "seed": 1})

    def test_fresh_pipeline_resume_reuses_completed_assignments(self):
        output = self.root / "fresh"

        def fake_train(sample, native, clusters, max_iter):
            matrix(native / "heads.f32", [[.8, 0, 0], [0, .7, 0], [0, 0, .6]])
            return {"metric": "l2", "sample_rows": common.matrix_shape(sample)[0]}

        original_export = pipeline.export_postings
        config = dict(dim=3, batch_size=4, threads=2, sample_size=20, gpu=False)
        with patch.object(pipeline, "train", side_effect=fake_train), \
                patch.object(pipeline, "export_postings", side_effect=RuntimeError("export interrupted")):
            with self.assertRaisesRegex(RuntimeError, "export interrupted"):
                pipeline.build(self.files, output, 3, **config)
        state = json.loads((output / "build.json").read_text())
        self.assertIn("assignments", state["completed"])
        self.assertFalse((output / "_work").exists())
        with patch.object(pipeline, "train", side_effect=AssertionError("retrained")), \
                patch.object(pipeline, "export_postings", side_effect=original_export):
            result = pipeline.build(self.files, output, 3, resume=True, **config)
        self.assertEqual(result["posting_records"], len(self.records))

    def test_streamed_normalized_slices_are_bounded_and_keep_source_order(self):
        source = pipeline._NormalizedRecords(self.files, len(self.records), 3, 4)
        expected = self.records["vec"].astype(np.float32)
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        try:
            with self.assertRaisesRegex(ValueError, "bounded sequential"):
                source[0:5]
            batches = [source[0:3], source[3:7]]
            for first in range(7, len(self.records), 4):
                batches.append(source[first:min(first + 4, len(self.records))])
            np.testing.assert_array_equal(np.concatenate(batches), expected)
            with self.assertRaisesRegex(ValueError, "bounded sequential"):
                source[0:1]
        finally:
            source.close()

    def test_trained_heads_resume_without_retaining_a_sample_or_full_fp32_corpus(self):
        output = self.root / "streamed"

        def fake_train(sample, native, clusters, max_iter):
            selected = np.sort(np.random.default_rng(42).choice(len(self.records), 20, replace=False))
            expected = self.records["vec"][selected].astype(np.float32)
            expected /= np.linalg.norm(expected, axis=1)[:, None]
            np.testing.assert_array_equal(common.read_matrix(sample), expected)
            self.assertFalse((output / "_work/vectors.f32").exists())
            matrix(native / "heads.f32", [[.8, 0, 0], [0, .7, 0], [0, 0, .6]])
            return {"metric": "l2", "sample_rows": len(selected)}

        config = dict(dim=3, batch_size=4, threads=2, sample_size=20, gpu=False)
        with patch.object(pipeline, "train", side_effect=fake_train), \
                patch.object(pipeline, "assign_vectors", side_effect=RuntimeError("assignment interrupted")):
            with self.assertRaisesRegex(RuntimeError, "assignment interrupted"):
                pipeline.build(self.files, output, 3, **config)
        state = json.loads((output / "build.json").read_text())
        self.assertEqual(state["completed"], ["vectors", "heads"])
        self.assertEqual(set(state["staging"]), {"sample"})
        self.assertFalse((output / "_work").exists())
        with patch.object(pipeline, "train", side_effect=AssertionError("retrained")), \
                patch.object(pipeline, "iter_record_batches", wraps=common.iter_record_batches):
            result = pipeline.build(self.files, output, 3, resume=True, **config)
        self.assertEqual(result["posting_records"], len(self.records))

    def test_brotli_size_guard_precedes_assignment_and_retains_trained_heads(self):
        output = self.root / "brotli"

        def fake_train(sample, native, clusters, max_iter):
            matrix(native / "heads.f32", [[.8, 0, 0], [0, .7, 0], [0, 0, .6]])
            return {"metric": "l2", "sample_rows": common.matrix_shape(sample)[0]}

        config = dict(dim=3, batch_size=4, threads=2, sample_size=20, gpu=False, centroid_brotli=True)
        with patch.object(pipeline, "train", side_effect=fake_train), \
                patch.object(pipeline, "export_centroids_brotli", side_effect=ValueError("Brotli exceeds asset limit")), \
                patch.object(pipeline, "assign_vectors", side_effect=AssertionError("assigned before size check")):
            with self.assertRaisesRegex(ValueError, "asset limit"):
                pipeline.build(self.files, output, 3, **config)
        state = json.loads((output / "build.json").read_text())
        self.assertEqual(state["completed"], ["vectors", "heads"])
        self.assertTrue(state["parameters"]["centroid_brotli"])
        self.assertFalse((output / "_work").exists())
        self.assertTrue((output / "native/heads.f32").exists())
        self.assertFalse((output / "native/assignments.i32").exists())
        with patch.object(pipeline, "train", side_effect=AssertionError("retrained")):
            result = pipeline.build(self.files, output, 3, resume=True, **config)
        self.assertEqual(result["posting_records"], len(self.records))
        self.assertTrue((output / "centroids.lvq6.bin.br").exists())
        centroid_bytes = (output / "centroids.lvq6.bin.br").read_bytes()
        with patch.object(pipeline, "export_centroids", side_effect=AssertionError("reencoded centroids")):
            pipeline.build(self.files, output, 3, resume=True, **config)
        self.assertEqual((output / "centroids.lvq6.bin.br").read_bytes(), centroid_bytes)

    def test_staging_cleanup_refuses_unrelated_files(self):
        output = self.root / "protected"

        def fake_train(sample, native, clusters, max_iter):
            matrix(native / "heads.f32", [[.8, 0, 0], [0, .7, 0], [0, 0, .6]])
            (sample.parent / "keep.txt").write_text("unrelated")
            return {"metric": "l2", "sample_rows": common.matrix_shape(sample)[0]}

        with patch.object(pipeline, "train", side_effect=fake_train):
            with self.assertRaisesRegex(ValueError, "unrelated build staging"):
                pipeline.build(self.files, output, 3, dim=3, batch_size=4,
                               sample_size=20, threads=2, gpu=False)
        self.assertEqual((output / "_work/keep.txt").read_text(), "unrelated")
        self.assertTrue((output / "_work/sample.f32").exists())
        self.assertTrue((output / "native/heads.f32").exists())

    def test_imported_comparison_heads_skip_sampling_and_training_and_resume_independently(self):
        heads = self.comparison_heads()
        original = heads.read_bytes()
        output = self.root / "reused"
        config = dict(dim=3, batch_size=4, threads=2, sample_size=20, gpu=False,
                      centroid_brotli=True, heads_from=heads)
        with patch.object(pipeline, "train", side_effect=AssertionError("retrained")), \
                patch.object(pipeline, "prepare_vectors", side_effect=AssertionError("restaged")):
            result = pipeline.build(self.files, output, 3, **config)
        self.assertEqual(result["posting_records"], len(self.records))
        self.assertEqual((output / "native/heads.f32").read_bytes(), original)
        self.assertFalse((output / "_work").exists())
        state = json.loads((output / "build.json").read_text())
        self.assertNotIn("vectors", state["completed"])
        self.assertEqual(state["parameters"]["heads_from"], str(heads.resolve()))
        imported = json.loads((output / "native/training.json").read_text())
        self.assertEqual(imported["source_heads"]["path"], str(heads.resolve()))
        self.assertEqual(imported["heads"], common.artifact(output / "native/heads.f32"))
        heads.unlink()
        heads.with_name("training.json").unlink()
        (heads.parents[2] / "results.json").unlink()
        with patch.object(pipeline, "train", side_effect=AssertionError("retrained")):
            self.assertEqual(pipeline.build(self.files, output, 3, resume=True, **config), result)

    def test_imported_heads_reject_changed_provenance_or_artifacts_before_assignment(self):
        for failure in ("inputs", "sample", "iterations", "head_hash"):
            with self.subTest(failure=failure):
                heads = self.comparison_heads(f"comparison-{failure}")
                comparison_path = heads.parents[2] / "results.json"
                training_path = heads.with_name("training.json")
                comparison = json.loads(comparison_path.read_text())
                training = json.loads(training_path.read_text())
                if failure == "inputs":
                    comparison["inputs"][0]["bytes"] += 1
                if failure == "sample":
                    training["sample"]["sha256"] = "0" * 64
                if failure == "iterations":
                    training["params"]["hierarchical_n_iters"] = 21
                if failure == "head_hash":
                    training["heads"]["sha256"] = "0" * 64
                common.write_json(comparison_path, comparison)
                common.write_json(training_path, training)
                output = self.root / f"rejected-{failure}"
                with patch.object(pipeline, "train", side_effect=AssertionError("retrained")), \
                        patch.object(pipeline, "assign_vectors", side_effect=AssertionError("assigned")):
                    with self.assertRaisesRegex(ValueError, "provenance|artifact"):
                        pipeline.build(self.files, output, 3, dim=3, batch_size=4,
                                       threads=2, sample_size=20, gpu=False, heads_from=heads)
                self.assertFalse((output / "native/assignments.i32").exists())

    @unittest.skipUnless(os.environ.get("BALANCED_L2_TEST_GPU") == "1", "GPU opt-in")
    def test_real_gpu_training_and_assignment(self):
        from balanced_l2.build import train
        source, native = self.root / "vectors.f32", self.root / "native"
        values = np.random.default_rng(3).normal(size=(256, 8)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1)[:, None]
        matrix(source, values)
        result = train(source, native, clusters=16, max_iter=3)
        self.assertEqual(result["head_count"], 16)
        for gpu in (False, True):
            pipeline.assign(source, native / "heads.f32", native / f"labels_{gpu}.i32", gpu=gpu)
        self.assertEqual((native / "labels_False.i32").read_bytes(), (native / "labels_True.i32").read_bytes())


if __name__ == "__main__":
    unittest.main()
