import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from balanced_l2.common import record_dtype
from balanced_l2.export import export_centroids
from balanced_l2 import export_posting_quantization as exporter
from balanced_l2.posting_quantization import decode_posting


class PostingExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "clusters").mkdir()
        heads = np.random.default_rng(8).normal(size=(3, 7)).astype(np.float32)
        native = self.root / "heads.bin"
        with native.open("wb") as stream:
            stream.write(struct.pack("<ii", 3, 7))
            heads.tofile(stream)
        export_centroids(self.root, native)
        for cid in range(3):
            rows = np.zeros(5, record_dtype(7))
            rows["code"], rows["page"] = np.arange(5) + 100 * cid, 1
            rows["vec"] = heads[cid] + np.random.default_rng(cid).normal(0, .1, size=(5, 7))
            (self.root / "clusters" / f"cluster_{cid}.bin.gz").write_bytes(gzip.compress(rows.tobytes(), mtime=0))

    def test_export_identity_sizes_resume_and_source_change_guard(self):
        originals = {p.name: p.read_bytes() for p in (self.root / "clusters").iterdir()}
        result = exporter.export(self.root, workers=2)
        self.assertEqual(result["head_count"], 3)
        self.assertEqual(set(result["formats"]), {"fp16", "lvq6", "lvq8"})
        for name, value in result["formats"].items():
            self.assertEqual(value["total_gzip_bytes"], sum(value["gzip_bytes"]))
            for cid in range(3):
                path = self.root / value["path"] / f"cluster_{cid}.bin.gz"
                self.assertEqual(path.stat().st_size, value["gzip_bytes"][cid])
                if name != "fp16":
                    codes, pages, values = decode_posting(gzip.decompress(path.read_bytes()), 7)
                    np.testing.assert_array_equal(codes, np.arange(5) + 100 * cid)
                    np.testing.assert_array_equal(pages, 1)
                    self.assertTrue(np.isfinite(values).all())
        self.assertEqual(originals, {p.name: p.read_bytes() for p in (self.root / "clusters").iterdir()})
        with patch.object(exporter, "encode_posting", side_effect=AssertionError("reencoded completed file")):
            self.assertEqual(exporter.export(self.root, workers=2), result)
        path = self.root / "clusters/cluster_0.bin.gz"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "sources or encoder changed"):
            exporter.export(self.root)

    def test_output_corruption_and_unrelated_directory_are_preserved(self):
        directory = self.root / "clusters_lvq6"
        directory.mkdir()
        foreign = directory / "cluster_0.bin.gz"
        foreign.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "unrelated existing"):
            exporter.export(self.root)
        self.assertEqual(foreign.read_bytes(), b"keep")
        foreign.unlink()
        exporter.export(self.root, workers=2)
        foreign.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Saved quantized posting changed"):
            exporter.export(self.root, workers=2)
        self.assertEqual(foreign.read_bytes(), b"corrupt")

    def test_uncheckpointed_durable_files_are_verified_before_adoption(self):
        first = exporter.export(self.root, workers=2)
        path = self.root / "posting-codecs-export.json"
        state = json.loads(path.read_text())
        state["completed"].pop("1")
        state["status"] = "running"
        path.write_text(json.dumps(state))
        self.assertEqual(exporter.export(self.root, workers=2), first)

    def test_interrupted_export_publishes_no_manifest_and_resumes(self):
        original = exporter._export_one

        def interrupted(*args):
            result = original(*args)
            raise KeyboardInterrupt("test interrupt after writing files")

        with patch.object(exporter, "_export_one", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                exporter.export(self.root, workers=1)
        self.assertFalse((self.root / "posting-codecs.json").exists())
        self.assertEqual(exporter.export(self.root, workers=2)["head_count"], 3)


if __name__ == "__main__":
    unittest.main()
