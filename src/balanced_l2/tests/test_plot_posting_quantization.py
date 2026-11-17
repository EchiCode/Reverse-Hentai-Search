"""Saved plotting results must retain the benchmark's query and byte accounting."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from balanced_l2 import plot_posting_quantization as plots
from balanced_l2.plot_posting_quantization import render


class PostingQuantizationPlotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="posting-quantization-plots-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.saved = {
            "manifest": {"query_count": 100, "noise_std": .025, "probes": [1, 2, 4, 8],
                         "head_count": 8000, "dimensions": 384,
                         "methods": ["fp16", "lvq6", "sq8", "lvq7"]},
            "status": "complete",
            "points": [],
            "cases": {"fp16": {"status": "complete"}, "lvq6": {"status": "complete"},
                      "sq8": {"status": "complete"},
                      "lvq7": {"status": "rejected", "reason": "nonfinite <range>"}},
        }
        for method, family, recall, size, model in (("fp16", "reference", 99, 30, 0),
                                                    ("lvq6", "lvq", 98, 12, 0),
                                                    ("sq8", "sq", 95, 3, 0)):
            for probes in self.saved["manifest"]["probes"]:
                self.saved["points"].append({"method": method, "family": family,
                    "queries": 100, "noise_std": .025, "probes": probes,
                    "recall_percent": recall - 3 + probes.bit_length(), "recall_vs_fp16_percent": recall,
                    "routing_coverage_percent": 100, "posting_gzip_MB_mean": size * probes,
                    "posting_gzip_MB_p95": size * probes * 1.5, "model_gzip_MB": model,
                    "centroid_gzip_MB": 2, "cold_gzip_MB_mean": size * probes + model + 2,
                    "total_posting_gzip_GB": size, "encode_seconds": .5,
                    "decode_seconds": .1, "score_seconds": .2})

    def plot(self, saved=None):
        (self.root / "results.json").write_text(json.dumps(saved or self.saved))
        return render(self.root)

    def test_writes_all_formats_and_discloses_rejected_methods(self):
        result = self.plot()
        self.assertEqual(result, {"figures": ["quantization", "storage_recall", "comparison_lvq"],
                                  "points": 4, "rejected": 1})
        for stem in result["figures"]:
            for extension in ("png", "svg", "pdf"):
                self.assertGreater((self.root / "plots" / f"{stem}.{extension}").stat().st_size, 100)
        page = (self.root / "plots/index.html").read_text()
        self.assertIn("../results.csv", page)
        self.assertIn("nonfinite &lt;range&gt;", page)
        self.assertIn("not real-image recall", page)
        self.assertIn("timings exclude browser/network latency", page)
        self.assertIn("tie-aware recall is in CSV/JSON", page)
        self.assertIn("2 measured methods across 2 probe counts", page)
        self.assertIn("href='#comparison_lvq'", page)

    def test_rejects_unaccounted_methods_and_invalid_measurements(self):
        alterations = [
            lambda saved: saved["points"].append(saved["points"][0]),
            lambda saved: saved["points"].pop(),
            lambda saved: saved["cases"].pop("lvq7"),
            lambda saved: saved["points"][0].update(cold_gzip_MB_mean=1),
            lambda saved: saved["points"][0].update(recall_percent=float("nan")),
            lambda saved: saved["points"][0].update(posting_gzip_MB_mean=0),
            lambda saved: saved["points"][0].update(queries=99),
            lambda saved: saved["points"][0].update(probes=3),
        ]
        for alter in alterations:
            with self.subTest(alter=alter):
                saved = deepcopy(self.saved)
                alter(saved)
                with self.assertRaises(ValueError):
                    self.plot(saved)

    def test_excludes_unwanted_codecs_and_adds_fp16_to_each_single_panel_comparison(self):
        saved = deepcopy(self.saved)
        excluded = (("rq4x6", "rq", "rq"), ("lsq4x6", "lsq", "lsq"),
                    ("gguf_q4_0", "gguf", "gguf"), ("opq64x8", "opq", "opq"),
                    ("pq192x8", "pq", "pq_m192"), ("pq96x8", "pq", "pq_m96"),
                    ("pq64x6", "pq", "pq_m64"), ("pq32x4", "pq", "pq_m32"),
                    ("fp8_e4m3_row", "fp8", "fp8_e4m3"))
        for method, family, group in excluded:
            saved["manifest"]["methods"].append(method)
            saved["cases"][method] = {"status": "complete"}
            for point in self.saved["points"][:4]:
                saved["points"].append(dict(point, method=method, family=family))
        output = self.root / "plots"
        output.mkdir()
        for method, family, group in excluded:
            for extension in ("png", "svg", "pdf"):
                (output / f"family_{group}.{extension}").write_text("stale plot")
        (output / "notes.txt").write_text("keep")
        with patch.object(plots, "_curves", wraps=plots._curves) as curves:
            result = self.plot(saved)
        self.assertEqual(result["points"], 4)
        self.assertEqual(json.loads((self.root / "results.json").read_text()), saved)
        for call in curves.call_args_list:
            axis, points, cost = call.args
            self.assertEqual(len(axis.figure.axes), 1)
            self.assertEqual(cost, "posting_gzip_MB_mean")
            self.assertEqual([p["probes"] for p in points if p["method"] == "fp16"], [4, 8])
            self.assertTrue(all(point["probes"] >= 4 for point in points))
            self.assertFalse(any(p["method"] in {method for method, _, _ in excluded} for p in points))
            self.assertFalse(any(p["family"] == "sq" for p in points))
            baseline = next(line for line in axis.lines if line.get_label() == "FP16")
            self.assertEqual(baseline.get_linestyle(), "--")
            self.assertTrue({1, 2, 3, 5, 7, 10} <= set(axis.xaxis.get_major_locator().tick_values(1, 10)))
            self.assertEqual(axis.xaxis.get_major_formatter()(2), "2")
            self.assertEqual(axis.xaxis.get_minor_formatter()(4), "")
        self.assertEqual(len(curves.call_args_list), 1)
        for method, family, group in excluded:
            self.assertFalse(any(output.glob(f"family_{group}.*")))
        self.assertEqual((output / "notes.txt").read_text(), "keep")
        page = (output / "index.html").read_text()
        for method, family, group in excluded:
            self.assertNotIn(f"href='#family_{group}'", page)
        self.assertIn("Cold measurements remain in CSV/JSON", page)
        self.assertIn("probes 4, 8", page)

    def test_combines_all_lvq_with_fp16_and_removes_sq_and_old_comparison_plots(self):
        saved = deepcopy(self.saved)
        for method in ("lvq4", "lvq5", "lvq8", "sq4", "sq6", "sq6_quantile0.005"):
            saved["manifest"]["methods"].append(method)
            saved["cases"][method] = {"status": "complete"}
            saved["points"].extend(dict(point, method=method, family="lvq" if method.startswith("lvq") else "sq")
                                   for point in self.saved["points"][:4])
        output = self.root / "plots"
        output.mkdir()
        old_groups = ("reference", "lvq", "sq", "sq4", "sq6", "sq8")
        for group in old_groups:
            for extension in ("png", "svg", "pdf"):
                (output / f"family_{group}.{extension}").write_text("old family plot")
        for group in ("4bit", "5_6bit", "8bit"):
            for extension in ("png", "svg", "pdf"):
                (output / f"comparison_{group}.{extension}").write_text("old comparison plot")
        with patch.object(plots, "_curves", wraps=plots._curves) as curves:
            result = self.plot(saved)
        expected = [{"fp16", "lvq4", "lvq5", "lvq6", "lvq8"}]
        self.assertEqual([{point["method"] for point in call.args[1]} for call in curves.call_args_list], expected)
        self.assertEqual(result["figures"], ["quantization", "storage_recall", "comparison_lvq"])
        self.assertEqual(json.loads((self.root / "results.json").read_text()), saved)
        for call in curves.call_args_list:
            self.assertEqual([point["probes"] for point in call.args[1] if point["method"] == "fp16"], [4, 8])
        for group in old_groups:
            self.assertFalse(any(output.glob(f"family_{group}.*")))
        for group in ("4bit", "5_6bit", "8bit"):
            self.assertFalse(any(output.glob(f"comparison_{group}.*")))


if __name__ == "__main__":
    unittest.main()
