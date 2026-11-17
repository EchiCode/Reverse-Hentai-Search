"""Byte preservation, shard boundaries, and refusal to overwrite data."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rebalance_vectors import rebalance


class RebalanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def test_raw_records_cross_files_in_numeric_order_and_keep_every_bit(self):
        # Include signed zero and NaN bit patterns; no floating-point round trip is allowed.
        raw = b"\x01\x00\x00\x00\x01\x00\x00\x00\x00\x80\x01\x7e\xff\x7b" + bytes(range(256)) * (7 * 774 // 256 + 1)
        raw = raw[:7 * 774]
        for name, content in (("1.bin", b""), ("2.bin", raw[:774]), ("10.bin", raw[774:])):
            (self.source / name).write_bytes(content)
        destination = self.root / "output"
        report = rebalance(self.source, destination, 3)
        self.assertEqual([p.stat().st_size for p in sorted(destination.iterdir())], [3 * 774, 3 * 774, 774])
        self.assertEqual(b"".join((destination / f"{i}.bin").read_bytes() for i in (1, 2, 3)), raw)
        self.assertEqual(report["records"], 7)
        self.assertEqual(report["source_sha256"], report["output_sha256"])
        self.assertEqual((self.source / "10.bin").read_bytes(), raw[774:])

    def test_exact_multiple_does_not_add_an_empty_file(self):
        (self.source / "1.bin").write_bytes(b"\xff" * 4 * 774)
        destination = self.root / "output"
        report = rebalance(self.source, destination, 2)
        self.assertEqual(report["output_files"], 2)
        self.assertEqual(report["last_file_records"], 2)

    def test_write_failure_keeps_source_and_removes_partial_output(self):
        original = b"a" * 2 * 774
        path = self.source / "1.bin"
        path.write_bytes(original)
        destination = self.root / "output"
        with patch("rebalance_vectors.os.fsync", side_effect=OSError("disk write failed")):
            with self.assertRaisesRegex(OSError, "disk write failed"):
                rebalance(self.source, destination, 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(destination.exists())

    def test_invalid_or_existing_destinations_never_overwrite_inputs(self):
        path = self.source / "1.bin"
        path.write_bytes(b"a" * 774)
        with self.assertRaises(ValueError):
            rebalance(self.source, self.source / "nested")
        with self.assertRaises(ValueError):
            rebalance(self.source, self.source)
        destination = self.root / "output"
        destination.mkdir()
        (destination / "1.bin").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            rebalance(self.source, destination)
        self.assertEqual((destination / "1.bin").read_bytes(), b"keep")
        path.write_bytes(b"truncated")
        with self.assertRaises(ValueError):
            rebalance(self.source, self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())
        self.assertEqual(path.read_bytes(), b"truncated")


if __name__ == "__main__":
    unittest.main()
