"""Repack original vector records without decoding or changing any bytes.

Usage: .venv/bin/python rebalance_vectors.py vectors vectors_rebalanced
Input files follow the clustering pipeline's numeric filename order.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from balanced_l2.common import input_files, iter_record_batches, record_dtype, require_disk_space, snapshot


def rebalance(source, destination, records_per_file=34000):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    record_bytes = record_dtype(384).itemsize
    if (isinstance(records_per_file, bool) or not isinstance(records_per_file, int)
            or not 0 < records_per_file <= 256 * 1024 ** 2 // record_bytes):
        raise ValueError("Records per file must be positive and fit a 256 MiB buffer")
    if not source.is_dir() or destination.resolve().is_relative_to(source):
        raise ValueError("Use an input directory and a separate output directory")
    files = input_files(source)
    before = [snapshot(path) for path in files]
    total_bytes = sum(item["bytes"] for item in before)
    if not total_bytes or any(item["bytes"] % record_bytes for item in before):
        raise ValueError("Expected nonempty input with complete 774-byte records")
    require_disk_space(destination.parent, total_bytes)
    destination.mkdir()
    expected, actual = hashlib.sha256(), hashlib.sha256()
    sizes = []
    try:
        for index, records in enumerate(iter_record_batches(files, 384, records_per_file), start=1):
            raw = memoryview(records).cast("B")
            expected.update(raw)
            with (destination / f"{index}.bin").open("xb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            sizes.append(len(raw))
            if index % 100 == 0:
                print(f"Wrote {index} files", file=sys.stderr, flush=True)
        if sum(sizes) != total_bytes:
            raise ValueError("Repacked byte count differs from the source")
        print("Verifying output bytes", file=sys.stderr, flush=True)
        for index, size in enumerate(sizes, start=1):
            path = destination / f"{index}.bin"
            if path.stat().st_size != size:
                raise ValueError(f"Output size changed: {path}")
            with path.open("rb") as output:
                for block in iter(lambda: output.read(1024 ** 2), b""):
                    actual.update(block)
            if index % 100 == 0:
                print(f"Verified {index} files", file=sys.stderr, flush=True)
        if actual.digest() != expected.digest():
            raise ValueError("Output bytes differ from the original record stream")
        if [snapshot(path) for path in input_files(source)] != before:
            raise ValueError("Source files changed during repacking")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {"source": str(source), "destination": str(destination), "record_bytes": record_bytes,
            "records": total_bytes // record_bytes, "bytes": total_bytes, "input_files": len(files),
            "output_files": len(sizes), "records_per_file": records_per_file,
            "last_file_records": sizes[-1] // record_bytes,
            "source_sha256": expected.hexdigest(), "output_sha256": actual.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--records-per-file", type=int, default=34000)
    args = parser.parse_args()
    print(json.dumps(rebalance(args.source, args.destination, args.records_per_file), indent=2))
