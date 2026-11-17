"""Export the browser's LVQ6/LVQ8 posting files without changing FP16 sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

from balanced_l2.common import artifact, record_dtype, require_disk_space, snapshot, source_path, write_json
from balanced_l2.export import decode_centroids
from balanced_l2.posting_quantization import decode_posting, encode_posting


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = source_path(ROOT, "client/data")
BITS = (6, 8)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _owned_file(path):
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError(f"Owned export file must be a regular file: {path}")


def _export_one(directory, cid, dimensions, heads, source_snapshot, previous):
    source = directory / "clusters" / f"cluster_{cid}.bin.gz"
    if snapshot(source) != source_snapshot:
        raise ValueError(f"Source posting changed: {source}")
    compressed_source = source.read_bytes()
    source_sha256 = _digest(compressed_source)
    if snapshot(source) != source_snapshot:
        raise ValueError(f"Source posting changed while reading: {source}")
    if previous:
        if previous["source_sha256"] != source_sha256:
            raise ValueError(f"Source posting checksum changed: {source}")
        for bits in BITS:
            path = directory / f"clusters_lvq{bits}" / source.name
            _owned_file(path)
            saved = previous["formats"][f"lvq{bits}"]
            data = path.read_bytes()
            if len(data) != saved["bytes"] or _digest(data) != saved["sha256"]:
                raise ValueError(f"Saved quantized posting changed: {path}")
        return previous

    raw = gzip.decompress(compressed_source)
    dtype = record_dtype(dimensions)
    if len(raw) % dtype.itemsize:
        raise ValueError(f"Incomplete source records: {source}")
    records = np.frombuffer(raw, dtype)
    result = {"source_sha256": source_sha256, "count": len(records), "formats": {}}
    for bits in BITS:
        path = directory / f"clusters_lvq{bits}" / source.name
        temporary = path.with_suffix(path.suffix + ".tmp")
        _owned_file(path)
        _owned_file(temporary)
        payload = encode_posting(raw, dimensions, bits)
        encoded_dtype = np.dtype([("code", "<i4"), ("page", "<i2"), ("range", "<f2", (2,)),
                                  ("packed", "u1", ((dimensions * bits + 7) // 8,))])
        encoded = np.frombuffer(payload, encoded_dtype, count=len(records), offset=16 + 2 * dimensions)
        if (not np.array_equal(records["code"], encoded["code"])
                or not np.array_equal(records["page"], encoded["page"])):
            raise ValueError("Posting quantization changed record identities")
        if cid % 512 == 0 or cid == heads - 1:
            codes, pages, vectors = decode_posting(payload, dimensions)
            if (not np.array_equal(codes, records["code"]) or not np.array_equal(pages, records["page"])
                    or vectors.shape != (len(records), dimensions) or not np.isfinite(vectors).all()):
                raise ValueError("Quantized posting decode check failed")
        compressed = gzip.compress(payload, compresslevel=9, mtime=0)
        expected = {"bytes": len(compressed), "sha256": _digest(compressed)}
        if path.exists():
            # A kill can leave a durable file before its checkpoint was saved.
            if path.read_bytes() != compressed:
                raise ValueError(f"Refusing to replace an unrelated posting: {path}")
        else:
            with temporary.open("wb") as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        temporary.unlink(missing_ok=True)
        result["formats"][f"lvq{bits}"] = expected
    return result


def export(directory=DEFAULT_DATA, *, workers=8):
    directory = Path(directory).resolve()
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("Expected between 1 and 16 export workers")
    centroid_path = directory / "centroids.lvq6.bin.gz"
    centroid = artifact(centroid_path)
    payload = gzip.decompress(centroid_path.read_bytes())
    if len(payload) < 16:
        raise ValueError("Truncated centroid index")
    magic, version, dimensions, heads = struct.unpack("<8sHHI", payload[:16])
    if magic != b"LVQ6IDX\0" or version != 2:
        raise ValueError("Posting export requires the existing LVQ6 L2 routing index")
    decode_centroids(payload, heads, dimensions)
    sources = [directory / "clusters" / f"cluster_{cid}.bin.gz" for cid in range(heads)]
    if set((directory / "clusters").glob("cluster_*.bin.gz")) != set(sources):
        raise ValueError("Source postings must exactly cover the centroid index")
    inputs = [snapshot(path) for path in sources]
    manifest = {"schema_version": 1, "directory": str(directory), "dimensions": dimensions,
                "head_count": heads, "centroids": centroid, "sources": inputs,
                "bits": list(BITS), "gzip_level": 9,
                "encoder_sha256": artifact(ROOT / "balanced_l2/posting_quantization.py")["sha256"]}
    checkpoint, public = directory / "posting-codecs-export.json", directory / "posting-codecs.json"
    for path in (checkpoint, public):
        _owned_file(path)
    outputs = [directory / f"clusters_lvq{bits}" for bits in BITS]
    for path in outputs:
        if path.is_symlink() or path.exists() and not path.is_dir():
            raise ValueError(f"Owned posting directory must not be a symlink: {path}")
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["manifest"] != manifest:
            raise ValueError("Posting sources or encoder changed; refusing to mix export datasets")
    else:
        if public.exists() or any(path.exists() and any(path.iterdir()) for path in outputs):
            raise ValueError("Refusing unrelated existing quantized posting files")
        state = {"manifest": manifest, "status": "running", "completed": {}}
        write_json(checkpoint, state)
    if any(not str(cid).isdecimal() or not 0 <= int(cid) < heads for cid in state["completed"]):
        raise ValueError("Checkpoint contains an unknown posting ID")
    names = {path.name for path in sources}
    for path in outputs:
        path.mkdir(exist_ok=True)
        if any(item.name not in names and item.name not in {name + ".tmp" for name in names}
               for item in path.iterdir()):
            raise ValueError(f"Posting directory contains unrelated files: {path}")
    remaining_rows = 0
    for cid, source in enumerate(sources):
        if str(cid) not in state["completed"]:
            with source.open("rb") as stream:
                stream.seek(-4, os.SEEK_END)
                uncompressed = struct.unpack("<I", stream.read(4))[0]
            if uncompressed % record_dtype(dimensions).itemsize:
                raise ValueError("Source posting has incomplete records")
            remaining_rows += uncompressed // record_dtype(dimensions).itemsize
    require_disk_space(directory, remaining_rows * (20 + dimensions * 14 // 8))
    started = time.monotonic()
    finished = 0
    state["status"] = "running"
    write_json(checkpoint, state)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(_export_one, directory, cid, dimensions, heads, inputs[cid],
                                 state["completed"].get(str(cid))): cid for cid in range(heads)}
            try:
                for future in as_completed(tasks):
                    cid = tasks[future]
                    state["completed"][str(cid)] = future.result()
                    finished += 1
                    # Durable files are verified on resume; amortize the growing full checkpoint.
                    if finished % 512 == 0 or finished == heads:
                        write_json(checkpoint, state)
                    if finished % 128 == 0 or finished == heads:
                        print(f"Posting codecs {finished}/{heads}; {time.monotonic() - started:.1f}s", flush=True)
            except BaseException:
                for future in tasks:
                    future.cancel()
                raise
        if inputs != [snapshot(path) for path in sources] or artifact(centroid_path) != centroid:
            raise ValueError("Source dataset changed during posting export")
        formats = {"fp16": {"path": "clusters", "gzip_bytes": [item["bytes"] for item in inputs]}}
        for bits in BITS:
            name = f"lvq{bits}"
            formats[name] = {"path": f"clusters_{name}", "gzip_bytes": [state["completed"][str(cid)]["formats"][name]["bytes"]
                                                                      for cid in range(heads)]}
        for value in formats.values():
            value["total_gzip_bytes"] = sum(value["gzip_bytes"])
        published = {"schema_version": 1, "dimensions": dimensions, "head_count": heads,
                     "centroid_sha256": centroid["sha256"], "formats": formats}
        state["status"] = "complete"
        state.pop("last_error", None)
        write_json(checkpoint, state)
        write_json(public, published)
        return published
    except BaseException as error:
        state["last_error"] = str(error)
        write_json(checkpoint, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = export(args.data, workers=args.workers)
    print(json.dumps({name: value["total_gzip_bytes"] for name, value in result["formats"].items()}, indent=2))
