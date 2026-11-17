"""Bounded, resumable gzip posting export in original source-record order."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import struct

import numpy as np

from balanced_l2.common import (artifact, iter_record_batches, read_matrix, record_dtype,
                                require_disk_space, snapshot, source_path, write_json)


def decode_centroids(payload, count, dimensions):
    """Decode the production LVQ6 L2 index with the browser's float32 rounding."""
    prefix = 16 + 2 * dimensions + 4 * count
    row_bytes = (dimensions * 6 + 7) // 8
    if (count <= 0 or dimensions <= 0 or len(payload) != prefix + count * row_bytes
            or struct.unpack("<8sHHI", payload[:16]) != (b"LVQ6IDX\0", 2, dimensions, count)):
        raise ValueError("Invalid LVQ6 L2 centroid header or payload size")
    metadata = np.frombuffer(payload, dtype="<f2", count=dimensions + 2 * count, offset=16).astype(np.float64)
    mean, lower, delta = np.split(metadata, [dimensions, dimensions + count])
    if not np.isfinite(metadata).all() or (delta <= 0).any():
        raise ValueError("LVQ6 centroid metadata must be finite with positive scales")
    packed = np.frombuffer(payload, dtype=np.uint8, offset=prefix).reshape(count, row_bytes)
    used_bits = dimensions * 6 % 8
    if used_bits and (packed[:, -1] >> used_bits).any():
        raise ValueError("LVQ6 row padding must be zero")
    bits = np.unpackbits(packed, axis=1, count=dimensions * 6, bitorder="little").reshape(count, dimensions, 6)
    codes = np.sum(bits << np.arange(6, dtype=np.uint8), axis=2, dtype=np.uint8)
    decoded = (lower[:, None] + codes * delta[:, None] + mean).astype(np.float32)
    if not np.isfinite(decoded).all():
        raise ValueError("LVQ6 centroid reconstruction must be finite")
    return decoded


def export_centroids(output, heads_path):
    output = Path(output)
    heads = read_matrix(heads_path)
    with np.errstate(over="ignore"):
        rounded = heads.astype("<f2")
    if not np.isfinite(rounded).all():
        raise ValueError("Centroids cannot be represented as finite float16")
    source = output / "centroids.bin.gz"
    temporary = source.with_suffix(".tmp")
    with temporary.open("wb") as target, gzip.GzipFile(filename=source.name, mode="wb", fileobj=target, mtime=0) as compressed:
        compressed.write(rounded.tobytes())
    temporary.replace(source)
    encoder_path = source_path(Path(__file__).resolve().parents[1], "client/scripts/encode-centroids.py")
    spec = importlib.util.spec_from_file_location("balanced_centroid_encoder", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    destination = output / "centroids.lvq6.bin.gz"
    encoder.encode(source, destination, heads.shape[1], bits=6, metric="l2")
    return {"heads": len(heads), "dimensions": heads.shape[1], "centroid_quantization": "LVQ6",
            "centroid_metric": "l2", "centroid_index": destination.name,
            "centroid_gzip_bytes": destination.stat().st_size}


def export_centroids_brotli(output):
    """Prepare the one-file static index before spending time on full assignment."""
    import brotli

    output = Path(output)
    source = output / "centroids.lvq6.bin.gz"
    target = output / "centroids.lvq6.bin.br"
    temporary = target.with_name(target.name + ".tmp")
    for path in (source, target, temporary, output / "_headers", output / "centroids.bin.gz"):
        if path.is_symlink():
            raise ValueError("Centroid artifacts must not be symlinks")
    raw = gzip.decompress(source.read_bytes())
    if len(raw) < 16:
        raise ValueError("Truncated centroid index")
    magic, version, dim, count = struct.unpack_from("<8sHHI", raw)
    expected = 16 + 2 * dim + count * (4 + (dim * 6 + 7) // 8)
    if magic != b"LVQ6IDX\0" or version != 2 or not dim or not count or len(raw) != expected:
        raise ValueError("Expected a complete LVQ6 L2 centroid index")
    encoded = brotli.compress(raw, quality=11)
    limit = 25 * 1024 ** 2
    if len(encoded) > limit:
        raise ValueError(f"Brotli centroid index is {len(encoded):,} bytes, above the {limit:,}-byte "
                         "Pages limit. Trained heads are retained; assignment has not started.")
    if brotli.decompress(encoded) != raw:
        raise ValueError("Brotli centroid round-trip changed the index")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    (output / "_headers").write_text(
        "/*\n  Access-Control-Allow-Origin: *\n  Cross-Origin-Resource-Policy: cross-origin\n"
        "  Timing-Allow-Origin: *\n\n/centroids.lvq6.bin.br\n  Content-Encoding: br\n"
        "  Content-Type: application/octet-stream\n"
        "  Cache-Control: public, max-age=31536000, immutable, no-transform\n")
    (output / "centroids.bin.gz").unlink(missing_ok=True)
    return {"centroid_brotli_index": target.name, "centroid_brotli_bytes": len(encoded),
            "centroid_brotli_sha256": hashlib.sha256(encoded).hexdigest(),
            "centroid_brotli_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "centroid_brotli_encoding": "br"}


def export_postings(files, labels_path, heads, output, *, dim=384, batch_size=250000,
                    threads=8, heads_per_shard=512):
    files = [Path(path).resolve() for path in files]
    labels_path, output = Path(labels_path).resolve(), Path(output)
    if output.is_symlink():
        raise ValueError("Export output must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    record_bytes = record_dtype(dim).itemsize
    inputs = [snapshot(path) for path in files]
    count = sum(item["bytes"] // record_bytes for item in inputs)
    if (dim <= 0 or count <= 0 or heads <= 0 or threads <= 0 or batch_size <= 0
            or batch_size * record_bytes > 256 * 1024 ** 2
            or not 1 <= heads_per_shard <= 512 or len(set(files)) != len(files)
            or any(item["bytes"] % record_bytes for item in inputs)
            or labels_path.stat().st_size != count * 4):
        raise ValueError("Invalid original records, single assignments, or export memory bounds")
    shards = (heads + heads_per_shard - 1) // heads_per_shard
    descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if descriptor_limit != resource.RLIM_INFINITY and max(2 * shards, heads_per_shard) + 64 > descriptor_limit:
        raise OSError("Insufficient file descriptors for bounded posting export")
    manifest = {"inputs": inputs, "labels": artifact(labels_path), "heads": heads,
                "dimensions": dim, "heads_per_shard": heads_per_shard, "compression_level": 9}
    checkpoint = output / "export.json"
    spool, clusters = output / "_export_spool", output / "clusters"
    for path in (checkpoint, spool, clusters):
        if path.is_symlink():
            raise ValueError("Owned export paths must not be symlinks")
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["manifest"] != manifest:
            raise ValueError("Export inputs changed; refusing to mix datasets")
        counts = np.asarray(state.get("record_counts", []), np.int64)
    else:
        if (spool.exists() or (clusters.exists() and any(clusters.iterdir()))):
            raise ValueError("Refusing unrelated existing export files")
        state = {"manifest": manifest, "status": "spooling", "completed_shards": [], "source_rows": count}
        counts = np.zeros(heads, np.int64)
        write_json(checkpoint, state)

    def unchanged():
        if inputs != [snapshot(path) for path in files] or artifact(labels_path) != manifest["labels"]:
            raise ValueError("Original records or assignment map changed during export")

    def paths(shard):
        return spool / f"records_{shard}.bin", spool / f"labels_{shard}.i32"

    if state["status"] == "spooling":
        require_disk_space(output, count * (record_bytes + 4))
        shutil.rmtree(spool, ignore_errors=True)
        spool.mkdir()
        counts = np.zeros(heads, np.int64)
        offset = 0
        with ExitStack() as stack:
            records_out = [stack.enter_context(paths(shard)[0].open("xb")) for shard in range(shards)]
            labels_out = [stack.enter_context(paths(shard)[1].open("xb")) for shard in range(shards)]
            for batch in iter_record_batches(files, dim, batch_size):
                labels = np.memmap(labels_path, dtype="<i4", mode="r", offset=offset * 4, shape=(len(batch),))
                if (labels < 0).any() or (labels >= heads).any():
                    raise ValueError("Every source row must have exactly one valid centroid ID")
                counts += np.bincount(labels, minlength=heads)
                shard_ids = labels // heads_per_shard
                order = np.argsort(shard_ids, kind="stable")
                ordered = shard_ids[order]
                boundaries = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1, len(order)]
                packed = batch.view(f"V{record_bytes}")
                for first, last in zip(boundaries[:-1], boundaries[1:]):
                    shard = int(ordered[first])
                    rows = order[first:last]
                    records_out[shard].write(packed[rows].tobytes())
                    labels_out[shard].write(labels[rows].tobytes())
                offset += len(batch)
                del labels
            for stream in records_out + labels_out:
                stream.flush()
                os.fsync(stream.fileno())
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if offset != count:
            raise ValueError("Original source row count changed")
        unchanged()
        state.update(status="compressing", record_counts=counts.tolist())
        write_json(checkpoint, state)
        del batch, packed, order, ordered, shard_ids
    if counts.shape != (heads,) or int(counts.sum()) != count or (counts < 0).any():
        raise ValueError("Invalid saved posting counts")
    clusters.mkdir(exist_ok=True)
    require_disk_space(output, 0 if state["status"] == "complete" else count * record_bytes)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for shard in range(shards):
            first, last = shard * heads_per_shard, min((shard + 1) * heads_per_shard, heads)
            record_path, label_path = paths(shard)
            if shard not in state["completed_shards"]:
                expected = int(counts[first:last].sum())
                if record_path.stat().st_size != expected * record_bytes or label_path.stat().st_size != expected * 4:
                    raise ValueError("Truncated export spool shard")
                emitted = np.zeros(last - first, np.int64)
                with ExitStack() as stack:
                    streams = []
                    for cid in range(first, last):
                        target = stack.enter_context((clusters / f"cluster_{cid}.bin.gz.tmp").open("wb"))
                        streams.append(stack.enter_context(gzip.GzipFile(filename=f"cluster_{cid}.bin.gz",
                                                                       fileobj=target, mode="wb", compresslevel=9)))
                    offset = 0
                    for batch in iter_record_batches([record_path], dim, batch_size):
                        labels = np.memmap(label_path, dtype="<i4", mode="r", offset=offset * 4, shape=(len(batch),))
                        if (labels < first).any() or (labels >= last).any():
                            raise ValueError("Spool labels escaped their assigned shard")
                        order = np.argsort(labels, kind="stable")
                        ordered = labels[order]
                        boundaries = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1, len(order)]
                        packed = batch.view(f"V{record_bytes}")

                        def write_group(bounds):
                            left, right = bounds
                            local = int(ordered[left]) - first
                            rows = order[left:right]
                            streams[local].write(packed[rows].tobytes())
                            emitted[local] += len(rows)

                        list(pool.map(write_group, zip(boundaries[:-1], boundaries[1:])))
                        offset += len(batch)
                        del labels
                if offset != expected or not np.array_equal(emitted, counts[first:last]):
                    raise ValueError("Exported shard differs from its posting assignments")
                for cid in range(first, last):
                    (clusters / f"cluster_{cid}.bin.gz.tmp").replace(clusters / f"cluster_{cid}.bin.gz")
                state["completed_shards"].append(shard)
                write_json(checkpoint, state)
            for cid in range(first, last):
                path = clusters / f"cluster_{cid}.bin.gz"
                with path.open("rb") as compressed:
                    compressed.seek(-4, os.SEEK_END)
                    size = struct.unpack("<I", compressed.read(4))[0]
                if size != int(counts[cid]) * record_bytes % 2 ** 32:
                    raise ValueError("Exported gzip footer differs from its posting record count")
            record_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
    unchanged()
    state["status"] = "complete"
    write_json(checkpoint, state)
    if spool.exists():
        spool.rmdir()
    np.save(output / "posting_counts.npy", counts)
    sizes = np.array([(clusters / f"cluster_{cid}.bin.gz").stat().st_size for cid in range(heads)], dtype=np.int64)
    np.save(output / "posting_gzip_sizes.npy", sizes)
    return {"input_vectors": count, "heads": heads, "posting_records": count,
            "assignments_per_vector": 1, "export_mode": "bounded_shard_gzip", "posting_gzip_bytes": int(sizes.sum()),
            "nonempty_clusters": int(np.count_nonzero(counts)), "cluster_size_min": int(counts.min()),
            "cluster_size_median": float(np.median(counts)), "cluster_size_p95": float(np.percentile(counts, 95)),
            "cluster_size_p99": float(np.percentile(counts, 99)), "cluster_size_max": int(counts.max())}
