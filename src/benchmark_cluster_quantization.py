"""Compare 20 representative posting codecs, or the full 114-configuration grid."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUPY_TF32", "0")
import numpy as np

from balanced_l2.benchmark_ground_truth import exact_gpu, query_sets
from balanced_l2.common import artifact, check_artifact, record_dtype, require_disk_space, snapshot, source_path, write_json
from balanced_l2.export import decode_centroids
from balanced_l2 import posting_external_codecs as external
from balanced_l2 import posting_product_codecs as product
from balanced_l2.posting_quantization import decode_fp16, decode_posting, encode_posting
from balanced_l2.quantized_routing import route_queries


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = source_path(ROOT, "client/data")
NOISE = .025
PROBES = (1, 2, 4, 8, 16, 32)
DEFAULT_METHODS = (
    "fp16",
    "lvq4", "lvq5", "lvq6", "lvq8",
    "sq4", "sq6", "sq8", "sq6_quantile0.005",
    "pq32x4", "pq64x6", "pq96x8", "pq192x8",
    "opq64x8", "rq4x6", "lsq4x6",
    "gguf_q4_0", "gguf_q4_1", "gguf_q5_0",
    "fp8_e4m3_row",
)
IDENTITY = np.dtype([("code", "<i4"), ("page", "<i2")])
ENVELOPE = struct.Struct("<8sHHI")
MAGIC = b"POSTCODE"


def all_methods():
    methods = ("fp16", "bf16", *(f"lvq{bits}" for bits in range(2, 9)),
               *(method for method in product.METHODS if method != "bf16"), *external.METHODS)
    assert len(methods) == len(set(methods)) == 114
    return methods


def family(method):
    return next((name for name in ("lvq", "sq", "pq", "opq", "rq", "lsq", "gguf", "fp8")
                 if method.startswith(name)), "reference")


class NormalizedVectors:
    """Expose normalized batches without a second, full float32 corpus on disk."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, rows):
        values = self.records["vec"][rows].astype(np.float32)
        norms = np.linalg.norm(values, axis=-1)
        if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Source vectors must be finite and nonzero")
        return values / norms[..., None]


def _save_npz(path, **arrays):
    if path.is_symlink():
        raise ValueError(f"Owned artifact must not be a symlink: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _save_results(output, ledger):
    write_json(output / "results.json", ledger)
    if ledger["points"]:
        temporary = output / "results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ledger["points"][0]))
            writer.writeheader()
            writer.writerows(ledger["points"])
        temporary.replace(output / "results.csv")


def _cleanup_staging(output, ledger):
    work, path = output / "_staging", output / "_staging/records.bin"
    if work.is_symlink() or path.is_symlink():
        raise ValueError("Owned staging must not be a symlink")
    if path.exists():
        if ledger["staging"]["path"] != str(path.resolve()):
            raise ValueError("Staging checkpoint belongs to another output directory")
        path.unlink()
    if work.exists():
        work.rmdir()
    ledger["staging_removed"] = True
    _save_results(output, ledger)


def _source(directory):
    index_path = directory / "centroids.lvq6.bin.gz"
    payload = gzip.decompress(index_path.read_bytes())
    if len(payload) < 16:
        raise ValueError("Truncated centroid index")
    magic, version, dimensions, heads = struct.unpack("<8sHHI", payload[:16])
    if magic != b"LVQ6IDX\0" or version != 2:
        raise ValueError("This comparison requires the existing LVQ6 L2 routing index")
    decoded = decode_centroids(payload, heads, dimensions)
    paths = [directory / "clusters" / f"cluster_{cid}.bin.gz" for cid in range(heads)]
    if len(list((directory / "clusters").glob("cluster_*.bin.gz"))) != heads:
        raise ValueError("Posting IDs must cover precisely the centroid index")
    counts = []
    for path in paths:
        with path.open("rb") as stream:
            stream.seek(-4, os.SEEK_END)
            size = struct.unpack("<I", stream.read(4))[0]
        if size % record_dtype(dimensions).itemsize:
            raise ValueError(f"Incomplete source records: {path}")
        counts.append(size // record_dtype(dimensions).itemsize)
    if not sum(counts):
        raise ValueError("Empty source corpus")
    return paths, np.asarray(counts, np.int64), decoded, artifact(index_path)


def _stage(output, ledger, paths, counts, dimensions):
    work = output / "_staging"
    if work.is_symlink():
        raise ValueError("Staging must not be a symlink")
    work.mkdir(exist_ok=True)
    path = work / "records.bin"
    expected_bytes = int(counts.sum()) * record_dtype(dimensions).itemsize
    if "staging" in ledger:
        if ledger["staging"]["path"] != str(path.resolve()):
            raise ValueError("Staging checkpoint belongs to another output directory")
        check_artifact(ledger["staging"])
    else:
        require_disk_space(output, expected_bytes)
        temporary = work / "records.bin.tmp"
        digest = hashlib.sha256()
        with temporary.open("wb") as target:
            for cid, source in enumerate(paths):
                raw = gzip.decompress(source.read_bytes())
                _, _, vectors = decode_fp16(raw, dimensions)
                if len(vectors) != counts[cid]:
                    raise ValueError(f"Source gzip count differs: {source}")
                target.write(raw)
                digest.update(raw)
                if cid % 256 == 0 or cid + 1 == len(paths):
                    print(f"Staging source postings {cid + 1}/{len(paths)}", flush=True)
        if temporary.stat().st_size != expected_bytes:
            raise ValueError("Incomplete corpus staging")
        if ledger["manifest"]["inputs"] != [snapshot(source) for source in paths]:
            raise ValueError("Source postings changed during staging")
        temporary.replace(path)
        ledger["staging"] = {**snapshot(path), "sha256": digest.hexdigest()}
        _save_results(output, ledger)
    if path.stat().st_size != expected_bytes:
        raise ValueError("Saved corpus size differs")
    return np.memmap(path, dtype=record_dtype(dimensions), mode="r", shape=(int(counts.sum()),))


def _truth_and_routes(output, ledger, records, decoded, query_count, probes, gpu):
    vectors = NormalizedVectors(records)
    sampled, sets = query_sets(vectors, query_count, (NOISE,))
    queries = sets[NOISE]
    path = output / "truth.npz"
    if "truth" in ledger:
        check_artifact(ledger["truth"])
        with np.load(path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["sampled"], sampled)
                    or not np.array_equal(saved["queries"].view(np.uint32), queries.view(np.uint32))):
                raise ValueError("Saved query replay differs")
            truth, values = saved["truth"], saved["values"]
    else:
        print("Computing exhaustive full-corpus truth at noise 0.025", flush=True)
        started = time.perf_counter()
        if gpu:
            truth, values = exact_gpu(vectors, queries, progress=True)
            truth = truth[:, 0]
        else:
            truth, values = _cpu_truth(vectors, queries)
        _save_npz(path, sampled=sampled, queries=queries, truth=truth, values=values)
        ledger["truth_seconds"] = time.perf_counter() - started
        ledger["truth"] = artifact(path)
        _save_results(output, ledger)
    if (truth.shape != (query_count,) or truth.dtype.kind not in "iu"
            or (truth < 0).any() or (truth >= len(records)).any()):
        raise ValueError("Invalid truth rows")
    route_path = output / "routes.npz"
    if "routes" in ledger:
        check_artifact(ledger["routes"])
        with np.load(route_path, allow_pickle=False) as saved:
            routes = saved["routes"]
    else:
        started = time.perf_counter()
        routes = route_queries(decoded, queries, probes=max(probes), gpu=gpu, metric="l2")
        _save_npz(route_path, routes=routes)
        ledger["routing_seconds"] = time.perf_counter() - started
        ledger["routes"] = artifact(route_path)
        _save_results(output, ledger)
    if (routes.shape != (query_count, max(probes)) or (routes < 0).any()
            or (routes >= len(decoded)).any()
            or (np.diff(np.sort(routes, axis=1), axis=1) == 0).any()):
        raise ValueError("Invalid saved routes")
    return queries, truth, values, routes


def _cpu_truth(vectors, queries):
    """Bounded CPU fallback for small integration fixtures."""
    best = np.full(len(queries), -np.inf, np.float32)
    rows = np.full(len(queries), -1, np.int64)
    for first in range(0, len(vectors), 8192):
        block = vectors[first:first + 8192]
        for start in range(0, len(queries), 256):
            stop = min(start + 256, len(queries))
            scores = queries[start:stop] @ block.T
            local = np.argmax(scores, axis=1)
            values = scores[np.arange(stop - start), local]
            better = values > best[start:stop]
            best[start:stop] = np.where(better, values, best[start:stop])
            rows[start:stop] = np.where(better, first + local, rows[start:stop])
    return rows, best


def _encode(raw, method, dimensions, model):
    if method == "fp16":
        return raw
    if method.startswith("lvq"):
        return encode_posting(raw, dimensions, int(method[3:]))
    codes, pages, vectors = decode_fp16(raw, dimensions)
    ids = np.empty(len(codes), IDENTITY)
    ids["code"], ids["page"] = codes, pages
    if model is None:
        body = external.encode(vectors, method)
    else:
        body = b"".join(model.encode(vectors[first:first + 4096]).tobytes()
                        for first in range(0, len(vectors), 4096))
    return ENVELOPE.pack(MAGIC, 1, dimensions, len(codes)) + ids.tobytes() + body


def _decode(payload, method, dimensions, model):
    if method == "fp16":
        return decode_fp16(payload, dimensions)
    if method.startswith("lvq"):
        return decode_posting(payload, dimensions)
    if len(payload) < ENVELOPE.size:
        raise ValueError("Truncated posting envelope")
    magic, version, dim, count = ENVELOPE.unpack_from(payload)
    offset = ENVELOPE.size + count * IDENTITY.itemsize
    if magic != MAGIC or version != 1 or dim != dimensions or len(payload) < offset:
        raise ValueError("Invalid posting envelope")
    ids = np.frombuffer(payload, IDENTITY, count=count, offset=ENVELOPE.size)
    body = payload[offset:]
    if model is None:
        vectors = external.decode(body)
    else:
        if len(body) != count * model.code_size:
            raise ValueError("Encoded vector count differs")
        encoded = np.frombuffer(body, np.uint8).reshape(count, model.code_size)
        vectors = np.empty((count, dimensions), np.float32)
        for first in range(0, count, 4096):
            vectors[first:first + 4096] = model.decode(encoded[first:first + 4096])
    if vectors.shape != (count, dimensions) or not np.isfinite(vectors).all():
        raise ValueError("Invalid reconstructed vectors")
    return ids["code"], ids["page"], vectors


def prefix_winners(rows, scores, probes):
    best_rows = np.full(len(rows), -1, np.int64)
    best_scores = np.full(len(rows), -np.inf, np.float64)
    selected_rows, selected_scores = [], []
    for rank in range(rows.shape[1]):
        better = scores[:, rank] > best_scores
        best_rows[better], best_scores[better] = rows[better, rank], scores[better, rank]
        if rank + 1 in probes:
            selected_rows.append(best_rows.copy())
            selected_scores.append(best_scores.copy())
    return np.stack(selected_rows, axis=1), np.stack(selected_scores, axis=1)


def _checkpoint(case, done, rows, scores, sizes, timings):
    marker = case / "progress.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    path = case / f"checkpoint_{done}.npz"
    _save_npz(path, rows=rows, scores=scores, sizes=sizes)
    write_json(marker, {"done": done, "arrays": artifact(path), "timings": timings})
    if previous and previous["arrays"]["path"] != str(path.resolve()):
        _checkpoint_path(case, previous).unlink(missing_ok=True)


def _checkpoint_path(case, progress):
    path = Path(progress["arrays"]["path"])
    if path.parent != case.resolve() or path.name != f"checkpoint_{progress['done']}.npz":
        raise ValueError("Posting checkpoint belongs to another case directory")
    return path


def _case(output, method, records, counts, offsets, queries, routes, order, starts, sample, gpu):
    from balanced_l2.posting_scoring import best_matches

    case = output / "cases" / method
    if case.is_symlink():
        raise ValueError("Case directory must not be a symlink")
    case.mkdir(parents=True, exist_ok=True)
    model = None
    model_bytes = 0
    train_seconds = 0.
    if method in product.METHODS:
        model_path, model_marker = case / "model.bin.gz", case / "model.json"
        if model_marker.exists():
            saved = json.loads(model_marker.read_text())
            check_artifact(saved["model"])
            model = product.load(gzip.decompress(model_path.read_bytes()))
            train_seconds = saved["train_seconds"]
        else:
            started = time.perf_counter()
            model = product.train(method, sample)
            encoded = gzip.compress(model.serialize(), compresslevel=9, mtime=0)
            temporary = model_path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            temporary.replace(model_path)
            model = product.load(gzip.decompress(model_path.read_bytes()))
            train_seconds = time.perf_counter() - started
            write_json(model_marker, {"model": artifact(model_path), "train_seconds": train_seconds})
        if model.name != method or model.dimensions != records.dtype["vec"].shape[0]:
            raise ValueError("Saved codec model differs")
        model_bytes = model_path.stat().st_size
    dimensions = records.dtype["vec"].shape[0]
    source_directory = Path(json.loads((output / "results.json").read_text())["manifest"]["source"])
    marker = case / "progress.json"
    if marker.exists():
        progress = json.loads(marker.read_text())
        _checkpoint_path(case, progress)
        check_artifact(progress["arrays"])
        with np.load(progress["arrays"]["path"], allow_pickle=False) as saved:
            rows, scores, sizes = saved["rows"], saved["scores"], saved["sizes"]
        done, timings = progress["done"], progress["timings"]
        if rows.shape != routes.shape or scores.shape != routes.shape or sizes.shape != counts.shape:
            raise ValueError("Checkpoint array shapes differ")
    else:
        rows = np.full(routes.shape, -1, np.int64)
        scores = np.full(routes.shape, -np.inf, np.float64)
        sizes = np.zeros(len(counts), np.int64)
        done = 0
        timings = {"encode_seconds": 0., "decode_seconds": 0., "score_seconds": 0.}
    for cid in range(done, len(counts)):
        source = records[offsets[cid]:offsets[cid + 1]]
        raw = source.tobytes()
        started = time.perf_counter()
        if method == "fp16":
            compressed = (source_directory / "clusters" / f"cluster_{cid}.bin.gz").read_bytes()
        else:
            compressed = gzip.compress(_encode(raw, method, dimensions, model), compresslevel=9, mtime=0)
        sizes[cid] = len(compressed)
        timings["encode_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        codes, pages, vectors = _decode(gzip.decompress(compressed), method, dimensions, model)
        if not np.array_equal(codes, source["code"]) or not np.array_equal(pages, source["page"]):
            raise ValueError("Posting IDs/pages changed during codec round trip")
        timings["decode_seconds"] += time.perf_counter() - started
        positions = order[starts[cid]:starts[cid + 1]]
        if len(positions) and len(source):
            query_ids, ranks = positions // routes.shape[1], positions % routes.shape[1]
            started = time.perf_counter()
            local, similarities = best_matches(queries[query_ids], vectors, gpu=gpu)
            rows[query_ids, ranks] = offsets[cid] + local
            scores[query_ids, ranks] = similarities
            timings["score_seconds"] += time.perf_counter() - started
        if (cid + 1) % 512 == 0 or cid + 1 == len(counts):
            _checkpoint(case, cid + 1, rows, scores, sizes, timings)
            print(f"{method}: postings {cid + 1}/{len(counts)}", flush=True)
    return rows, scores, sizes, model_bytes, train_seconds, timings


def _same_images(records, rows, expected):
    valid = (rows >= 0) & (expected >= 0)
    result = np.zeros(rows.shape, bool)
    result[valid] = ((records["code"][rows[valid]] == records["code"][expected[valid]])
                     & (records["page"][rows[valid]] == records["page"][expected[valid]]))
    return result


def _measure(method, records, truth, queries, routes, offsets, sizes, winners, similarities,
             baseline, probes, model_bytes, centroid_bytes, train_seconds, timings):
    vectors = NormalizedVectors(records)
    true_vectors = vectors[truth].astype(np.float64)
    true_vectors /= np.linalg.norm(true_vectors, axis=1)[:, None]
    true_scores = np.einsum("ij,ij->i", true_vectors, queries.astype(np.float64))
    true_clusters = np.searchsorted(offsets[1:], truth, side="right")
    points = []
    for column, probe in enumerate(probes):
        rows = winners[:, column]
        valid = rows >= 0
        hits = _same_images(records, rows, truth)
        reference = baseline[:, column] if baseline is not None else rows
        agreement = _same_images(records, rows, reference)
        source_scores = np.full(len(queries), -1., np.float64)
        selected = vectors[rows[valid]].astype(np.float64)
        selected /= np.linalg.norm(selected, axis=1)[:, None]
        source_scores[valid] = np.einsum("ij,ij->i", selected, queries[valid].astype(np.float64))
        download = sizes[routes[:, :probe]].sum(axis=1)
        mean_mb = float(download.mean() / 1e6)
        points.append({"method": method, "family": family(method), "queries": len(queries),
                       "noise_std": NOISE, "probes": probe,
                       "recall_percent": float(hits.mean() * 100),
                       "recall_vs_fp16_percent": float(agreement.mean() * 100),
                       "tie_aware_recall_percent": float((source_scores >= true_scores - 2e-6).mean() * 100),
                       "routing_coverage_percent": float((routes[:, :probe] == true_clusters[:, None]).any(axis=1).mean() * 100),
                       "mean_original_cosine_loss": float(np.maximum(0, true_scores - source_scores).mean()),
                       "mean_score_error": float(np.abs(similarities[valid, column] - source_scores[valid]).mean()) if valid.any() else 0.,
                       "posting_gzip_MB_mean": mean_mb,
                       "posting_gzip_MB_p95": float(np.percentile(download, 95) / 1e6),
                       "model_gzip_MB": model_bytes / 1e6, "centroid_gzip_MB": centroid_bytes / 1e6,
                       "cold_gzip_MB_mean": mean_mb + model_bytes / 1e6 + centroid_bytes / 1e6,
                       "total_posting_gzip_GB": int(sizes.sum()) / 1e9,
                       "train_seconds": train_seconds, **timings})
    return points


def run(source=DEFAULT_SOURCE, output="cluster_quantization_8k", *, query_count=100000,
        probes=PROBES, methods=None, train_size=65536, gpu=True):
    source, output = Path(source).resolve(), Path(output)
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Benchmark output must be separate from its source dataset")
    paths, counts, decoded, centroid = _source(source)
    probes = tuple(probes)
    methods = tuple(dict.fromkeys(("fp16", *(methods or DEFAULT_METHODS))))
    if (not isinstance(query_count, int) or not 0 < query_count <= int(counts.sum())
            or not isinstance(train_size, int) or train_size <= 0
            or not probes or tuple(sorted(set(probes))) != probes
            or any(not isinstance(p, int) or not 1 <= p <= len(counts) for p in probes)
            or any(method not in all_methods() for method in methods)):
        raise ValueError("Invalid query count, training size, probe grid, or codec list")
    producers = ("benchmark_cluster_quantization.py", "balanced_l2/common.py", "balanced_l2/posting_quantization.py",
                 "balanced_l2/posting_external_codecs.py", "balanced_l2/posting_product_codecs.py",
                 "balanced_l2/posting_scoring.py", "balanced_l2/quantized_routing.py",
                 "balanced_l2/benchmark_ground_truth.py", "client/search.js")
    manifest = {"schema_version": 1, "source": str(source), "inputs": [snapshot(path) for path in paths],
                "centroids": centroid, "head_count": len(counts), "dimensions": decoded.shape[1],
                "source_rows": int(counts.sum()), "query_count": query_count, "noise_std": NOISE,
                "probes": list(probes), "methods": list(methods), "training_sample_size": min(train_size, int(counts.sum())),
                "seed": 42, "gpu": gpu, "gzip_level": 9,
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "faiss-cpu", "torch", "gguf", "cupy-cuda12x")},
                "producers": {name: artifact(source_path(ROOT, name))["sha256"] for name in producers},
                "truth_definition": "Exhaustive float32 dot over unit-normalized original FP16 vectors; strict image-ID recall and 2e-6 tie-aware cosine recall reported",
                "scoring_definition": "Reconstructed Float32 vectors, browser-style cosine scoring, stable posting/centroid ties, no similarity threshold",
                "size_definition": "Actual gzip9 posting bytes including IDs, headers, local scales/means; shared global models counted once in cold query bytes",
                "limits": "Noisy replay, not real-image embeddings; timings are offline codec/compute measurements, not browser/network latency"}
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    if results_path.exists():
        ledger = json.loads(results_path.read_text())
        if ledger["manifest"] != manifest:
            raise ValueError("Sources, settings, package versions, or benchmark code changed; use a fresh --output")
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing an unrelated nonempty output directory")
        ledger = {"manifest": manifest, "status": "running", "points": [], "cases": {}}
        _save_results(output, ledger)
    if ledger["status"] == "complete":
        for name in ("truth", "routes"):
            check_artifact(ledger[name])
        for case in ledger["cases"].values():
            for value in case.get("artifacts", []):
                check_artifact(value)
        from balanced_l2.plot_posting_quantization import render
        render(output)
        _cleanup_staging(output, ledger)
        return ledger
    records = _stage(output, ledger, paths, counts, decoded.shape[1])
    queries, truth, _, routes = _truth_and_routes(output, ledger, records, decoded, query_count, probes, gpu)
    sampled = np.sort(np.random.default_rng(42).choice(len(records), manifest["training_sample_size"], replace=False))
    sample = records["vec"][sampled].astype(np.float32)
    offsets = np.r_[0, np.cumsum(counts)]
    order = np.argsort(routes.reshape(-1), kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(routes.reshape(-1), minlength=len(counts)))]
    baseline = None
    try:
        for index, method in enumerate(methods):
            if manifest["inputs"] != [snapshot(path) for path in paths]:
                raise ValueError("Source postings changed during comparison")
            print(f"Codec {index + 1}/{len(methods)}: {method}", flush=True)
            previous = ledger["cases"].get(method)
            if previous:
                for value in previous.get("artifacts", []):
                    check_artifact(value)
                if method == "fp16":
                    with np.load(output / "cases/fp16/winners.npz", allow_pickle=False) as saved:
                        baseline = saved["rows"]
                continue
            try:
                rows, scores, sizes, model_bytes, seconds, timings = _case(
                    output, method, records, counts, offsets, queries, routes, order, starts, sample, gpu)
                winners, winner_scores = prefix_winners(rows, scores, probes)
                points = _measure(method, records, truth, queries, routes, offsets, sizes, winners,
                                  winner_scores, baseline, probes, model_bytes, centroid["bytes"], seconds, timings)
                case = output / "cases" / method
                destination = case / "winners.npz"
                _save_npz(destination, rows=winners, scores=winner_scores, gzip_sizes=sizes)
                artifacts = [artifact(destination)]
                if (case / "model.json").exists():
                    artifacts.append(json.loads((case / "model.json").read_text())["model"])
                ledger["cases"][method] = {"status": "complete", "artifacts": artifacts}
                ledger["points"].extend(points)
                if method == "fp16":
                    baseline = winners
            except (ValueError, FloatingPointError, RuntimeError) as error:
                if method == "fp16" or any(word in str(error).lower() for word in ("out of memory", "bad_alloc")):
                    raise
                ledger["cases"][method] = {"status": "rejected", "reason": str(error)}
                print(f"Rejected {method}: {error}", flush=True)
            _save_results(output, ledger)
            case = output / "cases" / method
            marker = case / "progress.json"
            if marker.exists():
                _checkpoint_path(case, json.loads(marker.read_text())).unlink(missing_ok=True)
                marker.unlink()
        if manifest["inputs"] != [snapshot(path) for path in paths] or artifact(source / "centroids.lvq6.bin.gz") != centroid:
            raise ValueError("Source dataset changed during comparison")
        ledger["status"] = "complete"
        ledger.pop("last_error", None)
        _save_results(output, ledger)
        from balanced_l2.plot_posting_quantization import render
        render(output)
        del records
        _cleanup_staging(output, ledger)
        return ledger
    except BaseException as error:
        ledger["status"] = "running"
        ledger["last_error"] = str(error)
        _save_results(output, ledger)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "cluster_quantization_8k")
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--probes", type=int, nargs="+", default=list(PROBES))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--methods", nargs="+", help="Custom codec list; FP16 is always included")
    selection.add_argument("--all-methods", action="store_true", help="Run all 114 configurations instead of the default 20")
    parser.add_argument("--train-size", type=int, default=65536)
    parser.add_argument("--cpu", action="store_true", help="CPU fallback for small smoke fixtures")
    parser.add_argument("--list-methods", action="store_true", help="List the selected codecs without running")
    args = parser.parse_args()
    methods = all_methods() if args.all_methods else args.methods
    if args.list_methods:
        print("\n".join(dict.fromkeys(("fp16", *(methods or DEFAULT_METHODS)))))
    else:
        result = run(args.source, args.output, query_count=args.queries, probes=args.probes,
                     methods=methods, train_size=args.train_size, gpu=not args.cpu)
        print(f"Comparison {result['status']}: {args.output / 'results.json'}")
