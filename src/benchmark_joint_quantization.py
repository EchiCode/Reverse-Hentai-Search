"""Sequential cluster-count × centroid-codec × posting-codec comparison."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUPY_TF32", "0")
import numpy as np

from balanced_l2 import build as pipeline
from balanced_l2.benchmark_ground_truth import exact_gpu, query_sets
from balanced_l2.centroid_quantization import METHODS, decode, encode
from balanced_l2.common import (artifact, check_artifact, input_files,
                                matrix_shape, read_matrix, record_dtype,
                                require_disk_space, sha256, snapshot, source_path, write_json)
from balanced_l2.posting_quantization import decode_fp16, decode_posting, encode_posting
from balanced_l2.posting_scoring import best_matches
from balanced_l2.quantized_routing import route_queries
from balanced_l2.source_records import SourceRecords
from benchmark_cluster_quantization import NormalizedVectors, _cpu_truth, _same_images, prefix_winners


ROOT = Path(__file__).resolve().parent
HEADS = (64000, 80000, 96000, 128000)
PROBES = (4, 8, 16, 32, 64)
NOISE = .025
CODECS = ("fp16", "lvq8", "lvq6", "lvq5")


def _release_gpu():
    if "cupy" in sys.modules:
        cp = sys.modules["cupy"]
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _guard(path):
    if Path(path).is_symlink():
        raise ValueError(f"Owned benchmark path must not be a symlink: {path}")


def _npz(path, **values):
    temporary = path.with_name(path.name + ".tmp")
    _guard(path)
    _guard(temporary)
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(path)


def _check(value, root):
    if Path(value["path"]).resolve().parent != root.resolve():
        raise ValueError("Saved benchmark artifact belongs to another directory")
    _guard(Path(value["path"]))
    check_artifact(value)


def _save(output, ledger):
    _guard(output / "results.json")
    write_json(output / "results.json", ledger)
    if ledger["points"]:
        temporary = output / "results.csv.tmp"
        _guard(temporary)
        _guard(output / "results.csv")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ledger["points"][0]))
            writer.writeheader()
            writer.writerows(ledger["points"])
        temporary.replace(output / "results.csv")


def _unchanged(files, manifest):
    if [snapshot(path) for path in files] != manifest["inputs"]:
        raise ValueError("Original source vectors changed during comparison")


def _scratch_bytes(config):
    """Conservative disk bound for the sample and one count's atomic checkpoints."""
    rows, dim = config["source_rows"], config["dimensions"]
    heads, queries = max(config["head_targets"]), config["query_count"]
    codecs, probes = len(CODECS), max(config["probes"])
    sample = min(rows, config["sample_size"]) * dim * 4 + 8
    native_and_indexes = rows * 4 + heads * dim * 16
    truth_and_routes = queries * (dim * 4 + 16 + codecs * probes * 4)
    progress = 2 * (codecs * queries * probes * 12 + heads * 8)
    winners = codecs ** 2 * (queries * len(config["probes"]) * 16 + heads * 8)
    # zlib's compressBound formula plus a gzip header/footer for each saved posting.
    raw = rows * record_dtype(dim).itemsize
    postings = raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + heads * 32
    return sample + native_and_indexes + truth_and_routes + progress + winners + postings + 64 * 1024 ** 2


def _stage(output, ledger, files, records):
    config = ledger["manifest"]
    work = output / "_staging"
    _guard(work)
    work.mkdir(exist_ok=True)
    paths = {"sample.f32": work / "sample.f32"}
    if {path.name for path in work.iterdir()} - {"sample.f32", "sample.f32.tmp"}:
        raise ValueError("Refusing unrelated files in shared sample staging")
    _unchanged(files, config)
    if ledger.get("staging"):
        for value in ledger["staging"].values():
            _check(value, work)
        temporary = paths["sample.f32"].with_name("sample.f32.tmp")
        _guard(temporary)
        temporary.unlink(missing_ok=True)
        return paths
    rows, dim = config["source_rows"], config["dimensions"]
    require_disk_space(output, _scratch_bytes(config))
    for path in paths.values():
        _guard(path)
        _guard(path.with_name(path.name + ".tmp"))
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)
    selected = np.sort(np.random.default_rng(42).choice(rows, min(rows, config["sample_size"]), replace=False))
    print("Preparing the shared training sample directly from original vector files", flush=True)
    sample = paths["sample.f32"].with_name("sample.f32.tmp")
    vectors = NormalizedVectors(records)
    with sample.open("xb") as stream:
        stream.write(struct.pack("<ii", len(selected), dim))
        for first in range(0, len(selected), 65536):
            vectors[selected[first:first + 65536]].astype("<f4", copy=False).tofile(stream)
    _unchanged(files, config)
    sample.replace(paths["sample.f32"])
    ledger["staging"] = {name: artifact(path) for name, path in paths.items()}
    _save(output, ledger)
    return paths


def _truth(output, ledger, vectors, files, gpu):
    config = ledger["manifest"]
    rows = len(vectors)
    _unchanged(files, config)
    sampled, sets = query_sets(vectors, config["query_count"], (config["noise_std"],))
    queries = sets[config["noise_std"]]
    destination = output / "truth.npz"
    if ledger.get("truth"):
        _check(ledger["truth"], output)
        with np.load(destination, allow_pickle=False) as saved:
            if (not np.array_equal(sampled, saved["sampled"])
                    or not np.array_equal(queries.view(np.uint32), saved["queries"].view(np.uint32))):
                raise ValueError("Shared truth differs from the seeded queries")
            truth = saved["truth"]
    else:
        print("Computing shared exhaustive full-corpus truth", flush=True)
        started = time.perf_counter()
        truth = exact_gpu(vectors, queries, progress=True)[0][:, 0] if gpu else _cpu_truth(vectors, queries)[0]
        _unchanged(files, config)
        _npz(destination, sampled=sampled, queries=queries, truth=truth)
        ledger["truth"] = artifact(destination)
        ledger["truth_seconds"] = time.perf_counter() - started
        _save(output, ledger)
    if (truth.shape != (len(queries),) or truth.dtype.kind not in "iu"
            or (truth < 0).any() or (truth >= rows).any()):
        raise ValueError("Invalid exhaustive truth rows")
    _unchanged(files, config)
    return queries, truth


def _train(sample, native, heads, gpu, iterations):
    if gpu:
        _release_gpu()
        native.mkdir(exist_ok=True)
        with (native / "training.log").open("a") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--train-worker",
                                     str(sample), str(native), str(heads), str(iterations)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Balanced L2 training failed; see {native / 'training.log'}")
        return
    # shortcut: CPU training is a tiny-fixture k-means smoke path; full runs use cuVS balanced L2.
    import faiss
    values = read_matrix(sample)
    faiss.omp_set_num_threads(1)
    model = faiss.Kmeans(values.shape[1], heads, niter=iterations, seed=42, min_points_per_centroid=1)
    model.train(values)
    path = native / "heads.f32"
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", heads, values.shape[1]))
        model.centroids.astype("<f4").tofile(stream)
    write_json(native / "training.json", {"metric": "l2", "library": "faiss_cpu_fixture_only",
               "sample": artifact(sample), "heads": artifact(path),
               "params": {"n_clusters": heads, "hierarchical_n_iters": iterations}})


def _case_directory(output, target, config_hash):
    parent = output / "_cases"
    _guard(parent)
    parent.mkdir(exist_ok=True)
    case = parent / f"h{target}"
    _guard(case)
    marker = case / "owned.json"
    expected = {"output": str(output), "head_count": target, "manifest_sha256": config_hash}
    if case.exists():
        _guard(marker)
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise ValueError("Benchmark case belongs to another run")
    else:
        case.mkdir()
        write_json(marker, expected)
    return case


def _clean_case(output, target, config_hash):
    parent, case = output / "_cases", output / "_cases" / f"h{target}"
    _guard(parent)
    _guard(case)
    if case.exists():
        _case_directory(output, target, config_hash)
        shutil.rmtree(case)


def _model(output, ledger, saved, case, target, paths, files, vectors):
    config = ledger["manifest"]
    native = case / "native"
    _guard(native)
    native.mkdir(exist_ok=True)
    marker = native / "training.json"
    if not marker.exists():
        print(f"Training {target:,} balanced L2 clusters", flush=True)
        _train(paths["sample.f32"], native, target, config["gpu"], config["max_iter"])
    _guard(marker)
    training = json.loads(marker.read_text())
    if (training["metric"] != "l2" or training["params"]["n_clusters"] != target
            or training["params"]["hierarchical_n_iters"] != config["max_iter"]
            or training["sample"]["sha256"] != ledger["staging"]["sample.f32"]["sha256"]):
        raise ValueError("Training differs from the shared sample or settings")
    _check(training["heads"], native)
    if matrix_shape(native / "heads.f32") != (target, config["dimensions"]):
        raise ValueError("Trained centroid count differs from the requested count")
    labels_path = native / "assignments.i32"
    assignment_marker = native / "assignment.json"
    _guard(labels_path)
    _guard(assignment_marker)
    source_identity = {"inputs": config["inputs"], "dimensions": config["dimensions"],
                       "normalization": "float32_l2"}
    if assignment_marker.exists():
        assignment = json.loads(assignment_marker.read_text())
        if (assignment["source"] != source_identity
                or assignment["heads"] != training["heads"]):
            raise ValueError("Saved assignment inputs differ")
        _check(assignment["labels"], native)
    else:
        labels_path.unlink(missing_ok=True)
        _guard(labels_path.with_name(labels_path.name + ".tmp"))
        labels_path.with_name(labels_path.name + ".tmp").unlink(missing_ok=True)
        print(f"Assigning the full corpus to {target:,} learned centroids", flush=True)
        metadata = pipeline.assign_vectors(vectors, native / "heads.f32", labels_path, gpu=config["gpu"],
                                           validate_inputs=lambda: _unchanged(files, config))
        _release_gpu()
        write_json(assignment_marker, {"source": source_identity, "heads": training["heads"],
                                      "labels": artifact(labels_path), "assignment": metadata})
    if labels_path.stat().st_size != config["source_rows"] * 4:
        raise ValueError("Assignments do not cover every source row")
    labels = np.memmap(labels_path, "<i4", "r", shape=(config["source_rows"],))
    if (labels < 0).any() or (labels >= target).any():
        raise ValueError("Invalid single centroid assignment")
    counts = np.bincount(labels, minlength=target)
    saved["provenance"] = {"heads_sha256": training["heads"]["sha256"], "labels_sha256": sha256(labels_path),
                           "training_library": training.get("library", "cuvs"),
                           "sample_sha256": training["sample"]["sha256"]}
    _save(output, ledger)
    return read_matrix(native / "heads.f32"), labels, counts


def _routes(output, ledger, saved, case, centroids, queries):
    methods, probes = ledger["manifest"]["centroid_methods"], ledger["manifest"]["probes"]
    states = saved.setdefault("routing", {})
    routes, sizes = {}, {}
    for method in methods:
        destination = case / f"centroids_{method}.bin.gz"
        route_path = case / f"routes_{method}.npz"
        if method in states:
            _check(states[method]["centroids"], case)
            _check(states[method]["routes"], case)
            with np.load(route_path, allow_pickle=False) as loaded:
                routed = loaded["routes"]
        else:
            print(f"Routing queries with {method} centroids", flush=True)
            payload = encode(centroids, method)
            decoded = decode(payload, method, len(centroids), centroids.shape[1])
            _guard(destination)
            temporary = destination.with_name(destination.name + ".tmp")
            _guard(temporary)
            temporary.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
            temporary.replace(destination)
            routed = route_queries(decoded, queries, probes=max(probes), gpu=ledger["manifest"]["gpu"], metric="l2").astype(np.int32)
            _npz(route_path, routes=routed)
            states[method] = {"centroids": artifact(destination), "routes": artifact(route_path)}
            _save(output, ledger)
        if (routed.shape != (len(queries), max(probes)) or routed.dtype.kind not in "iu"
                or (routed < 0).any() or (routed >= len(centroids)).any()
                or (np.diff(np.sort(routed, axis=1), axis=1) == 0).any()):
            raise ValueError("Invalid quantized centroid routes")
        routes[method], sizes[method] = routed, destination.stat().st_size
    return routes, sizes


def _checkpoint_path(case, name, state):
    if isinstance(state["done"], bool) or not isinstance(state["done"], int) or state["done"] < 1:
        raise ValueError("Invalid completed checkpoint count")
    path = case / f"{name}_{state['done']}.npz"
    if Path(state["arrays"]["path"]) != path:
        raise ValueError("Checkpoint belongs to another stage")
    _guard(path)
    return path


def _write_checkpoint(case, name, done, arrays, **metadata):
    marker = case / f"{name}.json"
    _guard(marker)
    _guard(marker.with_suffix(".tmp"))
    previous = json.loads(marker.read_text()) if marker.exists() else None
    old = _checkpoint_path(case, name, previous) if previous else None
    if previous and done <= previous["done"]:
        raise ValueError("Checkpoint must advance its completed posting count")
    path = case / f"{name}_{done}.npz"
    _guard(path)
    path.unlink(missing_ok=True)
    _npz(path, **arrays)
    write_json(marker, {"done": done, "arrays": artifact(path), **metadata})
    if old is not None:
        old.unlink(missing_ok=True)


def _clean_checkpoint(case, name, keep=None):
    pattern = re.compile(rf"{re.escape(name)}_[0-9]+\.npz(?:\.tmp)?")
    for path in case.iterdir():
        if keep is not None and path in (keep, case / f"{name}.json"):
            continue
        if path.name in (f"{name}.json", f"{name}.tmp") or pattern.fullmatch(path.name):
            _guard(path)
            path.unlink(missing_ok=True)


def _postings(case, records, counts, posting_rows, labels_sha256):
    """Save one clustered FP16 corpus directly, without raw export spool files."""
    directory = case / "clusters"
    _guard(directory)
    directory.mkdir(exist_ok=True)
    names = {f"cluster_{cid}.bin.gz" for cid in range(len(counts))}
    if {path.name for path in directory.iterdir()} - names - {name + ".tmp" for name in names}:
        raise ValueError("Refusing unrelated files in the saved posting directory")
    marker = case / "postings.json"
    _guard(marker)
    identity = {"labels_sha256": labels_sha256,
                "counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest()}
    sizes = np.zeros(len(counts), np.int64)
    hashes = np.zeros(len(counts), dtype="S64")
    done = 0
    checkpoint = None
    if marker.exists():
        previous = json.loads(marker.read_text())
        checkpoint = _checkpoint_path(case, "postings", previous)
        _check(previous["arrays"], case)
        if previous["identity"] != identity:
            raise ValueError("Saved postings use different fixed assignments")
        with np.load(checkpoint, allow_pickle=False) as loaded:
            sizes, hashes = loaded["sizes"], loaded["hashes"]
        done = previous["done"]
        if (not isinstance(done, int) or not 0 < done <= len(counts)
                or sizes.shape != counts.shape or sizes.dtype != np.int64
                or hashes.shape != counts.shape or hashes.dtype != np.dtype("S64")
                or (sizes[:done] <= 0).any() or (sizes[done:] != 0).any()):
            raise ValueError("Invalid saved posting export checkpoint")
    _clean_checkpoint(case, "postings", keep=checkpoint)
    for cid in range(done):
        path = directory / f"cluster_{cid}.bin.gz"
        _guard(path)
        data = path.read_bytes()
        if len(data) != sizes[cid] or hashlib.sha256(data).hexdigest().encode() != hashes[cid]:
            raise ValueError("Saved clustered posting changed")
    offsets = np.r_[0, np.cumsum(counts)]
    interval = max(1, len(counts) // 16)
    for cid in range(done, len(counts)):
        path = directory / f"cluster_{cid}.bin.gz"
        temporary = path.with_name(path.name + ".tmp")
        _guard(path)
        _guard(temporary)
        raw = records[posting_rows[offsets[cid]:offsets[cid + 1]]].tobytes()
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        if path.exists():
            if path.read_bytes() != compressed:
                raise ValueError("Refusing to overwrite a different clustered posting")
        else:
            temporary.write_bytes(compressed)
            temporary.replace(path)
        temporary.unlink(missing_ok=True)
        sizes[cid], hashes[cid] = len(compressed), hashlib.sha256(compressed).hexdigest().encode()
        if (cid + 1) % interval == 0 or cid + 1 == len(counts):
            _write_checkpoint(case, "postings", cid + 1, {"sizes": sizes, "hashes": hashes}, identity=identity)
            print(f"Saved FP16 postings {cid + 1:,}/{len(counts):,}", flush=True)
    return sizes, hashes


def _checkpoint(case, method, done, rows, scores, sizes, timings):
    _write_checkpoint(case, f"progress_{method}", done,
                      {"rows": rows, "scores": scores, "sizes": sizes}, timings=timings)


def _posting_case(case, method, records, counts, posting_rows, queries, routes, gpu):
    names = list(routes)
    shape = (len(names), len(queries), next(iter(routes.values())).shape[1])
    rows = np.full(shape, -1, np.int32)
    scores = np.full(shape, -np.inf, np.float64)
    sizes = np.zeros(len(counts), np.int64)
    timings = dict(encode_seconds=0., decode_seconds=0., score_seconds=0.)
    done = 0
    marker = case / f"progress_{method}.json"
    _guard(marker)
    checkpoint = None
    if marker.exists():
        previous = json.loads(marker.read_text())
        checkpoint = _checkpoint_path(case, f"progress_{method}", previous)
        _check(previous["arrays"], case)
        with np.load(checkpoint, allow_pickle=False) as loaded:
            rows, scores, sizes = loaded["rows"], loaded["scores"], loaded["sizes"]
        done, timings = previous["done"], previous["timings"]
        if (rows.shape != shape or rows.dtype != np.int32 or scores.shape != shape or scores.dtype != np.float64
                or sizes.shape != counts.shape or sizes.dtype != np.int64 or not isinstance(done, int)
                or not 0 <= done <= len(counts) or (sizes[:done] <= 0).any()
                or (sizes[done:] != 0).any() or (rows < -1).any() or (rows >= len(records)).any()
                or np.isnan(scores).any() or np.isposinf(scores).any()):
            raise ValueError("Invalid posting checkpoint arrays")
    _clean_checkpoint(case, f"progress_{method}", keep=checkpoint)
    offsets = np.r_[0, np.cumsum(counts)]
    export_state = json.loads((case / "postings.json").read_text())
    export_path = _checkpoint_path(case, "postings", export_state)
    _check(export_state["arrays"], case)
    if export_state["done"] != len(counts):
        raise ValueError("Clustered postings must be fully saved before testing")
    with np.load(export_path, allow_pickle=False) as loaded:
        saved_sizes, saved_hashes = loaded["sizes"], loaded["hashes"]
    orders = [np.argsort(routes[name].reshape(-1), kind="stable") for name in names]
    starts = [np.r_[0, np.cumsum(np.bincount(routes[name].reshape(-1), minlength=len(counts)))] for name in names]
    interval = max(1, len(counts) // 16)
    for cid in range(done, len(counts)):
        dimensions = records.dtype["vec"].shape[0]
        global_rows = posting_rows[offsets[cid]:offsets[cid + 1]]
        path = case / "clusters" / f"cluster_{cid}.bin.gz"
        _guard(path)
        compressed_source = path.read_bytes()
        if len(compressed_source) != saved_sizes[cid] or hashlib.sha256(compressed_source).hexdigest().encode() != saved_hashes[cid]:
            raise ValueError("Saved clustered posting changed before testing")
        raw = gzip.decompress(compressed_source)
        if len(raw) != int(counts[cid]) * record_dtype(dimensions).itemsize:
            raise ValueError("Saved posting count changed before testing")
        original = np.frombuffer(raw, records.dtype)
        started = time.perf_counter()
        encoded = raw if method == "fp16" else encode_posting(raw, dimensions, int(method[3:]))
        compressed = gzip.compress(encoded, compresslevel=9, mtime=0)
        sizes[cid] = len(compressed)
        timings["encode_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        codes, pages, vectors = decode_fp16(encoded, dimensions) if method == "fp16" else decode_posting(encoded, dimensions)
        if (not np.array_equal(codes, original["code"])
                or not np.array_equal(pages, original["page"])):
            raise ValueError("Posting export or codec changed source record order/identities")
        timings["decode_seconds"] += time.perf_counter() - started
        selected = [order[start[cid]:start[cid + 1]] for order, start in zip(orders, starts)]
        if len(vectors) and any(len(ids) for ids in selected):
            started = time.perf_counter()
            query_ids = np.unique(np.concatenate([ids // shape[2] for ids in selected]))
            winners, values = best_matches(queries[query_ids], vectors, gpu=gpu)
            for variant, ids in enumerate(selected):
                qids, ranks = ids // shape[2], ids % shape[2]
                take = np.searchsorted(query_ids, qids)
                rows[variant, qids, ranks] = global_rows[winners[take]]
                scores[variant, qids, ranks] = values[take]
            timings["score_seconds"] += time.perf_counter() - started
        if (cid + 1) % interval == 0 or cid + 1 == len(counts):
            _checkpoint(case, method, cid + 1, rows, scores, sizes, timings)
            print(f"{method}: {cid + 1:,}/{len(counts):,} postings, all centroid routes", flush=True)
    return rows, scores, sizes, timings


def _measure(records, labels, counts, truth, queries, routes, winners, scores, sizes,
             baseline, head_count, centroid_method, posting_method, centroid_bytes, config, timings):
    vectors = NormalizedVectors(records)
    actual_truth = vectors[truth].astype(np.float64)
    actual_truth /= np.linalg.norm(actual_truth, axis=1)[:, None]
    true_scores = np.einsum("ij,ij->i", actual_truth, queries.astype(np.float64))
    points = []
    for column, probe in enumerate(config["probes"]):
        selected = winners[:, column]
        valid = selected >= 0
        original_scores = np.full(len(queries), -np.inf)
        original = vectors[selected[valid]].astype(np.float64)
        original /= np.linalg.norm(original, axis=1)[:, None]
        original_scores[valid] = np.einsum("ij,ij->i", original, queries[valid].astype(np.float64))
        download = sizes[routes[:, :probe]].sum(axis=1)
        average = float(download.mean() / 1e6)
        points.append({"head_count": head_count, "target_heads": head_count, "actual_heads": head_count,
                       "centroid_method": centroid_method, "posting_method": posting_method,
                       "probes": probe, "queries": len(queries), "noise_std": config["noise_std"],
                       "recall_percent": float(_same_images(records, selected, truth).mean() * 100),
                       "tie_aware_recall_percent": float((original_scores >= true_scores - 2e-6).mean() * 100),
                       "recall_vs_fp16_percent": float(_same_images(records, selected, baseline[:, column]).mean() * 100),
                       "routing_coverage_percent": float((routes[:, :probe] == labels[truth, None]).any(axis=1).mean() * 100),
                       "mean_original_cosine_loss": float(np.maximum(0, true_scores - original_scores).mean()),
                       "posting_gzip_MB_mean": average, "posting_gzip_MB_p95": float(np.percentile(download, 95) / 1e6),
                       "centroid_gzip_MB": centroid_bytes / 1e6, "cold_gzip_MB_mean": average + centroid_bytes / 1e6,
                       "total_posting_gzip_GB": int(sizes.sum()) / 1e9,
                       "records_fetched_mean": float(counts[routes[:, :probe]].sum(axis=1).mean()),
                       **timings})
    return points


def _render(output):
    from balanced_l2.plot_joint_quantization import render
    return render(output)


def _check_metrics(output, ledger, target):
    saved = ledger["cases"][str(target)]
    _check(saved["metrics"], output)
    retained = json.loads(Path(saved["metrics"]["path"]).read_text())
    points = [point for point in ledger["points"] if point["head_count"] == target]
    expected = {(centroid, posting, probe) for centroid in CODECS for posting in CODECS
                for probe in ledger["manifest"]["probes"]}
    observed = [(point["centroid_method"], point["posting_method"], point["probes"]) for point in points]
    if retained["points"] != points or len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Retained metrics differ from the complete codec grid")


def _finish(output, ledger):
    _render(output)
    stage = output / "_staging"
    _guard(stage)
    if stage.exists():
        expected = {"sample.f32"}
        if {path.name for path in stage.iterdir()} - expected:
            raise ValueError("Refusing unrelated files in shared staging")
        for path in stage.iterdir():
            _guard(path)
            if ledger["staging"][path.name]["path"] != str(path.resolve()):
                raise ValueError("Staging artifact belongs to another output")
            path.unlink()
        stage.rmdir()
    parent = output / "_cases"
    _guard(parent)
    if parent.exists():
        parent.rmdir()
    ledger["staging_removed"] = True
    _save(output, ledger)


def run(paths=ROOT / "vectors", output=ROOT / "joint_quantization_comparison", *, head_targets=HEADS,
        query_count=100000, probes=PROBES, noise=NOISE, dim=384, sample_size=2500000,
        threads=8, gpu=True, max_iter=20):
    files = input_files(paths)
    inputs = [snapshot(path) for path in files]
    heads, probes = tuple(head_targets), tuple(probes)
    rows = sum(item["bytes"] // record_dtype(dim).itemsize for item in inputs)
    if (any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (query_count, dim, sample_size, threads, max_iter))
            or not 0 < query_count <= rows or not heads or len(set(heads)) != len(heads)
            or any(isinstance(head, bool) or not isinstance(head, int) or not 2 <= head <= min(rows, sample_size) for head in heads)
            or not probes or tuple(sorted(set(probes))) != probes
            or any(isinstance(probe, bool) or not isinstance(probe, int) or not 1 <= probe <= min(heads) for probe in probes)
            or not np.isfinite(noise) or noise < 0 or any(item["bytes"] % record_dtype(dim).itemsize for item in inputs)
            or not gpu and rows > 100000):
        raise ValueError("Invalid comparison settings; CPU mode is restricted to tiny fixtures")
    requested = Path(output)
    _guard(requested)
    output = requested.resolve()
    for path in files:
        if output == path or output in path.parents or path.parent == output or path.parent in output.parents:
            raise ValueError("Comparison output must be separate from original source directories")
    producers = ("benchmark_joint_quantization.py", "balanced_l2/build.py", "balanced_l2/common.py",
                 "balanced_l2/source_records.py", "balanced_l2/benchmark_ground_truth.py", "balanced_l2/centroid_quantization.py",
                 "balanced_l2/posting_quantization.py", "balanced_l2/posting_scoring.py", "balanced_l2/quantized_routing.py",
                 "benchmark_cluster_quantization.py", "client/scripts/encode-centroids.py")
    manifest = {"schema_version": 2, "head_targets": list(heads), "centroid_methods": list(CODECS),
                "posting_methods": list(CODECS), "probes": list(probes), "noise_std": float(noise),
                "query_count": query_count, "dimensions": dim, "source_rows": rows, "inputs": inputs,
                "sample_size": sample_size, "sample_seed": 42, "max_iter": max_iter, "threads": threads, "gpu": gpu,
                "training_initialization": "Native cuVS default; Python API exposes no initializer seed" if gpu else "Faiss fixture seed 42",
                "training_method": "cuvs_balanced_squared_l2" if gpu else "faiss_cpu_fixture_only",
                "gzip_level": 9, "cleanup": "Delete each head-count build after all 16 codec pairs are durable, before the next count",
                "storage": "Original files mapped read-only; shared training sample and one directly exported FP16 clustered corpus plus current-count checkpoints",
                "truth_definition": "Shared exhaustive float32 dot over original unit-normalized vectors; image-ID/page recall and 2e-6 tie-aware recall",
                "scoring_definition": "Literal decoded Float32 postings, browser-style Float64 cosine products and Float32 inverse norms",
                "size_definition": "Actual gzip9 posting records including IDs, headers, and means/scales; cold bytes add the selected centroid file once",
                "limits": "Noisy replay, not real-image recall; timings exclude browser and network latency",
                "producer_sha256": {name: sha256(source_path(ROOT, name)) for name in producers},
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "faiss-cpu", "cupy-cuda12x", "cuvs-cu12")}}
    config_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "results.json"
    _guard(ledger_path)
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["manifest"] != manifest:
            raise ValueError("Sources, benchmark code, versions, or settings changed; use a fresh --output")
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing an unrelated nonempty comparison directory")
        ledger = {"manifest": manifest, "status": "running", "points": [], "cases": {}}
        _save(output, ledger)
    if ledger["status"] == "complete":
        _check(ledger["truth"], output)
        for target in heads:
            _check_metrics(output, ledger, target)
            _clean_case(output, target, config_hash)
        _finish(output, ledger)
        return ledger
    records = None
    try:
        records = SourceRecords(files, dim)
        vectors = NormalizedVectors(records)
        paths = _stage(output, ledger, files, records)
        queries, truth = _truth(output, ledger, vectors, files, gpu)
        _release_gpu()
        for target in heads:
            _unchanged(files, manifest)
            saved = ledger["cases"].setdefault(str(target), {"status": "running", "pairs": {}})
            if saved["status"] == "complete":
                _check_metrics(output, ledger, target)
                _clean_case(output, target, config_hash)
                continue
            case = _case_directory(output, target, config_hash)
            print(f"Comparing {target:,} clusters, all 16 centroid/posting codec pairs", flush=True)
            centroids, labels, counts = _model(output, ledger, saved, case, target, paths, files, vectors)
            posting_rows = np.argsort(labels, kind="stable")
            _postings(case, records, counts, posting_rows, saved["provenance"]["labels_sha256"])
            _unchanged(files, manifest)
            routes, centroid_sizes = _routes(output, ledger, saved, case, centroids, queries)
            for method in CODECS:
                for previous in saved["pairs"].values():
                    _check(previous["winners"], case)
                if all(f"{centroid}/{method}" in saved["pairs"] for centroid in CODECS):
                    _clean_checkpoint(case, f"progress_{method}")
                    continue
                _unchanged(files, manifest)
                winners, scores, sizes, timings = _posting_case(case, method, records, counts, posting_rows, queries, routes, gpu)
                for variant, centroid in enumerate(CODECS):
                    pair = f"{centroid}/{method}"
                    if pair in saved["pairs"]:
                        continue
                    selected, similarities = prefix_winners(winners[variant], scores[variant], probes)
                    baseline_path = case / f"winners_{centroid}_fp16.npz"
                    if method == "fp16":
                        baseline = selected
                    else:
                        with np.load(baseline_path, allow_pickle=False) as loaded:
                            baseline = loaded["rows"]
                    points = _measure(records, labels, counts, truth, queries, routes[centroid], selected,
                                      similarities, sizes, baseline, target, centroid, method, centroid_sizes[centroid], manifest, timings)
                    destination = case / f"winners_{centroid}_{method}.npz"
                    _npz(destination, rows=selected, scores=similarities, gzip_sizes=sizes)
                    saved["pairs"][pair] = {"status": "complete", "winners": artifact(destination)}
                    ledger["points"].extend(points)
                    _save(output, ledger)
                _clean_checkpoint(case, f"progress_{method}")
                del winners, scores
                _release_gpu()
            points = [point for point in ledger["points"] if point["head_count"] == target]
            if len(points) != len(CODECS) ** 2 * len(probes):
                raise ValueError("Incomplete centroid/posting codec grid")
            metrics_path = output / f"metrics_h{target}.json"
            write_json(metrics_path, {"head_count": target, "provenance": saved["provenance"], "points": points})
            ledger["cases"][str(target)] = {"status": "complete", "metrics": artifact(metrics_path),
                                            "provenance": saved["provenance"]}
            _save(output, ledger)
            del labels, centroids, posting_rows, routes
            _clean_case(output, target, config_hash)
            _release_gpu()
            print(f"Removed {target:,}-cluster build; metrics retained", flush=True)
            _render(output)
        _unchanged(files, manifest)
        ledger["status"] = "complete"
        ledger.pop("last_error", None)
        _save(output, ledger)
        _finish(output, ledger)
        return ledger
    except BaseException as error:
        ledger["last_error"] = str(error)
        _save(output, ledger)
        raise
    finally:
        if records is not None:
            records.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="*", type=Path, default=[ROOT / "vectors"])
    parser.add_argument("--output", type=Path, default=ROOT / "joint_quantization_comparison")
    parser.add_argument("--heads", type=int, nargs="+", default=list(HEADS))
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--noise", type=float, default=NOISE)
    parser.add_argument("--probes", type=int, nargs="+", default=list(PROBES))
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--sample-size", type=int, default=2500000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu", action="store_true", help="Tiny fixtures only; uses ordinary CPU k-means")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--train-worker", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_worker:
        sample, native, heads, iterations = args.train_worker
        pipeline.train(sample, native, int(heads), int(iterations))
    elif args.list_cases:
        for head in args.heads:
            for centroid in CODECS:
                for posting in CODECS:
                    print(f"{head}: centroids={centroid}, postings={posting}, probes={','.join(map(str, args.probes))}")
    else:
        result = run(args.input, args.output, head_targets=args.heads, query_count=args.queries,
                     probes=args.probes, noise=args.noise, dim=args.dimensions,
                     sample_size=args.sample_size, threads=args.threads, gpu=not args.cpu)
        print(f"Comparison {result['status']}: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
