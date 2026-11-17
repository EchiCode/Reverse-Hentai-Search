"""Sequential balanced L2 head-count calibration with exact gzip download costs."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

from balanced_l2.build import assign
from balanced_l2.benchmark_ground_truth import benchmark_noises
from balanced_l2.common import (artifact, check_artifact, input_files, matrix_shape, prepare_vectors,
                                read_matrix, record_dtype, require_disk_space, sha256, snapshot, source_path, write_json)
from balanced_l2.export import decode_centroids, export_centroids, export_postings
from balanced_l2.quantized_routing import route_queries

ROOT = Path(__file__).resolve().parent
HEADS = (64000, 80000, 96000, 128000)
PROBES = (1, 2, 4, 5, 8, 10, 12, 16, 32, 64)
NOISES = (.005, .01, .025)
CONTROL_TARGET = 64000
CONTROL = ROOT / "clusters_balanced_l2_64k"
BENCHMARK_ROOT = ROOT / "balanced_l2_benchmark"


def release_gpu():
    if "cupy" in sys.modules:
        sys.modules["cupy"].get_default_memory_pool().free_all_blocks()
        sys.modules["cupy"].get_default_pinned_memory_pool().free_all_blocks()


def save(output, ledger):
    write_json(output / "comparison.json", ledger)
    points = [row for case in ledger["cases"].values() for rows in case.get("noise_results", {}).values() for row in rows]
    write_json(output / "points.json", {"manifest": ledger["manifest"], "status": ledger["status"], "points": points})
    if points:
        path = output / "points.partial.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
        path.replace(output / "points.csv")


def train_worker(sample, native, target):
    native.mkdir(parents=True, exist_ok=True)
    release_gpu()
    log = native / "training.log"
    with log.open("a") as stream:
        result = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--train-worker",
                                 str(sample), str(native), str(target)], cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "8",
                                "OPENBLAS_NUM_THREADS": "8", "CUPY_TF32": "0"})
    if result.returncode:
        raise RuntimeError(f"Balanced L2 training failed for {target} heads; see {log}")


def prepare_stage(output, ledger, files, dim, sample_size):
    work = output / "_staging"
    if work.is_symlink():
        raise ValueError("Benchmark staging must not be a symlink")
    work.mkdir(exist_ok=True)
    source, sample = work / "vectors.f32", work / "sample.f32"
    previous = ledger.get("staging")
    if previous:
        if snapshot(source) != previous["source"] or snapshot(sample) != previous["sample"]:
            raise ValueError("Benchmark staging changed; use a fresh output")
        return source, sample
    rows = ledger["manifest"]["source_rows"]
    sample_rows = min(rows, sample_size)
    retained = len(ledger["manifest"]["head_targets"]) * rows * 4 + sum(ledger["manifest"]["head_targets"]) * dim * 6
    require_disk_space(output, (rows + sample_rows) * dim * 4 + 16 + retained
                       + rows * (2 * record_dtype(dim).itemsize + 4)
                       + ledger["manifest"]["query_count"] * (dim * 4 + 16) * len(ledger["manifest"]["noise"]))
    temporary, sample_temporary = work / "vectors.partial.f32", work / "sample.partial.f32"
    for path in (source, sample, temporary, sample_temporary):
        if path.is_symlink():
            raise ValueError("Owned staging files must not be symlinks")
        path.unlink(missing_ok=True)
    selected = np.sort(np.random.default_rng(42).choice(rows, sample_rows, replace=False))
    print("Preparing common normalized corpus and training sample", flush=True)
    prepare_vectors(files, temporary, dim, sample_rows=selected, sample_path=sample_temporary)
    if ledger["manifest"]["inputs"] != [snapshot(path) for path in files]:
        raise ValueError("Original vectors changed during staging")
    temporary.replace(source)
    sample_temporary.replace(sample)
    ledger["staging"] = {"source": snapshot(source), "sample": snapshot(sample), "sample_sha256": sha256(sample)}
    save(output, ledger)
    return source, sample


def reuse_control(target, manifest, sample_sha):
    path = CONTROL / "build.json"
    if target != CONTROL_TARGET or not path.exists():
        return None
    state = json.loads(path.read_text())
    params = state["parameters"]
    if (state.get("method") != "balanced_l2" or state["inputs"] != manifest["inputs"]
            or state["rows"] != manifest["source_rows"]
            or params["dimensions"] != manifest["dimensions"]
            or params["sample_size"] != manifest["sample_size"] or params["seed"] != 42
            or params["max_iter"] != 20 or "assignments" not in state["completed"]):
        return None
    expected_sample = state.get("training", {}).get("sample_sha256")
    if expected_sample is not None and expected_sample != sample_sha:
        raise ValueError("Cached 64k control uses a different training sample")
    for name in ("heads", "assignments"):
        check_artifact(state["artifacts"][name])
    sizes, counts = CONTROL / "posting_gzip_sizes.npy", CONTROL / "posting_counts.npy"
    if not sizes.exists() or not counts.exists():
        return None
    return state


def counts_for(labels_path, rows, heads):
    if labels_path.stat().st_size != rows * 4:
        raise ValueError("Assignments do not cover the entire corpus")
    counts = np.zeros(heads, np.int64)
    for first in range(0, rows, 250000):
        batch = np.memmap(labels_path, dtype="<i4", mode="r", offset=first * 4, shape=(min(250000, rows - first),))
        if (batch < 0).any() or (batch >= heads).any():
            raise ValueError("Invalid single assignment in the full corpus")
        counts += np.bincount(batch, minlength=heads)
        del batch
    return counts


def evaluate(case, target, actual, labels, counts, gzip_sizes, queries, nearest, noise, decoded, centroid_bytes, fixed_bytes):
    destination = case / f"noise_{noise:g}"
    destination.mkdir(exist_ok=True)
    routes_path = destination / f"routes{max(PROBES)}.npy"
    query_sha = hashlib.sha256(queries.tobytes()).hexdigest()
    identity = {"query_sha256": query_sha, "decoded_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
                "probes": min(max(PROBES), actual), "metric": "l2"}
    marker = destination / "routes.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity:
            raise ValueError("Cached routes differ from the quantized centroids or queries")
        check_artifact(saved["routes"])
        routes = np.load(routes_path, allow_pickle=False)
    else:
        routes = route_queries(decoded, queries, probes=max(PROBES), query_block=4096, gpu=True, metric="l2")
        temporary = routes_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, routes, allow_pickle=False)
        temporary.replace(routes_path)
        write_json(marker, {"identity": identity, "routes": artifact(routes_path)})
    if (routes.shape != (len(queries), min(max(PROBES), actual)) or routes.dtype.kind not in "iu"
            or (routes < 0).any() or (routes >= actual).any()
            or (np.diff(np.sort(routes, axis=1), axis=1) == 0).any()):
        raise ValueError("Invalid saved route shape or centroid IDs")
    nearest_labels = np.asarray(labels[nearest]).reshape(-1)
    np.save(destination / "nearest_labels.npy", nearest_labels)
    points = []
    for requested in PROBES:
        selected = routes[:, :requested]
        hits = (selected == nearest_labels[:, None]).any(axis=1)
        download = gzip_sizes[selected].sum(axis=1)
        points.append({"target_heads": target, "actual_heads": actual, "noise_std": noise,
                       "queries": len(queries), "probes": requested, "actual_probes": selected.shape[1],
                       "recovered_exact_nearest_rows": int(hits.sum()), "recall_percent": float(hits.mean() * 100),
                       "centroid_gzip_MB": centroid_bytes / 1e6,
                       "posting_gzip_MB_mean": float(download.mean() / 1e6),
                       "posting_gzip_MB_p95": float(np.percentile(download, 95) / 1e6),
                       "records_fetched_mean": float(counts[selected].sum(axis=1).mean()),
                       "fixed_model_and_libraries_gzip_MB": fixed_bytes / 1e6,
                       "cold_gzip_MB_excluding_metadata": float((fixed_bytes + centroid_bytes + download.mean()) / 1e6)})
    write_json(destination / "frontier.json", {"metric": "l2", "centroid_bits": 6, "points": points})
    return points


def clean_case(case):
    path = case / "clusters"
    if path.is_symlink():
        raise ValueError("Owned benchmark postings must not be a symlink")
    if path.exists():
        shutil.rmtree(path)


def finish(output, ledger):
    subprocess.run([sys.executable, str(ROOT / "balanced_l2/plot_benchmark.py"), str(output)], check=True)
    staging = output / "_staging"
    if staging.is_symlink():
        raise ValueError("Owned benchmark staging must not be a symlink")
    if staging.exists():
        shutil.rmtree(staging)
    ledger["staging_removed"] = True
    save(output, ledger)


def run(paths="vectors_old", output=BENCHMARK_ROOT / "comparison_lvq6_100k", *, head_targets=HEADS,
        noises=NOISES, query_count=100000, dim=384, sample_size=2500000, threads=8):
    files = input_files(paths)
    inputs = [snapshot(path) for path in files]
    record_bytes = record_dtype(dim).itemsize
    rows = sum(item["bytes"] // record_bytes for item in inputs)
    heads = tuple(head_targets)
    noises = tuple(dict.fromkeys(float(noise) for noise in noises))
    if (dim <= 0 or threads <= 0 or query_count <= 0 or query_count > rows or sample_size <= 0
            or not heads or len(set(heads)) != len(heads)
            or any(isinstance(head, bool) or not isinstance(head, int) or not 2 <= head <= min(rows, sample_size) for head in heads)
            or not noises or any(not math.isfinite(noise) or noise < 0 for noise in noises)
            or any(item["bytes"] % record_bytes for item in inputs)):
        raise ValueError("Invalid head counts, query/noise parameters, or complete source records")
    requested_output = Path(output)
    if requested_output.is_symlink():
        raise ValueError("Benchmark output must not be a symlink")
    output = requested_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets_path = BENCHMARK_ROOT / "client_assets.json"
    assets = json.loads(assets_path.read_text())
    producers = ("benchmark.py", "balanced_l2/benchmark_ground_truth.py", "balanced_l2/common.py",
                 "balanced_l2/build.py", "balanced_l2/export.py", "balanced_l2/quantized_routing.py", "client/scripts/encode-centroids.py")
    manifest = {"method": "balanced_l2", "head_targets": list(heads), "noise": list(noises), "query_count": query_count,
                "probes": list(PROBES), "metric": "l2", "centroid_bits": 6, "centroid_version": 2,
                "dimensions": dim, "source_rows": rows, "inputs": inputs, "sample_size": sample_size,
                "training_seed": 42, "max_iter": 20, "threads": threads,
                "fixed_model_and_libraries_gzip_bytes": assets["fixed_bytes"]["gzip"],
                "assets_sha256": sha256(assets_path),
                "producer_sha256": {name: sha256(source_path(ROOT, name)) for name in producers}}
    ledger_path = output / "comparison.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["manifest"] != manifest:
            raise ValueError("Benchmark inputs or producer code changed; use a fresh --output")
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing an unrelated nonempty benchmark directory")
        ledger = {"status": "pending", "manifest": manifest, "cases": {}}
        save(output, ledger)
    if ledger["status"] == "complete":
        for value in ledger["truth"].values():
            check_artifact(value)
        for case in ledger["cases"].values():
            for value in case["artifacts"]:
                check_artifact(value)
        finish(output, ledger)
        return ledger
    try:
        source, sample = prepare_stage(output, ledger, files, dim, sample_size)
        if not ledger.get("truth"):
            caches = benchmark_noises(source, output / "truth", query_count, noises, inputs=inputs,
                                       reuse_dir=BENCHMARK_ROOT / "truth" / f"q{query_count}")
            ledger["truth"] = {str(noise): artifact(path) for noise, path in caches.items()}
            save(output, ledger)
        for value in ledger["truth"].values():
            check_artifact(value)
        ledger["status"] = "running"
        for target in heads:
            if inputs != [snapshot(path) for path in files]:
                raise ValueError("Original source files changed")
            case = output / f"h{target}"
            if case.is_symlink():
                raise ValueError("Owned benchmark case must not be a symlink")
            case.mkdir(exist_ok=True)
            saved = ledger["cases"].setdefault(str(target), {"status": "running", "noise_results": {}})
            if saved["status"] == "complete":
                for value in saved["artifacts"]:
                    check_artifact(value)
                clean_case(case)
                continue
            started = time.perf_counter()
            print(f"Balanced L2 comparison: {target} heads", flush=True)
            control = reuse_control(target, manifest, ledger["staging"]["sample_sha256"])
            native = CONTROL / "native" if control is not None else case / "native"
            if native.is_symlink():
                raise ValueError("Centroid artifacts must not be a symlink")
            native.mkdir(exist_ok=True)
            if control is None:
                marker = native / "training.json"
                if not marker.exists():
                    train_worker(sample, native, target)
                training = json.loads(marker.read_text())
                if (training["params"]["n_clusters"] != target or training["params"]["hierarchical_n_iters"] != 20
                        or training["sample"]["sha256"] != ledger["staging"]["sample_sha256"]
                        or training["metric"] != "l2"):
                    raise ValueError("Saved training differs from the common sample or parameters")
                check_artifact(training["heads"])
                labels_path = native / "assignments.i32"
                assignment_marker = native / "assignment.json"
                if assignment_marker.exists():
                    assignment = json.loads(assignment_marker.read_text())
                    if assignment["heads"] != artifact(native / "heads.f32") or assignment["source"] != snapshot(source):
                        raise ValueError("Saved assignment inputs changed")
                    check_artifact(assignment["labels"])
                else:
                    labels_path.unlink(missing_ok=True)
                    labels_path.with_name(labels_path.name + ".tmp").unlink(missing_ok=True)
                    metadata = assign(source, native / "heads.f32", labels_path, gpu=True)
                    write_json(assignment_marker, {"source": snapshot(source), "heads": artifact(native / "heads.f32"),
                                                  "labels": artifact(labels_path), "assignment": metadata})
                    release_gpu()
            else:
                training = control["training"]
                labels_path = native / "assignments.i32"
            actual, head_dim = matrix_shape(native / "heads.f32")
            if head_dim != dim or (control is None and actual != target):
                raise ValueError("Centroid dimensions or requested cluster count differ")
            counts = counts_for(labels_path, rows, actual)
            saved.update(actual_heads=actual, control_reused=control is not None, training=training)
            if control is not None:
                gzip_sizes = np.load(CONTROL / "posting_gzip_sizes.npy", allow_pickle=False)
                measured_counts = np.load(CONTROL / "posting_counts.npy", allow_pickle=False)
                gzip_path = CONTROL / "posting_gzip_sizes.npy"
                saved["head_count_note"] = f"Retained 64k control has {actual} actual centroids"
            else:
                if saved.get("gzip"):
                    for value in saved["gzip"]:
                        check_artifact(value)
                else:
                    export_postings(files, labels_path, actual, case, dim=dim, threads=threads)
                    saved["gzip"] = [artifact(case / name) for name in ("posting_counts.npy", "posting_gzip_sizes.npy")]
                    save(output, ledger)
                gzip_path = case / "posting_gzip_sizes.npy"
                gzip_sizes = np.load(gzip_path, allow_pickle=False)
                measured_counts = np.load(case / "posting_counts.npy", allow_pickle=False)
            if (not np.array_equal(counts, measured_counts) or gzip_sizes.shape != (actual,)
                    or gzip_sizes.dtype.kind not in "iu" or (gzip_sizes <= 0).any()):
                raise ValueError("Full-corpus gzip sizes or posting counts differ from assignments")
            np.save(case / "posting_counts.npy", counts)
            if not saved.get("centroids"):
                export_centroids(case, native / "heads.f32")
                saved["centroids"] = artifact(case / "centroids.lvq6.bin.gz")
                save(output, ledger)
            check_artifact(saved["centroids"])
            raw = gzip.decompress((case / "centroids.lvq6.bin.gz").read_bytes())
            if struct.unpack_from("<H", raw, 8)[0] != 2:
                raise ValueError("Balanced L2 benchmark requires the metric-tagged centroid index")
            decoded = decode_centroids(raw, actual, dim)
            labels = np.memmap(labels_path, dtype="<i4", mode="r", shape=(rows,))
            for noise in noises:
                with np.load(ledger["truth"][str(noise)]["path"], allow_pickle=False) as truth:
                    points = evaluate(case, target, actual, labels, counts, gzip_sizes, truth["queries"],
                                      truth["truth"][:, 0], noise, decoded, saved["centroids"]["bytes"],
                                      manifest["fixed_model_and_libraries_gzip_bytes"])
                saved["noise_results"][str(noise)] = points
                save(output, ledger)
            saved["balance"] = {"nonempty_postings": int(np.count_nonzero(counts)), "empty_postings": int((counts == 0).sum()),
                                "median_records": float(np.median(counts)), "p95_records": float(np.percentile(counts, 95)),
                                "p99_records": float(np.percentile(counts, 99)), "max_records": int(counts.max())}
            retained = [native / "heads.f32", labels_path, case / "posting_counts.npy", gzip_path, case / "centroids.lvq6.bin.gz"]
            for noise in noises:
                retained += [case / f"noise_{noise:g}" / name for name in
                             ("frontier.json", f"routes{max(PROBES)}.npy", "nearest_labels.npy")]
            saved.update(status="complete", artifacts=[artifact(path) for path in retained],
                         elapsed_seconds_this_attempt=time.perf_counter() - started)
            save(output, ledger)
            del labels, decoded, raw
            release_gpu()
            clean_case(case)
        if inputs != [snapshot(path) for path in files] or snapshot(source) != ledger["staging"]["source"]:
            raise ValueError("Original or staged corpus changed during the benchmark")
        ledger["status"] = "complete"
        ledger.pop("reason", None)
        save(output, ledger)
        finish(output, ledger)
    except Exception as error:
        ledger.update(status="error", reason=str(error))
        save(output, ledger)
        raise
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="*", default=["vectors_old"])
    parser.add_argument("--output", type=Path, default=BENCHMARK_ROOT / "comparison_lvq6_100k")
    parser.add_argument("--heads", type=int, nargs="+", default=list(HEADS))
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--noise", type=float, nargs="+", default=list(NOISES))
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--train-worker", nargs=3, metavar=("SAMPLE", "OUTPUT", "HEADS"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_worker:
        from balanced_l2.build import train
        sample, output, heads = args.train_worker
        print(json.dumps(train(sample, output, int(heads)), indent=2), flush=True)
    else:
        result = run(args.input, args.output, head_targets=args.heads, noises=args.noise,
                     query_count=args.queries, dim=args.dimensions)
        print(f"Balanced L2 benchmark {result['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
