"""Sampled balanced L2 training, full assignment, and resumable website export."""

import json
import os
import struct
from pathlib import Path
import shutil
import time

import numpy as np

from balanced_l2.common import (artifact, check_artifact, input_files, iter_record_batches, matrix_shape, prepare_vectors,
                                read_matrix, record_dtype, require_disk_space, snapshot, write_json)
from balanced_l2.export import export_centroids, export_centroids_brotli, export_postings


GPU_SCORE_BYTES = 1024 ** 3


class _NormalizedRecords:
    """Sequential normalized slices over bounded original-record batches."""

    def __init__(self, files, count, dim, batch_size):
        self.count, self.dim, self.batch_size = count, dim, min(batch_size, 65536)
        self.batches = iter_record_batches(files, dim, self.batch_size)
        self.records = np.empty(0, dtype=record_dtype(dim))
        self.position = self.cursor = 0

    def __len__(self):
        return self.count

    def __getitem__(self, rows):
        if (not isinstance(rows, slice) or rows.step not in (None, 1)
                or rows.start != self.position or rows.stop is None
                or not self.position < rows.stop <= self.count
                or rows.stop - self.position > self.batch_size):
            raise ValueError("Original records require bounded sequential slices")
        values = np.empty((rows.stop - self.position, self.dim), np.float32)
        written = 0
        while written < len(values):
            if self.cursor == len(self.records):
                try:
                    self.records = next(self.batches)
                except StopIteration as error:
                    raise ValueError("Original vector row count changed") from error
                self.cursor = 0
            length = min(len(values) - written, len(self.records) - self.cursor)
            values[written:written + length] = self.records["vec"][self.cursor:self.cursor + length]
            self.cursor += length
            written += length
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Source vectors must be finite and nonzero")
        self.position = rows.stop
        return values / norms[:, None]

    def close(self):
        self.records = None
        self.batches.close()


def train(sample_path, output_dir, clusters=64000, max_iter=20):
    import cupy as cp
    import cuvs
    from cuvs.cluster import kmeans

    rows, dim = matrix_shape(sample_path)
    if not 2 <= clusters <= rows or max_iter <= 0:
        raise ValueError("Cluster count must be between two and sample rows; iterations must be positive")
    sample = np.memmap(sample_path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))
    for first in range(0, rows, 65536):
        block = sample[first:first + 65536]
        if (not np.isfinite(block).all()
                or not np.allclose(np.einsum("ij,ij->i", block, block), 1, atol=1e-4, rtol=0)):
            raise ValueError("Training sample must contain finite unit-normalized vectors")
    required = sample.nbytes + clusters * dim * 4 + 1024 ** 3
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < required:
        raise MemoryError(f"Balanced L2 training needs at least {required / 1024 ** 3:.2f} GiB free VRAM; "
                          f"found {free / 1024 ** 3:.2f} GiB")
    params = {"n_clusters": clusters, "metric": "sqeuclidean", "hierarchical": True,
              "hierarchical_n_iters": max_iter}
    started = time.perf_counter()
    centers, inertia, iterations = kmeans.fit(kmeans.KMeansParams(**params), cp.asarray(sample))
    centers = cp.asnumpy(cp.asarray(centers)).astype("<f4", copy=False)
    seconds = time.perf_counter() - started
    if centers.shape != (clusters, dim) or not np.isfinite(centers).all():
        raise ValueError("cuVS returned invalid centroids or an incorrect cluster count")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "heads.f32"
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as output:
        output.write(struct.pack("<ii", clusters, dim))
        centers.tofile(output)
    temporary.replace(target)
    metadata = {"method": "balanced_l2", "metric": "l2", "library": "cuvs", "library_version": cuvs.__version__,
                "params": params, "sample": artifact(sample_path), "sample_rows": rows, "head_count": clusters,
                "dimensions": dim, "heads": artifact(target), "training_seconds": seconds,
                "inertia": float(inertia), "iterations": int(iterations),
                "initialization": "Native library default; Python/C API exposes no RNG seed",
                "documentation": "https://docs.nvidia.com/cuvs/api-reference/python-api-cluster-kmeans"}
    write_json(output_dir / "training.json", metadata)
    cp.get_default_memory_pool().free_all_blocks()
    return metadata


def assign(source, heads, labels_path, batch_size=250000, gpu=True):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    source, heads, labels_path = Path(source), Path(heads), Path(labels_path)
    before = {"source": snapshot(source), "heads": snapshot(heads)}
    count, dim = matrix_shape(source)

    def unchanged():
        if before != {"source": snapshot(source), "heads": snapshot(heads)}:
            raise ValueError("Assignment inputs changed while reading")

    vectors = np.memmap(source, dtype="<f4", mode="r", offset=8, shape=(count, dim))
    return assign_vectors(vectors, heads, labels_path, batch_size=batch_size, gpu=gpu, validate_inputs=unchanged)


def assign_vectors(vectors, heads, labels_path, *, batch_size=250000, gpu=True, validate_inputs=None):
    """Assign indexed normalized blocks without requiring a full FP32 corpus file."""
    if batch_size <= 0 or len(vectors) <= 0:
        raise ValueError("Batch size and source row count must be positive")
    heads, labels_path = Path(heads), Path(labels_path)
    before = snapshot(heads)
    centers = read_matrix(heads)
    count, dim = len(vectors), centers.shape[1]
    if labels_path.exists():
        raise FileExistsError(labels_path)
    if validate_inputs is not None:
        validate_inputs()
    started = time.perf_counter()
    if gpu:
        import cupy as cp
        from cupy_backends.cuda.libs import cublas
        device_centers = cp.asarray(centers)
        norms = cp.sum(device_centers * device_centers, axis=1)
        handle = cp.cuda.device.get_cublas_handle()
        old_math = cublas.getMathMode(handle)
        cublas.setMathMode(handle, 2)
    else:
        norms = np.sum(centers * centers, axis=1)
    batch_size = min(batch_size, 4096 if gpu else 1024, max(1, GPU_SCORE_BYTES // (len(centers) * 4)))
    temporary = labels_path.with_name(labels_path.name + ".tmp")
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            for first in range(0, count, batch_size):
                length = min(batch_size, count - first)
                batch = np.asarray(vectors[first:first + length], dtype=np.float32)
                if batch.shape != (length, dim):
                    raise ValueError("Source and centroids have different dimensions")
                if not np.isfinite(batch).all():
                    raise ValueError("Non-finite source vector during assignment")
                scores = (cp.asarray(batch) @ device_centers.T) if gpu else batch @ centers.T
                scores *= -2
                scores += norms
                labels = cp.asnumpy(cp.argmin(scores, axis=1)) if gpu else np.argmin(scores, axis=1)
                output.write(labels.astype("<i4").tobytes())
                del scores, batch
        if before != snapshot(heads):
            raise ValueError("Assignment inputs changed while reading")
        if validate_inputs is not None:
            validate_inputs()
        os.link(temporary, labels_path)
        temporary.unlink()
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        if gpu:
            cublas.setMathMode(handle, old_math)
            scores = None
            del device_centers, norms
            cp.get_default_memory_pool().free_all_blocks()
    return {"method": "exhaustive_squared_l2", "backend": "cupy" if gpu else "numpy",
            "source_rows": count, "heads": len(centers), "query_batch_size": batch_size,
            "numeric_precision": "float32 exhaustive scores; lower-head-ID ties",
            "search_and_write_seconds": time.perf_counter() - started}


def parameters(clusters, dim, batch_size, threads, sample_size, seed, max_iter, gpu):
    if (any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (clusters, dim, batch_size, threads, sample_size, max_iter))
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or batch_size * record_dtype(dim).itemsize > 256 * 1024 ** 2):
        raise ValueError("Invalid clustering parameters or input buffer exceeds 256 MiB")
    return {"clusters": clusters, "dimensions": dim, "batch_size": batch_size, "threads": threads,
            "sample_size": sample_size, "seed": seed, "max_iter": max_iter, "gpu": bool(gpu)}


def source_rows(files, dim):
    size = record_dtype(dim).itemsize
    if any(Path(path).stat().st_size % size for path in files):
        raise ValueError("Truncated original vector records")
    count = sum(Path(path).stat().st_size // size for path in files)
    if not 0 < count <= np.iinfo(np.int32).max:
        raise ValueError("Expected between 1 and 2^31-1 source rows")
    return count


def adopt_build(output_dir, files, heads_path, labels_path, training, assignment, *, dim=384, threads=8,
                batch_size=250000, sample_size=2500000, seed=42, max_iter=20, gpu=True):
    """Retain an already verified model/map so export needs no retraining or staging."""
    files = input_files(files)
    count = source_rows(files, dim)
    clusters, head_dim = matrix_shape(heads_path)
    if head_dim != dim or Path(labels_path).stat().st_size != count * 4:
        raise ValueError("Cached centroids or assignments do not match source dimensions/rows")
    config = parameters(clusters, dim, batch_size, threads, sample_size, seed, max_iter, gpu)
    output = Path(output_dir)
    if output.is_symlink() or (output / "native").is_symlink() or (output / "build.json").exists():
        raise ValueError("Refusing an existing build checkpoint or symlink")
    native = output / "native"
    native.mkdir(parents=True, exist_ok=True)
    for source, name in ((Path(heads_path), "heads.f32"), (Path(labels_path), "assignments.i32")):
        target = native / name
        if source.resolve() != target.resolve():
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)
            if artifact(source)["sha256"] != artifact(target)["sha256"]:
                raise ValueError("Cached artifact changed during copy")
    labels = np.memmap(native / "assignments.i32", dtype="<i4", mode="r", shape=(count,))
    for first in range(0, count, batch_size):
        block = labels[first:first + batch_size]
        if (block < 0).any() or (block >= clusters).any():
            raise ValueError("Cached assignments contain invalid centroid IDs")
    del labels
    state = {"schema_version": 1, "method": "balanced_l2", "inputs": [snapshot(path) for path in files],
             "rows": count, "parameters": config, "completed": ["vectors", "heads", "assignments"],
             "seconds": {}, "training": training, "assignment": assignment,
             "artifacts": {"heads": artifact(native / "heads.f32"), "assignments": artifact(native / "assignments.i32")}}
    write_json(native / "training.json", training)
    write_json(output / "build.json", state)
    return state


def build(paths="vectors", output_dir="clusters_balanced_l2_64k", clusters=64000, *, dim=384,
          threads=8, batch_size=250000, sample_size=2500000, seed=42, max_iter=20,
          gpu=True, resume=False, export_only=False, centroid_brotli=False, heads_from=None):
    config = parameters(clusters, dim, batch_size, threads, sample_size, seed, max_iter, gpu)
    if centroid_brotli:
        config["centroid_brotli"] = True
    if heads_from is not None:
        heads_from = Path(heads_from).resolve()
        config["heads_from"] = str(heads_from)
    files = input_files(paths)
    count = source_rows(files, dim)
    if not 2 <= clusters <= min(count, sample_size):
        raise ValueError("Cluster count must be between two and the training sample size")
    inputs = [snapshot(path) for path in files]
    output = Path(output_dir).resolve()
    state_path = output / "build.json"
    if Path(output_dir).is_symlink():
        raise ValueError("Build output must not be a symlink")
    if resume:
        state = json.loads(state_path.read_text())
        if (state.get("schema_version") != 1 or state.get("method") != "balanced_l2"
                or state["inputs"] != inputs or state["parameters"] != config or state["rows"] != count):
            raise ValueError("Resume source files or clustering parameters differ")
    else:
        if export_only:
            raise ValueError("Export-only requires an existing build and --resume")
        output.mkdir(parents=True, exist_ok=False)
        state = {"schema_version": 1, "method": "balanced_l2", "inputs": inputs, "rows": count,
                 "parameters": config, "completed": [], "seconds": {}, "artifacts": {}}
        write_json(state_path, state)
    native, work = output / "native", output / "_work"
    if native.is_symlink() or work.is_symlink():
        raise ValueError("Owned build directories must not be symlinks")
    native.mkdir(exist_ok=True)

    def unchanged():
        if inputs != [snapshot(path) for path in files]:
            raise ValueError("Original vectors changed during the build")

    if heads_from is not None and "heads" not in state["completed"]:
        if (heads_from.name != "heads.f32" or heads_from.parent.name != "native"
                or heads_from.parent.parent.name != f"h{clusters}"):
            raise ValueError("Imported heads must come from the matching centroid comparison case")
        comparison = json.loads((heads_from.parents[2] / "results.json").read_text())
        training = json.loads(heads_from.with_name("training.json").read_text())
        expected_rows = min(count, sample_size)
        cases = [case for case in comparison.get("completed", []) if case.get("head_count") == clusters]
        params = training.get("params", {})
        if (comparison.get("schema_version") != 1 or comparison.get("status") != "complete"
                or comparison.get("inputs") != inputs or comparison.get("source_rows") != count
                or comparison.get("dimensions") != dim or comparison.get("sample_rows") != expected_rows
                or comparison.get("sample_seed") != seed or comparison.get("max_iter") != max_iter
                or len(cases) != 1 or cases[0].get("centroid_method") != "lvq6"
                or cases[0].get("metric") != "l2" or training.get("method") != "balanced_l2"
                or training.get("metric") != "l2" or training.get("head_count") != clusters
                or training.get("dimensions") != dim or training.get("sample_rows") != expected_rows
                or training.get("sample") != comparison.get("sample")
                or params.get("n_clusters") != clusters or params.get("metric") != "sqeuclidean"
                or params.get("hierarchical") is not True or params.get("hierarchical_n_iters") != max_iter
                or read_matrix(heads_from).shape != (clusters, dim)):
            raise ValueError("Imported centroid training provenance differs from this build")
        origin = artifact(heads_from)
        if training.get("heads") != origin:
            raise ValueError("Imported centroid artifact differs from its training checkpoint")
        target = native / "heads.f32"
        temporary = target.with_name(target.name + ".tmp")
        if target.is_symlink() or temporary.is_symlink():
            raise ValueError("Imported centroid targets must not be symlinks")
        shutil.copy2(heads_from, temporary)
        if artifact(heads_from) != origin or artifact(temporary)["sha256"] != origin["sha256"]:
            raise ValueError("Imported centroid artifact changed during copy")
        temporary.replace(target)
        unchanged()
        imported = artifact(target)
        training = {**training, "source_heads": origin, "heads": imported}
        write_json(native / "training.json", training)
        state["training"] = training
        state["artifacts"]["heads"] = imported
        state["completed"].append("heads")
        write_json(state_path, state)

    for name in ("heads", "assignments"):
        if name in state["completed"]:
            check_artifact(state["artifacts"][name])
    if "centroids" in state["completed"]:
        for name in ("centroid_gzip", "centroid_brotli"):
            check_artifact(state["artifacts"][name])
    if export_only and "assignments" not in state["completed"]:
        raise ValueError("Export-only requires completed full-corpus assignments")

    def step(name, action):
        if name in state["completed"]:
            return
        print(f"Starting {name}", flush=True)
        started = time.perf_counter()
        action()
        unchanged()
        state["seconds"][name] = time.perf_counter() - started
        state["completed"].append(name)
        write_json(state_path, state)
        print(f"Finished {name} in {state['seconds'][name]:.2f}s", flush=True)

    sample = work / "sample.f32"

    def clean_work():
        if not work.exists():
            return
        paths = list(work.iterdir())
        if any(path.is_symlink() or not path.is_file()
               or path.name not in {"sample.f32", "sample.f32.tmp", "vectors.f32"} for path in paths):
            raise ValueError("Refusing to remove unrelated build staging files")
        for path in paths:
            path.unlink()
        work.rmdir()

    def prepare():
        work.mkdir(exist_ok=True)
        size = min(count, sample_size)
        require_disk_space(output, size * dim * 4 + 8)
        temporary = sample.with_name(sample.name + ".tmp")
        for path in (sample, temporary):
            if path.is_symlink():
                raise ValueError("Staging files must not be symlinks")
            path.unlink(missing_ok=True)
        selected = np.sort(np.random.default_rng(seed).choice(count, size, replace=False))
        offset = 0
        with temporary.open("xb") as stream:
            stream.write(struct.pack("<ii", size, dim))
            for records in iter_record_batches(files, dim, min(batch_size, 65536)):
                left, right = np.searchsorted(selected, [offset, offset + len(records)])
                if left < right:
                    values = records["vec"][selected[left:right] - offset].astype(np.float32)
                    norms = np.linalg.norm(values, axis=1)
                    if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
                        raise ValueError("Source vectors must be finite and nonzero")
                    (values / norms[:, None]).astype("<f4", copy=False).tofile(stream)
                offset += len(records)
        if offset != count:
            raise ValueError("Original vector row count changed during sampling")
        unchanged()
        temporary.replace(sample)
        state["staging"] = {"sample": snapshot(sample)}

    if "heads" not in state["completed"]:
        step("vectors", prepare)

        def centers():
            if snapshot(sample) != state["staging"]["sample"]:
                raise ValueError("Training sample changed")
            state["training"] = train(sample, native, clusters, max_iter)
            state["artifacts"]["heads"] = artifact(native / "heads.f32")

        step("heads", centers)

    check_artifact(state["artifacts"]["heads"])
    clean_work()

    if centroid_brotli:
        def centroids():
            stats = export_centroids(output, native / "heads.f32")
            stats.update(export_centroids_brotli(output))
            state["centroid_export"] = stats
            state["artifacts"]["centroid_gzip"] = artifact(output / stats["centroid_index"])
            state["artifacts"]["centroid_brotli"] = artifact(output / stats["centroid_brotli_index"])

        step("centroids", centroids)

    if "assignments" not in state["completed"]:
        def assignments():
            require_disk_space(output, count * 4)
            labels = native / "assignments.i32"
            if labels.is_symlink() or labels.with_name(labels.name + ".tmp").is_symlink():
                raise ValueError("Assignment files must not be symlinks")
            labels.unlink(missing_ok=True)
            labels.with_name(labels.name + ".tmp").unlink(missing_ok=True)
            vectors = _NormalizedRecords(files, count, dim, batch_size)
            try:
                state["assignment"] = assign_vectors(vectors, native / "heads.f32", labels,
                                                      batch_size=batch_size, gpu=gpu,
                                                      validate_inputs=unchanged)
            finally:
                vectors.close()
            state["artifacts"]["assignments"] = artifact(labels)

        step("assignments", assignments)

    def export():
        stats = export_postings(files, native / "assignments.i32", clusters, output,
                                dim=dim, batch_size=batch_size, threads=threads)
        stats.update(state["centroid_export"] if centroid_brotli
                     else export_centroids(output, native / "heads.f32"))
        stats.update(method="balanced_l2", dimensions=dim, requested_clusters=clusters,
                     training_vectors=min(count, sample_size), sampling_seed=seed,
                     stage_seconds=state["seconds"], metric="l2")
        write_json(output / "stats.json", stats)

    step("export", export)
    stats = json.loads((output / "stats.json").read_text())
    stats["stage_seconds"] = state["seconds"]
    write_json(output / "stats.json", stats)
    return stats
