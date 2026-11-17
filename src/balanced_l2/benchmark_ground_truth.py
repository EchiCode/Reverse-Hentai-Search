"""Exact GPU nearest neighbors and SHA-verified noisy replay query caches."""

import hashlib
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUPY_TF32"] = "0"
import numpy as np


VECTOR_BLOCK = 65_536
QUERY_BLOCK = 512


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def report(value):
    print(json.dumps(value, default=json_scalar), flush=True)


def exact_gpu(vectors, queries, vector_block=VECTOR_BLOCK, query_block=QUERY_BLOCK, progress=False,
              source_digest=None):
    import cupy as cp
    q = cp.asarray(queries, dtype=cp.float32, order="C")
    best = cp.full(len(queries), -cp.inf, dtype=cp.float32)
    ids = cp.full(len(queries), -1, dtype=cp.int64)
    started = time.perf_counter()
    # Reading vectors in the outer loop scans the disk-backed corpus only once.
    for first in range(0, len(vectors), vector_block):
        host_block = vectors[first:first + vector_block]
        block = cp.asarray(host_block, dtype=cp.float32, order="C")
        if source_digest is not None:
            source_digest.update(memoryview(host_block).cast("B"))
        for q_first in range(0, len(queries), query_block):
            q_last = min(q_first + query_block, len(queries))
            scores = q[q_first:q_last] @ block.T
            local = cp.argmax(scores, axis=1)
            values = scores[cp.arange(q_last - q_first), local]
            better = values > best[q_first:q_last]
            # Strict improvement preserves the earliest global row when scores tie.
            best[q_first:q_last] = cp.where(better, values, best[q_first:q_last])
            ids[q_first:q_last] = cp.where(better, local.astype(cp.int64) + first, ids[q_first:q_last])
        if progress and (first == 0 or (first // vector_block + 1) % 16 == 0
                         or first + len(block) == len(vectors)):
            cp.cuda.get_current_stream().synchronize()
            done = first + len(block)
            elapsed = time.perf_counter() - started
            free, total = cp.cuda.runtime.memGetInfo()
            report({"event": "progress", "rows": done, "total": len(vectors),
                    "seconds": elapsed, "eta_seconds": elapsed * (len(vectors) - done) / done,
                    "gpu_used_mib": (total - free) / 2**20})
    result, values = cp.asnumpy(ids), cp.asnumpy(best)
    if not ((result >= 0) & (result < len(vectors))).all() or not np.isfinite(values).all():
        raise ValueError("Exact scan produced invalid nearest neighbors")
    return result[:, None], values


def check(vectors, queries, vector_block, query_block, label):
    expected_scores = queries @ vectors.T
    expected = np.argmax(expected_scores, axis=1)[:, None]
    actual, values = exact_gpu(vectors, queries, vector_block, query_block)
    np.testing.assert_array_equal(actual, expected, err_msg=label)
    np.testing.assert_allclose(values, expected_scores[np.arange(len(queries)), expected[:, 0]],
                               rtol=2e-6, atol=2e-6, err_msg=label)
    report({"event": "cpu_validation", "label": label, "rows": len(vectors),
            "queries": len(queries), "matches": len(queries)})


def query_sets(vectors, query_count, noises):
    if not 0 < query_count <= len(vectors):
        raise ValueError("Query count must be positive and no larger than the corpus")
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(vectors), query_count, replace=False)
    base = np.array(vectors[sampled])
    directions = rng.standard_normal(base.shape)
    queries = {}
    for noise in noises:
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("Noise standard deviations must be finite and nonnegative")
        batch = base.copy()
        batch += (directions * noise).astype(np.float32)
        norms = np.linalg.norm(batch, axis=1)
        if not np.isfinite(norms).all() or not (norms > 0).all():
            raise ValueError("Sampled vectors produced invalid queries")
        batch /= norms[:, None]
        queries[float(noise)] = batch
    return sampled, queries


def input_digest(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def benchmark_noises(source, output_dir, query_count=100000, noises=(.005, .01, .025),
                     inputs=None, reuse_dir=None):
    """Reuse exact truth only after corpus SHA256 and every seeded query bit agree."""
    from balanced_l2.common import artifact, matrix_shape, sha256, snapshot, write_json
    import shutil

    if Path(output_dir).is_symlink():
        raise ValueError("Truth output must not be a symlink")
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = list(inputs or [])
    if not inputs or any(snapshot(item["path"]) != item for item in inputs):
        raise ValueError("Exact truth requires unchanged original input snapshots")
    count, dim = matrix_shape(source)
    before = snapshot(source)
    digest, inputs_sha = sha256(source), input_digest(inputs)
    vectors = np.memmap(source, dtype="<f4", mode="r", offset=8, shape=(count, dim))
    noises = tuple(dict.fromkeys(float(value) for value in noises))
    if not noises or any(not np.isfinite(value) or value < 0 for value in noises):
        raise ValueError("Noise levels must be finite and nonnegative")
    sampled, queries = query_sets(vectors, query_count, noises)
    paths, pending, reused = {}, [], []

    def valid(path, noise):
        with np.load(path, allow_pickle=False) as cache:
            settings = json.loads(cache["settings"].item())
            truth, cached_queries = cache["truth"], cache["queries"]
            return (cache["normalized_corpus_sha256"].item() == digest
                    and cache["source_inputs_sha256"].item() == inputs_sha
                    and settings.get("seed") == 42 and settings.get("noise_std") == noise
                    and settings.get("count") == query_count
                    and settings.get("ground_truth") == "exact nearest neighbor over normalized source vectors"
                    and cached_queries.dtype == np.dtype("float32")
                    and np.array_equal(cached_queries.view(np.uint32), queries[noise].view(np.uint32))
                    and np.array_equal(cache["sampled_rows"], sampled)
                    and cache["query_sha256"].item() == hashlib.sha256(queries[noise].tobytes()).hexdigest()
                    and truth.shape == (query_count, 1) and truth.dtype == np.dtype("int64")
                    and ((truth >= 0) & (truth < count)).all())

    for noise in noises:
        path = output_dir / f"query_truth_noise_{noise:g}.npz"
        paths[noise] = path
        if path.is_symlink():
            raise ValueError("Truth cache must not be a symlink")
        if path.exists():
            if not valid(path, noise):
                raise ValueError(f"Saved truth differs from source or seeded queries: {path}")
            reused.append(noise)
            continue
        original = Path(reuse_dir) / path.name if reuse_dir is not None else None
        if original is not None and original.exists() and valid(original, noise):
            temporary = path.with_suffix(".tmp")
            shutil.copy2(original, temporary)
            if sha256(temporary) != sha256(original):
                raise ValueError("Truth cache changed while copying")
            temporary.replace(path)
            reused.append(noise)
        else:
            pending.append(noise)
    if pending:
        combined = np.concatenate([queries[noise] for noise in pending])
        nearest, _ = exact_gpu(vectors, combined, progress=True)
        for index, noise in enumerate(pending):
            truth = nearest[index * query_count:(index + 1) * query_count]
            settings = {"count": query_count, "seed": 42, "noise_std": noise,
                        "ground_truth": "exact nearest neighbor over normalized source vectors",
                        "limit": "Noisy replay queries, not independent semantic queries"}
            temporary = paths[noise].with_suffix(".tmp")
            temporary.unlink(missing_ok=True)
            with temporary.open("xb") as stream:
                np.savez(stream, queries=queries[noise], truth=truth, sampled_rows=sampled,
                         settings=json.dumps(settings, sort_keys=True), normalized_corpus_sha256=digest,
                         source_inputs_sha256=inputs_sha,
                         query_sha256=hashlib.sha256(queries[noise].tobytes()).hexdigest())
            temporary.replace(paths[noise])
    if snapshot(source) != before or any(snapshot(item["path"]) != item for item in inputs):
        raise ValueError("Source changed during exact-truth preparation")
    write_json(output_dir / "ground_truth.json", {"normalized_corpus_sha256": digest,
               "source_inputs_sha256": inputs_sha, "source_rows": count, "dimensions": dim,
               "queries_per_noise": query_count, "noise": list(noises), "computed_noise": pending,
               "reused_noise": reused, "caches": {str(noise): artifact(path) for noise, path in paths.items()}})
    return paths
