"""Bounded cosine winners with the current client's Float64 arithmetic."""

import numpy as np

from balanced_l2.quantized_routing import _LITERAL_SOURCE, _margin


SCORE_BYTES = 128 * 1024 ** 2

_COSINE_SOURCE = _LITERAL_SOURCE + r'''
extern "C" __global__ void posting_cosine(
    const float* queries, const float* vectors, const double* inverse_norms,
    const long long* rows, const long long* ids,
    int dimensions, unsigned long long count, double* scores) {
    for (unsigned long long pair = blockIdx.x * blockDim.x + threadIdx.x;
         pair < count; pair += gridDim.x * blockDim.x) {
        const float* query = queries + rows[pair] * dimensions;
        const float* vector = vectors + ids[pair] * dimensions;
        double score = 0.;
        int dimension = 0;
        for (; dimension + 3 < dimensions; dimension += 4) {
            double group = __dmul_rn(widen(query[dimension]), widen(vector[dimension]));
            group = __dadd_rn(group, __dmul_rn(widen(query[dimension+1]), widen(vector[dimension+1])));
            group = __dadd_rn(group, __dmul_rn(widen(query[dimension+2]), widen(vector[dimension+2])));
            group = __dadd_rn(group, __dmul_rn(widen(query[dimension+3]), widen(vector[dimension+3])));
            score = __dadd_rn(score, group);
        }
        for (; dimension < dimensions; ++dimension) {
            score = __dadd_rn(score, __dmul_rn(widen(query[dimension]), widen(vector[dimension])));
        }
        scores[pair] = __dmul_rn(score, inverse_norms[ids[pair]]);
    }
}
'''


def _inputs(queries, vectors, query_batch, vector_batch):
    with np.errstate(over="ignore", invalid="ignore"):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if (queries.ndim != 2 or vectors.ndim != 2 or not queries.shape[1]
            or queries.shape[1] != vectors.shape[1]
            or not np.isfinite(queries).all() or not np.isfinite(vectors).all()):
        raise ValueError("Queries and vectors must be finite matrices with equal dimensions")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (query_batch, vector_batch)):
        raise ValueError("Scoring batch sizes must be positive integers")
    # Includes worst-case screening, every candidate pair, and literal rescore buffers.
    cells = max(1, SCORE_BYTES // 64)
    query_batch = min(query_batch, max(1, len(queries)), cells)
    vector_batch = min(vector_batch, max(1, len(vectors)), max(1, cells // query_batch))
    return queries, vectors, query_batch, vector_batch


def _inverse_norms(vectors):
    squared = np.zeros(len(vectors), dtype=np.float64)
    for dimension in range(vectors.shape[1]):
        values = vectors[:, dimension].astype(np.float64)
        squared += values * values
    if not np.isfinite(squared).all() or (squared <= 0).any():
        raise ValueError("Posting vectors must have finite positive norms")
    # decodeSearchCluster caches these in Float32Array before the Float64 dot product.
    with np.errstate(over="ignore"):
        inverse = (1. / np.sqrt(squared)).astype(np.float32).astype(np.float64)
    if not np.isfinite(inverse).all() or (inverse <= 0).any():
        raise ValueError("Posting inverse norms must fit finite positive Float32 values")
    return inverse


def _cpu_winners(queries, vectors, inverse_norms):
    queries, vectors = queries.astype(np.float64), vectors.astype(np.float64)
    scores = np.zeros((len(queries), len(vectors)), dtype=np.float64)
    dimension = 0
    # Match vectorDot's four-product groups; a single sequential sum differs at ties.
    while dimension + 3 < queries.shape[1]:
        group = queries[:, dimension, None] * vectors[None, :, dimension]
        for offset in (1, 2, 3):
            group += queries[:, dimension + offset, None] * vectors[None, :, dimension + offset]
        scores += group
        dimension += 4
    for dimension in range(dimension, queries.shape[1]):
        scores += queries[:, dimension, None] * vectors[None, :, dimension]
    scores *= inverse_norms[None, :]
    ids = np.argmax(scores, axis=1)
    return ids, scores[np.arange(len(queries)), ids]


def _gpu_winners(queries, vectors, inverse_norms, handle, kernel):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    device_queries, device_vectors = cp.asarray(queries), cp.asarray(vectors)
    device_inverse = cp.asarray(inverse_norms)
    approximate = cp.empty((len(queries), len(vectors)), dtype=np.float32)
    alpha, beta = np.array(1., np.float32), np.array(0., np.float32)
    cublas.sgemm(handle, 1, 0, len(vectors), len(queries), queries.shape[1],
                 alpha.ctypes.data, device_vectors.data.ptr, queries.shape[1],
                 device_queries.data.ptr, queries.shape[1], beta.ctypes.data,
                 approximate.data.ptr, len(vectors))
    screened = approximate.astype(cp.float64)
    screened *= device_inverse[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        margin = _margin(queries, float(np.abs(vectors).max()), np.float32) * inverse_norms.max()
    cutoff = cp.max(screened, axis=1)
    threshold = cp.nextafter(cutoff - cp.asarray(margin), -cp.inf)
    # Overflow in the float32 screen cannot discard any candidates from that query.
    invalid = ~cp.isfinite(screened).all(axis=1)
    rows, ids = cp.nonzero((screened >= threshold[:, None]) | invalid[:, None])
    rows, ids = cp.ascontiguousarray(rows), cp.ascontiguousarray(ids)
    scores = cp.empty(len(ids), dtype=cp.float64)
    kernel(((len(ids) + 127) // 128,), (128,),
           (device_queries, device_vectors, device_inverse, rows, ids,
            np.int32(queries.shape[1]), np.uint64(len(ids)), scores))
    rows, ids, scores = cp.asnumpy(rows), cp.asnumpy(ids), cp.asnumpy(scores)
    counts = np.bincount(rows, minlength=len(queries))
    if (counts == 0).any():
        raise ValueError("Scoring screen discarded every candidate")
    order = np.lexsort((ids, -scores, rows))
    selected = order[np.cumsum(counts) - counts]
    return ids[selected], scores[selected]


def best_matches(queries, vectors, *, gpu=True, query_batch=256, vector_batch=8192):
    """Return each query's earliest best row and browser-style cosine score.

    Queries are already normalized by the caller. Posting vectors retain their
    decoded Float32 values; sequential Float64 norms feed the client's Float32
    inverse-norm cache before the Float64 dot product.
    """
    queries, vectors, query_batch, vector_batch = _inputs(queries, vectors, query_batch, vector_batch)
    best_rows = np.full(len(queries), -1, dtype=np.int64)
    best_scores = np.full(len(queries), -np.inf, dtype=np.float64)
    if not len(queries) or not len(vectors):
        return best_rows, best_scores
    inverse_norms = _inverse_norms(vectors)
    if gpu:
        import cupy as cp
        from cupy_backends.cuda.libs import cublas
        handle = cp.cuda.device.get_cublas_handle()
        old_math, old_pointer = cublas.getMathMode(handle), cublas.getPointerMode(handle)
    try:
        if gpu:
            cublas.setMathMode(handle, 2)
            cublas.setPointerMode(handle, 0)
            kernel = cp.RawKernel(_COSINE_SOURCE, 'posting_cosine', options=('--fmad=false', '--ftz=false'))
        for first in range(0, len(queries), query_batch):
            query = queries[first:first + query_batch]
            for start in range(0, len(vectors), vector_batch):
                vector = vectors[start:start + vector_batch]
                norms = inverse_norms[start:start + vector_batch]
                ids, scores = (_gpu_winners(query, vector, norms, handle, kernel) if gpu
                               else _cpu_winners(query, vector, norms))
                better = scores > best_scores[first:first + len(query)]
                best_rows[first:first + len(query)][better] = ids[better] + start
                best_scores[first:first + len(query)][better] = scores[better]
    finally:
        if gpu:
            cublas.setMathMode(handle, old_math)
            cublas.setPointerMode(handle, old_pointer)
    return best_rows, best_scores
