"""Exact client-style centroid routing, with bounded BLAS candidate screening."""

import numpy as np

GPU_SCORE_BYTES = 1024**3

_LITERAL_SOURCE = r'''
__device__ double widen(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0) {
        // Turing's float-to-double conversion flushes float32 subnormals.
        double magnitude = __dmul_rn((double)(bits & 0x7fffffu), 1.401298464324817e-45);
        return bits >> 31 ? -magnitude : magnitude;
    }
    return (double)value;
}
extern "C" __global__ void literal_scores(
    const float* queries, const float* heads,
    const long long* rows, const long long* ids,
    int dimensions, unsigned long long count, double* scores) {
    for (unsigned long long pair = blockIdx.x * blockDim.x + threadIdx.x;
         pair < count; pair += gridDim.x * blockDim.x) {
        const float* query = queries + rows[pair] * dimensions;
        const float* head = heads + ids[pair] * dimensions;
        double score = 0.;
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            double product = __dmul_rn(widen(query[dimension]), widen(head[dimension]));
            score = __dadd_rn(score, product);
        }
        scores[pair] = score;
    }
}
extern "C" __global__ void literal_scores_l2(
    const float* queries, const float* heads, const double* bias,
    const long long* rows, const long long* ids,
    int dimensions, unsigned long long count, double* scores) {
    for (unsigned long long pair = blockIdx.x * blockDim.x + threadIdx.x;
         pair < count; pair += gridDim.x * blockDim.x) {
        const float* query = queries + rows[pair] * dimensions;
        const float* head = heads + ids[pair] * dimensions;
        double score = 0.;
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            double product = __dmul_rn(widen(query[dimension]), widen(head[dimension]));
            score = __dadd_rn(score, product);
        }
        scores[pair] = __dadd_rn(score, bias[ids[pair]]);
    }
}
'''


def _margin(batch, max_component, screen_dtype, max_bias=0.):
    dim = batch.shape[1]
    eps = np.finfo(screen_dtype).eps
    if 2 * dim * eps >= .5:
        raise ValueError("Dimension is too large for the screening error bound")
    magnitude = np.abs(batch.astype(np.float64)).sum(axis=1) * max_component
    # Each score is within gamma(2d)*sum|qh| of the real dot product.
    # Twice the combined BLAS/client bound preserves every possible top-k tie.
    margin = 8 * dim * (eps + np.finfo(np.float64).eps) * magnitude
    if max_bias:
        # Include rounded bias storage/addition and the corresponding exact sum.
        margin += 8 * (eps + np.finfo(np.float64).eps) * (magnitude + max_bias)
    if screen_dtype == np.float32:
        margin += 4 * dim * np.finfo(np.float32).tiny * (
            1 + max_component + np.abs(batch).max(axis=1))
    return margin


def _select(rows, ids, scores, query_count, probes):
    counts = np.bincount(rows, minlength=query_count)
    if (counts < probes).any():
        raise ValueError("Screening failed to retain enough candidates")
    order = np.lexsort((ids, -scores, rows))
    starts = np.cumsum(counts) - counts
    chosen = order[starts[:, None] + np.arange(probes)]
    return ids[chosen].astype(np.int32), scores[chosen]


def _rescore(heads, batch, rows, ids, probes, bias=None):
    scores = np.zeros(len(ids), dtype=np.float64)
    for first in range(0, len(ids), 65_536):
        stop = first + 65_536
        for dimension in range(heads.shape[1]):
            scores[first:stop] += batch[rows[first:stop], dimension] * heads[ids[first:stop], dimension]
    if bias is not None:
        scores += bias[ids]
    return _select(rows, ids, scores, len(batch), probes)


def _deduplicate(heads, probes):
    unique, first, inverse = np.unique(heads, axis=0, return_index=True, return_inverse=True)
    if len(unique) == len(heads):
        return heads, None
    order = np.argsort(first)
    groups = np.empty(len(unique), dtype=np.int32)
    groups[order] = np.arange(len(unique))
    source_ids = np.full((len(unique), probes), -1, dtype=np.int32)
    counts = np.zeros(len(unique), dtype=np.int32)
    for source, group in enumerate(groups[inverse]):
        if counts[group] < probes:
            source_ids[group, counts[group]] = source
            counts[group] += 1
    return np.ascontiguousarray(unique[order]), source_ids


def _expand(routes, scores, source_ids, probes):
    ids = source_ids[routes].reshape(len(routes), -1)
    scores = np.repeat(scores, probes, axis=1)
    scores[ids < 0] = -np.inf
    order = np.lexsort((ids, -scores), axis=1)[:, :probes]
    return np.take_along_axis(ids, order, axis=1)


def route_queries(decoded, queries, probes=10, query_block=256, gpu=True, metric="l2"):
    """Return exact IDs using literal float64 dot sums, optionally with L2 norm bias."""
    heads32 = np.asarray(decoded, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if (heads32.ndim != 2 or queries.ndim != 2 or not all(heads32.shape)
            or heads32.shape[1] != queries.shape[1]
            or not np.isfinite(heads32).all() or not np.isfinite(queries).all()):
        raise ValueError("Heads and queries must be finite compatible matrices")
    if (isinstance(probes, bool) or not isinstance(probes, int) or probes <= 0
            or isinstance(query_block, bool) or not isinstance(query_block, int) or query_block <= 0):
        raise ValueError("Probes and query block must be positive integers")
    if metric not in ("ip", "l2"):
        raise ValueError("Routing metric must be ip or l2")
    probes = min(probes, len(heads32))
    result = np.empty((len(queries), probes), dtype=np.int32)
    if not len(queries):
        return result
    # Repeated PQ prototypes need one score, while preserving original-ID ties.
    heads32, source_ids = _deduplicate(heads32, probes)
    heads = heads32 if gpu else heads32.astype(np.float64)
    keep = min(probes, len(heads))
    max_component = float(np.abs(heads).max())
    bias = None
    max_bias = 0.
    if metric == "l2":
        bias = np.zeros(len(heads), dtype=np.float64)
        for dimension in range(heads.shape[1]):
            values = heads32[:, dimension].astype(np.float64)
            bias += values * values
        bias *= -.5
        max_bias = float(np.abs(bias).max())
    if gpu:
        query_block = min(query_block, max(1, GPU_SCORE_BYTES // (len(heads) * 4)))
        import cupy as cp
        from cupy_backends.cuda.libs import cublas
        handle = cp.cuda.device.get_cublas_handle()
        old_math = cublas.getMathMode(handle)
        old_pointer = cublas.getPointerMode(handle)
        # CUDA's CUBLAS_PEDANTIC_MATH=2 prevents reduced-precision input math.
        cublas.setMathMode(handle, 2)
        cublas.setPointerMode(handle, 0)
        alpha, beta = np.array(1., dtype=np.float32), np.array(0., dtype=np.float32)
    try:
        if gpu:
            device_heads = cp.asarray(np.ascontiguousarray(heads32))
            if bias is not None:
                device_bias = cp.asarray(bias)
                screen_bias = device_bias.astype(cp.float32)
            literal_scores = cp.RawKernel(_LITERAL_SOURCE, "literal_scores_l2" if bias is not None else "literal_scores",
                                          options=("--fmad=false", "--ftz=false"))
        for first in range(0, len(queries), query_block):
            batch32 = np.ascontiguousarray(queries[first:first + query_block])
            batch = batch32.astype(np.float64)
            if gpu:
                device_batch = cp.asarray(batch32)
                approximate = cp.empty((len(batch), len(heads)), dtype=cp.float32)
                cublas.sgemm(handle, 1, 0, len(heads), len(batch), heads.shape[1],
                             alpha.ctypes.data, device_heads.data.ptr, heads.shape[1],
                             device_batch.data.ptr, heads.shape[1], beta.ctypes.data,
                             approximate.data.ptr, len(heads))
                if bias is not None:
                    approximate += screen_bias
                scratch = approximate.copy()
                query_ids = cp.arange(len(batch))
                for rank in range(keep):
                    maxima = cp.argmax(scratch, axis=1)
                    cutoff = scratch[query_ids, maxima]
                    if rank + 1 < keep:
                        scratch[query_ids, maxima] = -cp.inf
                margin = _margin(batch32, max_component, np.float32, max_bias)
                threshold = cp.nextafter(cutoff - cp.asarray(margin, dtype=cp.float64), -cp.inf)
                rows, ids = cp.nonzero(approximate >= threshold[:, None])
                rows, ids = cp.ascontiguousarray(rows), cp.ascontiguousarray(ids)
                scores = cp.empty(len(ids), dtype=cp.float64)
                arguments = (device_batch, device_heads)
                if bias is not None:
                    arguments += (device_bias,)
                literal_scores(((len(ids) + 127) // 128,), (128,),
                               arguments + (rows, ids, np.int32(heads.shape[1]), np.uint64(len(ids)), scores))
                scores = cp.asnumpy(scores)
                rows, ids = cp.asnumpy(rows), cp.asnumpy(ids)
                routes, scores = _select(rows, ids, scores, len(batch), keep)
            else:
                approximate = batch @ heads.T
                if bias is not None:
                    approximate += bias
                cutoff = np.partition(approximate, len(heads) - keep, axis=1)[:, len(heads) - keep]
                margin = _margin(batch, max_component, np.float64, max_bias)
                rows, ids = np.nonzero(approximate >= (cutoff - margin)[:, None])
                routes, scores = _rescore(heads, batch, rows, ids, keep, bias)
            if source_ids is not None:
                routes = _expand(routes, scores, source_ids, probes)
            result[first:first + len(batch)] = routes
            if gpu:
                del approximate, scratch, device_batch
    finally:
        if gpu:
            cublas.setMathMode(handle, old_math)
            cublas.setPointerMode(handle, old_pointer)
    return result
