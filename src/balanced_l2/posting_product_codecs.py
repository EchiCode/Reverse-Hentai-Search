"""Trained, globally shared Faiss codecs for posting-vector comparisons."""

import hashlib
import json
import re
import struct

import faiss
import numpy as np


MAGIC = b"FPCODEC\0"
MAX_DISTANCE_BYTES = 512 * 1024 ** 2
PQ_PARAMETERS = tuple((m, bits) for m in (16, 32, 48, 64, 96, 128) for bits in range(2, 9)) + tuple(
    (m, bits) for m in (8, 12, 24, 192, 384) for bits in (4, 6, 8))
PRODUCT_METHODS = tuple(f"pq{m}x{bits}" for m, bits in PQ_PARAMETERS) + (
    "opq32x8", "opq64x8", "opq96x8", "rq4x6", "rq8x6", "lsq4x6", "lsq8x6")
SCALAR_PARAMETERS = {
    **{f"sq{bits}": (bits, "minmax", 0.) for bits in (4, 6, 8)},
    **{f"sq{bits}_uniform": (bits, "uniform", 0.) for bits in (4, 8)},
    **{f"sq{bits}_meanstd{factor}": (bits, "meanstd", float(factor))
       for bits in (4, 6, 8) for factor in (2, 3, 4)},
    **{f"sq{bits}_quantile{fraction:g}": (bits, "quantile", fraction)
       for bits in (4, 6, 8) for fraction in (.001, .005, .01)},
    **{f"sq{bits}_optim": (bits, "optim", 0.) for bits in (4, 6, 8)},
}
SCALAR_METHODS = tuple(SCALAR_PARAMETERS)
METHODS = SCALAR_METHODS + PRODUCT_METHODS + ("bf16",)


def _vectors(values, dimensions=None):
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.ascontiguousarray(values, dtype=np.float32)
    if (values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all()
            or dimensions is not None and values.shape[1] != dimensions):
        raise ValueError("Expected finite vectors with the codec's dimensions")
    return values


def _check_state(index):
    if not index.is_trained or index.ntotal:
        raise ValueError("Codec must be trained and contain no stored vectors")
    if isinstance(index, faiss.IndexPreTransform):
        for i in range(index.chain.size()):
            transform = faiss.downcast_VectorTransform(index.chain.at(i))
            for field in ("A", "b"):
                if hasattr(transform, field) and not np.isfinite(faiss.vector_to_array(getattr(transform, field))).all():
                    raise ValueError("Non-finite codec transform")
        _check_state(faiss.downcast_index(index.index))
    for field, weights in (("sq", "trained"), ("pq", "centroids"), ("rq", "codebooks"), ("lsq", "codebooks")):
        if hasattr(index, field) and not np.isfinite(faiss.vector_to_array(getattr(getattr(index, field), weights))).all():
            raise ValueError("Non-finite trained codec state")


class FaissCodec:
    def __init__(self, name, index, threads=8, references=()):
        _check_state(index)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("Threads must be a positive integer")
        self.name = name
        self.index = index
        self.dimensions = index.d
        self.code_size = index.sa_code_size()
        self.threads = threads
        self._references = references

    def encode(self, vectors):
        vectors = _vectors(vectors, self.dimensions)
        faiss.omp_set_num_threads(self.threads)
        if not len(vectors):
            return np.empty((0, self.code_size), dtype=np.uint8)
        return self.index.sa_encode(vectors)

    def decode(self, codes):
        codes = np.asarray(codes)
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != self.code_size:
            raise ValueError("Expected uint8 codes with the codec's encoded width")
        faiss.omp_set_num_threads(self.threads)
        if not len(codes):
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors = self.index.sa_decode(np.ascontiguousarray(codes))
        if not np.isfinite(vectors).all():
            raise ValueError("Codec produced non-finite reconstructed vectors")
        return vectors

    def serialize(self):
        _check_state(self.index)
        state = faiss.serialize_index(self.index).tobytes()
        metadata = json.dumps({"name": self.name, "dimensions": self.dimensions,
                               "code_size": self.code_size, "threads": self.threads,
                               "faiss_version": faiss.__version__, "state_bytes": len(state),
                               "state_sha256": hashlib.sha256(state).hexdigest()},
                              sort_keys=True, separators=(",", ":")).encode()
        return MAGIC + struct.pack("<I", len(metadata)) + metadata + state


def train(name, sample, threads=8):
    sample = _vectors(sample)
    if not len(sample) or isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("Training needs nonempty samples and a positive thread count")
    if name not in METHODS:
        raise ValueError(f"Unknown Faiss codec: {name}")
    dimensions = sample.shape[1]
    faiss.omp_set_num_threads(threads)
    references = []
    if name in SCALAR_PARAMETERS or name == "bf16":
        if name == "bf16":
            index = faiss.IndexScalarQuantizer(dimensions, faiss.ScalarQuantizer.QT_bf16)
        else:
            bits, statistic, argument = SCALAR_PARAMETERS[name]
            quantizer_type = getattr(faiss.ScalarQuantizer, f"QT_{bits}bit" + ("_uniform" if statistic == "uniform" else ""))
            index = faiss.IndexScalarQuantizer(dimensions, quantizer_type)
            index.sq.rangestat = {"minmax": faiss.ScalarQuantizer.RS_minmax,
                                  "uniform": faiss.ScalarQuantizer.RS_minmax,
                                  "meanstd": faiss.ScalarQuantizer.RS_meanstd,
                                  "quantile": faiss.ScalarQuantizer.RS_quantiles,
                                  "optim": faiss.ScalarQuantizer.RS_optim}[statistic]
            index.sq.rangestat_arg = argument
            # Faiss 1.13 non-minmax range training shares scratch storage across workers.
            if statistic not in ("minmax", "uniform"):
                faiss.omp_set_num_threads(1)
        try:
            index.train(sample)
        finally:
            faiss.omp_set_num_threads(threads)
        return FaissCodec(name, index, threads)
    family, m, bits = re.fullmatch(r"(pq|opq|rq|lsq)(\d+)x(\d+)", name).groups()
    m, bits = int(m), int(bits)
    if len(sample) < 1 << bits:
        raise ValueError("Training sample has fewer rows than the codec's codebook entries")
    if family in ("pq", "opq") and dimensions % m:
        raise ValueError("Product-quantizer subspaces must divide the vector dimensions")
    if family in ("pq", "opq"):
        index = faiss.IndexPQ(dimensions, m, bits)
        index.pq.cp.seed = 42
        index.pq.cp.niter = 20
        if family == "opq":
            transform = faiss.OPQMatrix(dimensions, m)
            transform.niter = 20
            transform.niter_pq_0 = 20
            transform.niter_pq = 4
            training_pq = faiss.ProductQuantizer(dimensions, m, bits)
            training_pq.cp.seed = 42
            transform.pq = training_pq
            references.extend((transform, training_pq, index))
            index = faiss.IndexPreTransform(transform, index)
    elif family == "rq":
        index = faiss.IndexResidualQuantizer(dimensions, m, bits, faiss.METRIC_L2,
                                            faiss.AdditiveQuantizer.ST_decompress)
        index.rq.cp.seed = 42
        index.rq.max_mem_distances = MAX_DISTANCE_BYTES
    else:
        index = faiss.IndexLocalSearchQuantizer(dimensions, m, bits, faiss.METRIC_L2,
                                               faiss.AdditiveQuantizer.ST_decompress)
        index.lsq.random_seed = 42
        index.lsq.max_mem_distances = MAX_DISTANCE_BYTES
    index.train(sample)
    return FaissCodec(name, index, threads, references)


def load(payload):
    if not isinstance(payload, bytes) or len(payload) < 12 or payload[:8] != MAGIC:
        raise ValueError("Invalid serialized Faiss codec")
    metadata_bytes = struct.unpack_from("<I", payload, 8)[0]
    if not 0 < metadata_bytes <= 65536 or len(payload) <= 12 + metadata_bytes:
        raise ValueError("Invalid serialized Faiss codec lengths")
    try:
        metadata = json.loads(payload[12:12 + metadata_bytes])
        state = payload[12 + metadata_bytes:]
        if (metadata["name"] not in METHODS or metadata["state_bytes"] != len(state)
                or metadata["state_sha256"] != hashlib.sha256(state).hexdigest()
                or metadata["faiss_version"] != faiss.__version__):
            raise ValueError("Serialized Faiss codec identity or checksum differs")
        index = faiss.deserialize_index(np.frombuffer(state, dtype=np.uint8).copy())
        codec = FaissCodec(metadata["name"], index, metadata["threads"])
        if codec.dimensions != metadata["dimensions"] or codec.code_size != metadata["code_size"]:
            raise ValueError("Serialized Faiss codec dimensions differ")
        return codec
    except (KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        raise ValueError("Invalid serialized Faiss codec state") from error
