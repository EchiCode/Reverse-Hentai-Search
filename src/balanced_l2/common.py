"""Shared file formats and bounded input preparation."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

import numpy as np


def source_path(root, relative):
    """Resolve client assets in nested development or sibling release layouts."""
    root, relative = Path(root), Path(relative)
    if relative.parts[:1] == ("client",) and not (root / "client").is_dir():
        root = root.parent
    return root / relative


def record_dtype(dim):
    return np.dtype([("code", "<i4"), ("page", "<i2"), ("vec", "<f2", (dim,))])


def input_files(paths):
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else paths
    files = []
    for item in paths:
        item = Path(item)
        found = list(item.glob("*.bin")) if item.is_dir() else [item]
        found.sort(key=lambda p: (0, int(p.stem)) if p.stem.isdecimal() else (1, p.name))
        files.extend(path.resolve() for path in found)
    if not files or len(set(files)) != len(files):
        raise ValueError("Expected distinct original vector files")
    return files


def snapshot(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    return {**snapshot(path), "sha256": sha256(path)}


def check_artifact(value):
    if artifact(value["path"]) != value:
        raise ValueError(f"Saved build artifact changed: {value['path']}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require_disk_space(path, size):
    if shutil.disk_usage(path).free < size + 1024 ** 3:
        raise OSError(f"Need {size / 1024 ** 3:.1f} GiB plus 1 GiB free on {path}")


def matrix_shape(path):
    path = Path(path)
    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise ValueError(f"Truncated float32 matrix: {path}")
    rows, dim = struct.unpack("<ii", header)
    if rows <= 0 or dim <= 0 or path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"Invalid float32 matrix: {path}")
    return rows, dim


def read_matrix(path):
    rows, dim = matrix_shape(path)
    values = np.fromfile(path, dtype="<f4", offset=8).reshape(rows, dim)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite float32 matrix: {path}")
    return values


def iter_record_batches(files, dim, batch_size=250000):
    """Read through a reusable buffer and release cache after each source file."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    buffer = np.empty(batch_size, dtype=record_dtype(dim))
    view, filled = memoryview(buffer).cast("B"), 0
    for path in files:
        if Path(path).stat().st_size % buffer.dtype.itemsize:
            raise ValueError(f"Truncated vector record: {path}")
        with Path(path).open("rb", buffering=0) as source:
            while True:
                length = source.readinto(view[filled:])
                if not length:
                    break
                filled += length
                if filled == len(view):
                    yield buffer
                    filled = 0
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(source.fileno(), 0, source.tell(), os.POSIX_FADV_DONTNEED)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if filled:
        if filled % buffer.dtype.itemsize:
            raise ValueError("Truncated source records")
        yield buffer[:filled // buffer.dtype.itemsize]


def prepare_vectors(files, vector_path, dim, batch_size=250000, sample_rows=None, sample_path=None):
    count = sum(Path(path).stat().st_size // record_dtype(dim).itemsize for path in files)
    if not 0 < count <= np.iinfo(np.int32).max:
        raise ValueError("Expected between 1 and 2^31-1 source vectors")
    if sample_rows is not None and (sample_rows.ndim != 1 or not len(sample_rows)
            or sample_rows[0] < 0 or sample_rows[-1] >= count or (np.diff(sample_rows) <= 0).any()):
        raise ValueError("Sample rows must be sorted, unique source indices")
    with Path(vector_path).open("xb") as target, (
            Path(sample_path).open("xb") if sample_rows is not None else nullcontext()) as sampled:
        target.write(struct.pack("<ii", count, dim))
        if sampled:
            sampled.write(struct.pack("<ii", len(sample_rows), dim))
        offset = 0
        for records in iter_record_batches(files, dim, batch_size):
            values = records["vec"].astype("<f4")
            norms = np.linalg.norm(values, axis=1)
            if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Source vectors must be finite and nonzero")
            values /= norms[:, None]
            target.write(values.tobytes())
            if sampled:
                left, right = np.searchsorted(sample_rows, [offset, offset + len(values)])
                sampled.write(values[sample_rows[left:right] - offset].tobytes())
            offset += len(values)
    if offset != count:
        raise ValueError("Source row count changed during preparation")
    return count
