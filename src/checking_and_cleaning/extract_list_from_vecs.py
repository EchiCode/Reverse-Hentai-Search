import struct

import numpy as np
import os

from tqdm import tqdm


def load_vectors(filepath, vec_dim=384):
    record_dtype = np.dtype([('code', '<i4'), ('page', '<i2'), ('vec', np.float16, vec_dim)])
    record_size = record_dtype.itemsize

    with open(filepath, "rb") as f:
        data = f.read()
        n_records = len(data) // record_size
        records = np.frombuffer(data[:n_records*record_size], dtype=record_dtype)

        codes = records['code']

    return codes

filenames = [fn for fn in os.listdir("vectors_new") if fn.endswith(".bin")]
seen = set()

for fn in tqdm(filenames):
    A = load_vectors(os.path.join("vectors_new", fn))
    seen.update(A)

print(len(seen))

np.save("vec_codes.npy", list(seen))