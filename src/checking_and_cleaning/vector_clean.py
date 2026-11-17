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
        pages = records['page']
        vecs = records['vec']

    return codes, pages, vecs

def save_vec(arr,path):
    with open(path, "wb") as f:
        for c,p,v in arr:
            code = int(c)
            page = int(p)
            vector = np.array(v, dtype=np.float16)
            f.write(struct.pack('<i', code))
            f.write(struct.pack('<h', page))
            f.write(vector.tobytes())


filenames = [fn for fn in os.listdir("vectors_new") if fn.endswith(".bin")]
ids = frozenset(np.load("loaded_new.npy"))
dup = set()
seen = set()
i =0
j = 0
for fn in tqdm(filenames):
    A,B,C = load_vectors(os.path.join("vectors_new", fn))
    cur = []
    for a,b,c in zip(A,B,C):
        if a in ids:
            if (a,b) not in seen:
                seen.add((a,b))
                cur.append((a,b,c))
            else:
                i+=1
        else:
            j += 1

    save_vec(cur, os.path.join("vectors_clean", fn))

print("Removed",i,"duplicates.")
print("Removed",j,"with no metadata.")