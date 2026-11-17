import os
import numpy as np
from tqdm import tqdm
folder = 'vectors'
vec_dim = 384

filenames = [fn for fn in os.listdir(folder) if fn.endswith(".bin")]
record_dtype = np.dtype([('code', '<i4'), ('page', '<i2'), ('vec', np.float16, vec_dim)])
record_size = record_dtype.itemsize

total = 0
for fn in tqdm(filenames):
    filepath = os.path.join(folder, fn)
    with open(filepath, "rb") as f:
        n_records = len(f.read()) // record_size
        total+= n_records

print(total)

# 29_658_233