import os
import shutil  # Needed to move files
from tqdm import tqdm
import numpy as np

path = 'clusters_balanced_l2_64k/clusters'
output_dir = 'clusters_balanced_l2_64k/split_clusters'
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(path)
keys = []
for f in files:
    f1 = f[8:]
    f2 = f1[:-7]
    keys.append(int(f2))
num = len(files)

I = np.argsort(keys)
files = list(np.array(files)[I])

chunk_size = 1000

# Write chunks
for i in range((num + chunk_size - 1) // chunk_size):
    start = i * chunk_size
    end = min(start + chunk_size, num)

    if start >= end:
        break

    subfolder = os.path.join(output_dir, str(i))
    os.makedirs(subfolder, exist_ok=True)

    for j in tqdm(range(start, end),desc="splitting chunk " + str(i)):
        src = os.path.join(path,files[j])
        dst = os.path.join(subfolder,os.path.basename(src))
        shutil.copy(src, dst)  # Move the file
