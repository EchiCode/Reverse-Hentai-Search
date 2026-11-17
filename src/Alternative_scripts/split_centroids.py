import numpy as np
import gzip
import os

vec_dim = 384
centroids_path = 'clusters/centroids.bin.gz'
output_dir = 'clusters/centroids_split'
num_parts = 8  # adjust as needed

os.makedirs(output_dir, exist_ok=True)

# Load and decompress
with gzip.open(centroids_path, "rb") as f:
    data = f.read()

n_centroids = len(data) // (vec_dim * 2)
centroids = np.frombuffer(data, dtype=np.float16).reshape(n_centroids, vec_dim)

# Compute split sizes
chunk_size = (n_centroids + num_parts - 1) // num_parts  # ceil division

# Write chunks
for i in range(num_parts):
    start = i * chunk_size
    end = min(start + chunk_size, n_centroids)
    if start >= end:
        break
    chunk = np.copy(centroids[start:end])
    # chunk[:,0] = start + np.arange(len(chunk))
    out_path = os.path.join(output_dir, f"centroid_{i}.bin.gz")
    with gzip.open(out_path, "wb") as f_out:
        f_out.write(chunk.astype(np.float16).tobytes())
    print(f"Saved {out_path} with {len(chunk)} vectors")

print("✅ Done splitting", n_centroids, "centroids into", i + 1, "files.")
