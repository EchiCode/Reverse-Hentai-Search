import struct
import shutil

import gzip
import faiss
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


record_struct = struct.Struct("<ih")  # int32 + int16

# =====================
# Streaming utilities
# =====================

import numpy as np
import os
import struct

def stream_vectors(folder, vec_dim=384, batch_size=500_000):
    """
    Fast streaming of vectors: reads entire files in one go, then yields batches.
    """
    filenames = [fn for fn in os.listdir(folder) if fn.endswith(".bin")]
    record_dtype = np.dtype([('code', '<i4'), ('page', '<i2'), ('vec', np.float16, vec_dim)])
    record_size = record_dtype.itemsize

    for fn in filenames:
        filepath = os.path.join(folder, fn)
        with open(filepath, "rb") as f:
            data = f.read()
            n_records = len(data) // record_size
            records = np.frombuffer(data[:n_records*record_size], dtype=record_dtype)

            codes = records['code']
            pages = records['page']
            vecs = records['vec']

            # yield in batches
            for start in range(0, n_records, batch_size):
                end = min(start + batch_size, n_records)
                yield codes[start:end], pages[start:end], vecs[start:end]

def sample_for_training(folder, vec_dim=384, sample_size=1_000_000, batch_size=500_000, fraction=0.1):
    """Sample a fraction of each batch until reaching sample_size."""
    samples = []

    for _, _, batch_vecs in tqdm(stream_vectors(folder, vec_dim, batch_size),total=1466,desc='Taking a sample of training'):
        k = max(1, int(len(batch_vecs) * fraction))
        idx = np.random.choice(len(batch_vecs), size=k, replace=False)
        samples.append(batch_vecs[idx])

    all_samples = np.concatenate(samples, axis=0)
    if len(all_samples) > sample_size:
        idx = np.random.choice(len(all_samples), sample_size, replace=False)
        all_samples = all_samples[idx]

    return all_samples.astype(np.float32)

def gzip_file(path):
    gz_path = path + ".gz"
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)

def compress_clusters_parallel(cluster_paths, max_workers=6):
    """
    Compress multiple cluster files in parallel.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(gzip_file, cluster_paths)

# =====================
# Clustering + saving
# =====================
def cluster_and_save_streaming(folder, output_dir, n_clusters=100, sample_size=1_000_000, batch_size=500_000, vec_dim=384,save=True):
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Train on sample ===
    sample = sample_for_training(folder, vec_dim, sample_size, batch_size)
    print(f"Training on {len(sample)} sampled vectors")
    clustering = faiss.Clustering(vec_dim, n_clusters)
    clustering.niter = 20
    clustering.max_points_per_centroid = 100_000
    index = faiss.IndexFlatL2(vec_dim)
    clustering.train(sample, index)


    # === Save centroids (gzip) ===
    centroids = faiss.vector_to_array(clustering.centroids).reshape(n_clusters, vec_dim)

    if save:
        centroids_fp16 = centroids.astype(np.float16)
        centroids_path = os.path.join(output_dir, "centroids.bin.gz")
        with gzip.open(centroids_path, "wb") as f:
            f.write(centroids_fp16.tobytes())
        print(f"Saved {centroids_path} with shape {centroids.shape}")

        os.makedirs(output_dir + "/clusters", exist_ok=True)
        cluster_paths = [os.path.join(output_dir, "clusters", f"cluster_{cid}.bin") for cid in range(n_clusters)]
        cluster_files = [open(p, "wb") for p in cluster_paths]

        total = 0
        for codes, pages, vecs in tqdm(stream_vectors(folder, vec_dim, batch_size),total=1466,desc='Saving data'):
            # 1. Assign clusters
            _, I = index.search(vecs.astype(np.float32), 1)
            cluster_ids = I[:, 0]

            # 2. For each cluster in the batch, collect vectors and write in one shot
            for cid in np.unique(cluster_ids):
                mask = cluster_ids == cid
                if not np.any(mask):
                    continue

                # Stack code (int32), page (int16), and vec (float16)
                batch_codes = codes[mask].astype(np.int32).reshape(-1, 1)
                batch_pages = pages[mask].astype(np.int16).reshape(-1, 1)
                batch_vecs = vecs[mask]  # float16

                packed = np.hstack([batch_codes.view(np.uint8), batch_pages.view(np.uint8), batch_vecs.view(np.uint8)])
                cluster_files[cid].write(packed.tobytes())

            # gc.collect()
            total += len(codes)
            print(f"Processed {total} vectors")

        for f in cluster_files:
            f.close()

        compress_clusters_parallel(cluster_paths)

    print("Finished clustering + saving (streaming).")
    return centroids, index

# =====================
# Benchmark recall (streaming)
# =====================

import math


def benchmark_recall_streaming(
        folder,
        index,
        n_queries=100,
        noise_std=0.01,
        vec_dim=384,
        batch_size=500_000,
        batch_sample_factor=100
):
    """
    Benchmark recall by sampling a fraction of each batch until n_queries total.

    Args:
        folder: folder with vector .bin files
        index: faiss index with trained centroids
        n_queries: total number of query vectors
        noise_std: std deviation of Gaussian noise added to queries
        vec_dim: vector dimensionality
        batch_size: batch size for streaming vectors
        batch_sample_factor: sample ~n_queries / batch_sample_factor from each batch
    """
    queries = []

    for _, _, batch_vecs in stream_vectors(folder, vec_dim, batch_size):
        remaining = n_queries - len(queries)
        if remaining <= 0:
            break

        k = min(len(batch_vecs), math.ceil(n_queries / batch_sample_factor), remaining)
        sampled_indices = np.random.choice(len(batch_vecs), size=k, replace=False)
        queries.extend(batch_vecs[sampled_indices])

    queries = np.array(queries, dtype=np.float32)
    noisy = queries + np.random.normal(scale=noise_std, size=queries.shape).astype(np.float32)

    _, assign_orig = index.search(queries, 1)
    _, assign_noisy = index.search(noisy, 1)

    recall = np.mean(assign_orig[:, 0] == assign_noisy[:, 0])
    recall*=100
    print(f"Benchmark recall: {recall:.4f}% over {len(queries)} queries")
    return recall



def load_centroids_and_build_index(centroids_path, vec_dim=384):
    """
    Load centroids from a .bin.gz file and build a FAISS IndexFlatL2.

    Args:
        centroids_path: path to centroids.bin.gz
        vec_dim: dimensionality of the vectors

    Returns:
        centroids (np.float16 array), faiss.IndexFlatL2
    """
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Centroids file not found: {centroids_path}")

    with gzip.open(centroids_path, "rb") as f:
        data = f.read()
        n_centroids = len(data) // (vec_dim * 2)  # float16 = 2 bytes
        centroids = np.frombuffer(data, dtype=np.float16).reshape(n_centroids, vec_dim)

    # Build FAISS index (needs float32)
    index = faiss.IndexFlatL2(vec_dim)
    index.add(centroids.astype(np.float32))

    print(f"Loaded {n_centroids} centroids and built index")
    return centroids, index

# =====================
# Main
# =====================

if __name__ == "__main__":
    folder = "vectors_new"
    output_dir = "clusters_"

    centroids, index = cluster_and_save_streaming(
         folder,
         output_dir,
         n_clusters=1_000,
         sample_size=500_000,
         batch_size=500_000,
         vec_dim=384,
         save =False)

    # centroids, index = load_centroids_and_build_index(output_dir)
    benchmark_recall_streaming(folder, index, n_queries=1_000, noise_std=0.001)
    benchmark_recall_streaming(folder, index, n_queries=1_000, noise_std=0.01)

    benchmark_recall_streaming(folder, index, n_queries=10_000, noise_std=0.001)
    benchmark_recall_streaming(folder, index, n_queries=10_000, noise_std=0.01)

    benchmark_recall_streaming(folder, index, n_queries=100_000, noise_std=0.001)
    benchmark_recall_streaming(folder, index, n_queries=100_000, noise_std=0.01)
