import os
import gzip
import shutil
import math
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
# from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm

# Use the new numpy RNG
rng = np.random.default_rng()
total_vectors_count = 1466 # len(os.listdir('vectors'))

# =====================
# Streaming utilities
# =====================

def stream_vectors(folder, vec_dim=384, batch_size=500_000):
    """
    Yield batches of (codes, pages, vectors) from binary files.
    Each record: int32 code, int16 page, float16 vector.
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
            codes, pages, vecs = records['code'], records['page'], records['vec']

            for start in range(0, n_records, batch_size):
                end = min(start + batch_size, n_records)
                yield codes[start:end], pages[start:end], vecs[start:end]

def sample_for_training(folder, vec_dim=384, sample_size=1_000_000, batch_size=500_000, fraction=0.1):
    """Sample a fraction of each batch until reaching sample_size."""
    samples = []
    for _, _, batch_vecs in tqdm(stream_vectors(folder, vec_dim, batch_size),total=total_vectors_count,desc='Taking a sample for training'):
        k = max(1, int(len(batch_vecs) * fraction))
        idx = rng.choice(len(batch_vecs), size=k, replace=False)
        samples.append(batch_vecs[idx])

    all_samples = np.concatenate(samples, axis=0)
    if len(all_samples) > sample_size:
        idx = rng.choice(len(all_samples), sample_size, replace=False)
        all_samples = all_samples[idx]

    return all_samples.astype(np.float32)

# =====================
# Parallel gzip
# =====================

def gzip_file(path):
    gz_path = path + ".gz"
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)

def compress_clusters_parallel(cluster_paths, max_workers=6):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(gzip_file, cluster_paths)

# =====================
# GPU FAISS clustering + assignment
# =====================

def cluster_and_save_streaming_gpu(folder, output_dir, n_clusters=10_000, sample_size=2_500_000, save=True,
                                   existing_centroids=False,
                                   batch_size=500_000, vec_dim=384, sample_fraction=0.1,
                                   max_gzip_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    centroids_path = os.path.join(output_dir, "centroids.bin.gz")

    # print("Sampling vectors for init...")
    # sample = sample_for_training(folder, vec_dim, init_sample_size, batch_size, sample_fraction)
    # print("Init...")
    # centroids,_ = kmeans_plusplus(sample,n_clusters)

    if existing_centroids:
        centroids, gpu_index = load_centroids_and_build_index(centroids_path,use_gpu=True)
    else:
        # === 1. Sample and train k-means on GPU ===
        print("Sampling vectors for training...")
        sample = sample_for_training(folder, vec_dim, sample_size, batch_size, sample_fraction)
        print(f"Training k-means on {len(sample)} vectors...")

        clustering = faiss.Clustering(vec_dim, n_clusters)
        clustering.niter = 50
        # clustering.centroids = centroids
        clustering.max_points_per_centroid = 10_000
        index_cpu = faiss.IndexFlatL2(vec_dim)

        # Move to GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        clustering.train(sample, gpu_index)

        # === 2. Save centroids (gzip) ===
        centroids = faiss.vector_to_array(clustering.centroids).reshape(n_clusters, vec_dim)
        centroids_fp16 = centroids.astype(np.float16)
        with gzip.open(centroids_path, "wb") as f:
            f.write(centroids_fp16.tobytes())
        print(f"Saved {centroids_path} with shape {centroids.shape}")

    # === 3. Prepare raw cluster files for streaming writes ===
    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)
    cluster_paths = [os.path.join(clusters_dir, f"cluster_{cid}.bin") for cid in range(n_clusters)]
    cluster_files = [open(p, "wb") for p in cluster_paths]

    if save:
        # === 4. Assign vectors in batches on GPU and write efficiently ===
        total = 0
        for codes, pages, vecs in tqdm(stream_vectors(folder, vec_dim, batch_size),total=total_vectors_count,desc='Saving data'):
            vecs_f32 = vecs.astype(np.float32)

            _, cluster_ids = gpu_index.search(vecs_f32, 1)
            cluster_ids = cluster_ids[:, 0]

            # For each cluster in this batch, write all records at once
            for cid in np.unique(cluster_ids):
                mask = cluster_ids == cid
                if not np.any(mask):
                    continue
                batch_codes = codes[mask].astype(np.int32).reshape(-1, 1)
                batch_pages = pages[mask].astype(np.int16).reshape(-1, 1)
                batch_vecs = vecs[mask]

                packed = np.hstack([
                    batch_codes.view(np.uint8),
                    batch_pages.view(np.uint8),
                    batch_vecs.view(np.uint8)
                ])
                cluster_files[cid].write(packed.tobytes())

            total += len(codes)
            # print(f"Processed {total} vectors")

        for f in cluster_files:
            f.close()

        # === 5. Parallel gzip compression ===
        compress_clusters_parallel(cluster_paths, max_workers=max_gzip_workers)

    print("Finished clustering + saving (GPU + streaming).")
    return centroids, gpu_index

# =====================
# Benchmark recall (streaming)
# =====================

def benchmark_recall_streaming(folder, index, n_queries=100, noise_std=0.01,
                               vec_dim=384, batch_size=500_000, batch_sample_factor=100):
    queries = []
    for _, _, batch_vecs in stream_vectors(folder, vec_dim, batch_size):
        remaining = n_queries - len(queries)
        if remaining <= 0:
            break
        k = min(len(batch_vecs), math.ceil(n_queries / batch_sample_factor), remaining)
        sampled_indices = rng.choice(len(batch_vecs), size=k, replace=False)
        queries.extend(batch_vecs[sampled_indices])

    queries = np.array(queries, dtype=np.float32)
    noisy = queries + rng.normal(scale=noise_std, size=queries.shape).astype(np.float32)

    _, assign_orig = index.search(queries, 1)
    _, assign_noisy = index.search(noisy, 1)

    recall = np.mean(assign_orig[:, 0] == assign_noisy[:, 0]) * 100
    print(f"Benchmark recall: {recall:.4f}% over {len(queries)} queries")
    return recall


def topk_recall_streaming(folder, index, k=5, n_queries=1000, noise_std=0.01,
                                vec_dim=384, batch_size=500_000, batch_sample_factor=100):
    """
    Compute top-k recall for noisy queries (Gaussian noise added).

    Parameters
    ----------
    folder : str
        Folder with binary vectors (same format as stream_vectors).
    index : faiss.Index
        FAISS index to search.
    k : int
        Top-k neighbors to consider for recall.
    n_queries : int
        Number of query vectors to sample.
    noise_std : float
        Standard deviation of Gaussian noise added to queries.
    vec_dim : int
        Dimensionality of vectors.
    batch_size : int
        Streaming batch size.
    batch_sample_factor : int
        How many vectors to sample per batch relative to n_queries.
    """
    # 1. Sample queries
    queries = []
    for _, _, batch_vecs in stream_vectors(folder, vec_dim, batch_size):
        remaining = n_queries - len(queries)
        if remaining <= 0:
            break
        sample_size = min(len(batch_vecs), math.ceil(n_queries / batch_sample_factor), remaining)
        sampled_indices = rng.choice(len(batch_vecs), size=sample_size, replace=False)
        queries.extend(batch_vecs[sampled_indices])

    queries = np.array(queries, dtype=np.float32)

    # 2. Add Gaussian noise
    noisy_queries = queries + rng.normal(scale=noise_std, size=queries.shape).astype(np.float32)

    # 3. Ground truth nearest neighbors
    _, true_nn = index.search(queries, 1)

    # 4. Top-k neighbors for noisy queries
    _, topk_nn = index.search(noisy_queries, k)

    # 5. Compute top-k recall
    hits = (topk_nn == true_nn).any(axis=1)
    recall = np.mean(hits) * 100

    print(f"Top-{k} recall (noise_std={noise_std}): {recall:.4f}% over {len(queries)} queries")
    return recall


# =====================
# Load centroids and build CPU/GPU index
# =====================

def load_centroids_and_build_index(centroids_path, vec_dim=384, use_gpu=False):
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Centroids file not found: {centroids_path}")

    with gzip.open(centroids_path, "rb") as f:
        data = f.read()
        n_centroids = len(data) // (vec_dim * 2)
        centroids = np.frombuffer(data, dtype=np.float16).reshape(n_centroids, vec_dim)

    index_cpu = faiss.IndexFlatL2(vec_dim)
    index_cpu.add(centroids.astype(np.float32))

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu

    print(f"Loaded {n_centroids} centroids and built {'GPU' if use_gpu else 'CPU'} index")
    return centroids, index


def load_centroids_and_build_index_split(centroids_folder, vec_dim=384, use_gpu=True):
    if not os.path.exists(centroids_folder):
        raise FileNotFoundError(f"Centroids file not found: {centroids_folder}")

    centroids = []
    files = list(os.listdir(centroids_folder))
    files = sorted(files)
    for path in files:
        with gzip.open(os.path.join(centroids_folder,path), "rb") as f:
            data = f.read()
            n_centroids = len(data) // (vec_dim * 2)
            cur = np.frombuffer(data, dtype=np.float16).reshape(n_centroids, vec_dim)

        centroids.extend(cur)
    centroids = np.array(centroids)

    index_cpu = faiss.IndexFlatL2(vec_dim)
    index_cpu.add(centroids.astype(np.float32))

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu

    print(f"Loaded {n_centroids} centroids and built {'GPU' if use_gpu else 'CPU'} index")
    return centroids, index
# =====================
# Main
# =====================

if __name__ == "__main__":
    folder = "vectors"
    output_dir = "clusters"

    total_vectors_count = len(os.listdir(folder))

    centroids, index = cluster_and_save_streaming_gpu(
        folder,
        output_dir,
        n_clusters= 20_000,
        sample_size= 5_000_000,
        batch_size=250_000,
        vec_dim=384,
        sample_fraction=0.15,
        max_gzip_workers=6,
        save=True,
        existing_centroids = True # do not retrain centroids, just cluster
    )

    # centroids, index = load_centroids_and_build_index('clusters/centroids.bin.gz')
    # centroids, index = load_centroids_and_build_index_split('clusters')
    # Benchmark

    for k in [5,10,20]:
        topk_recall_streaming(folder, index, k=k, n_queries=100_000, noise_std=0.01)
        topk_recall_streaming(folder, index, k=k, n_queries=100_000, noise_std=0.05)

        topk_recall_streaming(folder, index, k=k, n_queries=100_000, noise_std=0.01)
        topk_recall_streaming(folder, index, k=k, n_queries=100_000, noise_std=0.05)
