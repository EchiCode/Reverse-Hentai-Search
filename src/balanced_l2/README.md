# Balanced L2 clustering

`clustering.py` trains cuVS hierarchical balanced squared-L2 k-means on
a uniform sample, then assigns every normalized source vector to one nearest
learned centroid. Centroid norms are preserved. The browser reads LVQ6 version 2
and routes by `dot(query, centroid) - 0.5 * squared_norm(centroid)`.

Run these commands from `src`. Create a Python 3.13 environment and install the
core export dependencies with:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install numpy brotli
```

GPU training additionally needs compatible `cuvs-cu12` and `cupy-cuda12x`
packages and their CUDA runtime dependencies. CPU assignment uses NumPy;
`--cpu-assignment` does not make cuVS training CPU-only. Install `torch`,
`transformers`, and `pillow` separately for embedding tools. Codec comparisons
also need `faiss-cpu`, `torch`, `gguf`, and `matplotlib`; compression experiments
optionally use the `zstd` executable and Python `lz4` package. These optional
dependencies are not needed for the core exporter. No dependency lockfile is
included in this source export.

Run one heavy job at a time:

```bash
systemd-run --user --scope -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=512M \
  .venv/bin/python clustering.py vectors \
  --output clusters_balanced_l2_96k_final --clusters 96000 \
  --batch-size 65536 --threads 4 --brotli-centroids
```

The default training sample is 2.5 million normalized float32 vectors, selected
without replacement with seed 42. The cuVS hierarchical initializer uses its
library default because its Python API does not expose an RNG seed. Training
uses 20 iterations per hierarchical clustering step. Native balancing affects
training; final nearest-centroid assignment has no hard posting-size cap.

cuVS requires the full sample on the GPU. Before copying it, the trainer requires
sample bytes plus centroid bytes plus 1 GiB workspace headroom. Assignment score
tiles are capped at 1 GiB. The 24 GiB systemd ceiling covers host memory and file
cache, and leaves the desktop outside that scope.

The [cuVS clustering API](https://docs.nvidia.com/cuvs/api-reference/python-api-cluster-kmeans)
provides the training implementation. No native helper compilation is required.

The exporter preserves original `int32 code + int16 page + fp16[384]` records.
It spools one bounded group of centroids at a time, compresses at most 512 posting
streams together, checkpoints completed shards, and removes their temporary
files immediately. It retains `native/heads.f32` and `native/assignments.i32`,
plus actual compressed posting sizes and record counts. It creates a fetchable
gzip file for empty clusters too.

The output contains:

- `centroids.bin.gz`: original learned means rounded to FP16, removed after a
  successful `--brotli-centroids` export.
- `centroids.lvq6.bin.gz`: LVQ6 centroids marked for L2 routing.
- `centroids.lvq6.bin.br`: the same LVQ6 payload with Brotli compression when
  `--brotli-centroids` is enabled; it must fit the 25 MiB Pages asset limit.
- `clusters/cluster_<id>.bin.gz`: original records in source order.
- `stats.json`, `posting_counts.npy`, and `posting_gzip_sizes.npy`.

Repeat the identical command with `--resume` after an interruption. With completed
full-corpus assignments in that output, export directly using the same options:

```bash
systemd-run --user --scope -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=512M \
  .venv/bin/python clustering.py vectors \
  --output clusters_balanced_l2_96k_final --clusters 96000 \
  --batch-size 65536 --threads 4 --brotli-centroids --resume --export-only
```

New builds produce exactly the requested centroid count. Export requires
original sources and native labels/centroids, with no normalized full-corpus
staging or retraining. Export can temporarily use the original raw corpus size
for spool shards, plus compressed outputs; the build checks disk space.

Run the small checks with:

```bash
.venv/bin/python -m unittest balanced_l2.tests.test_pipeline
BALANCED_L2_TEST_GPU=1 .venv/bin/python -m unittest balanced_l2.tests.test_pipeline
```

Centroid and posting IDs must be deployed together. `deploy_data.py` uploads and
verifies the complete dataset before preparing the client's immutable data URLs.
The client release instructions contain the data-only deployment command. Publish the client
only after every data batch verifies. The local build does not publish anything.

## Head-count benchmark

This older startup-cost benchmark also requires a local
`balanced_l2_benchmark/client_assets.json` calibration input. Its
`fixed_bytes.gzip` field must contain the measured integer download size of the
model, libraries and runtime, excluding centroids and query-dependent files.
This deployment-specific measurement is not bundled with the source; the joint
quantization comparison below does not require it.

Run the full balanced L2 calibration sequentially with:

```bash
systemd-run --user --scope -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=512M \
  .venv/bin/python benchmark.py vectors
```

The default sweep uses 64k/80k/96k/128k centroids, 100,000 shared queries per
noise level (0.005/0.01/0.025), and probes 1/2/4/5/8/10/12/16/32/64. If a compatible
`clusters_balanced_l2_64k` control exists locally, it can be reused with its actual
count labeled explicitly; otherwise a new model is trained. New training
produces exactly the requested number of centroids. Exact truth
reuse requires the normalized corpus SHA256, source-input digest, and every
regenerated query bit to match.

Results, assignment maps, quantized indexes, routes, CSV/JSON and three plots
are saved under `balanced_l2_benchmark/comparison_lvq6_100k`. Each head count's
complete sweep is saved before the next starts. Temporary benchmark postings
are deleted after each case; shared staging is deleted after all cases finish.
An existing control's posting files are never removed by the runner.

Repeat the same command to resume. Use `--heads ...`, `--noise ...`, `--queries N`
and `--output fresh-directory` for a separate configuration. Generated comparison
results are not included in the source export.

## Fixed-cluster vector quantization

`benchmark_cluster_quantization.py` compares **20 representative codec
configurations by default** on a clustered dataset (`--source`, default `client/data`).
The original 114-configuration grid remains available with `--all-methods`.
It keeps the supplied LVQ6 L2 centroids and posting membership fixed. The
`cluster_quantization_8k` output name comes from the original 8k experiment;
the supplied dataset can use other centroid counts. It does not
recluster vectors or change the client.

The default is **100,000 shared replay queries, noise 0.025, and
1/2/4/8/16/32 probes**. Each format actually encodes, gzip-compresses, inflates,
decodes, and reranks the vectors. Recall measures the returned image ID/page
against exhaustive full-corpus truth; posting coverage is a separate metric.
Same-probe agreement with FP16, tie-aware recall, and original-vector cosine
loss help separate routing misses from quantization losses.

The default shortlist keeps every family while removing most bit-rate and
range-tuning permutations:

```text
fp16
lvq4 lvq5 lvq6 lvq8
sq4 sq6 sq8 sq6_quantile0.005
pq32x4 pq64x6 pq96x8 pq192x8
opq64x8 rq4x6 lsq4x6
gguf_q4_0 gguf_q4_1 gguf_q5_0
fp8_e4m3_row
```

This reduces the sweep from 684 to 120 method/probe points. It is a first-pass
selection, not an accuracy ranking. The full 114-configuration grid contains:

- FP16 and BF16 references.
- LVQ at 2–8 bits, with a local mean for each posting.
- 26 scalar variants: SQ4/SQ6/SQ8 with min–max, mean/std factors 2/3/4,
  quantile tails 0.001/0.005/0.01, optimized ranges, plus uniform SQ4/SQ8.
- 57 PQ layouts: 16/32/48/64/96/128 subquantizers at 2–8 bits, plus
  8/12/24/192/384 subquantizers at 4/6/8 bits.
- OPQ32×8/64×8/96×8, RQ4×6/8×6, and LSQ4×6/8×6.
- Genuine GGUF Q4_0/Q4_1/Q5_0/Q5_1/Q8_0.
- FP8 E4M3/E5M2, each unscaled, scaled per vector, or in 32/64/128-value blocks.

Faiss models train once per configuration on the same seeded sample of 65,536
original FP16 vectors. Codebooks and rotations are globally shared; they are
included once in cold-query bytes. Posting gzip sizes include IDs, pages,
headers, local means/scales, and codes. Cold-query costs also include the fixed
centroid index, and exclude the image model, web libraries, metadata, UI, and
HTTP overhead. Cosine scoring matches the current client's grouped Float64
dot products and Float32 inverse-norm cache. Exhaustive truth uses float32
normalized dot products; the additional tie-aware metric uses a 2e-6 tolerance.

From `src`, after installing the comparison dependencies above:

```bash
systemd-run --user --scope --unit=cluster-quantization-8k \
  -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=512M \
  .venv/bin/python -u benchmark_cluster_quantization.py \
  --source clusters_balanced_l2_96k_final --output cluster_quantization_8k \
  --queries 100000 --probes 1 2 4 8 16 32
```

Results go to `cluster_quantization_8k/results.csv` and `results.json`.
`plots/index.html` links accuracy-versus-gzip curves with probe labels. Generated
plots show FP16 and LVQ. SQ, RQ, LSQ, GGUF, OPQ, FP8 E4M3, PQ with
32/64/96/192 subquantizers, and probe counts below 4 are omitted from plots;
their measurements are retained in generated CSV/JSON. One comparison plot
includes every LVQ format and the dashed FP16 reference. Overview plots show the combined
frontier. PNG, SVG, and PDF are saved. Compare codec/probe pairs at similar
download sizes: LVQ permits more probes for the same download budget, while the
measured recall determines which codec/probe pair to use.

The download axis shows the mean sum of compressed posting bytes per query,
with labeled logarithmic ticks at 1/2/3/5/7 multiples. It does not simulate posting
cache hits. Cold gzip, retained in CSV/JSON, additionally counts the compressed
centroid index and shared codec model once; those can be reused after loading.

Repeat the identical command to resume. Codec models, completed methods, and
posting checkpoints are verified before reuse. Invalid codecs are recorded as
rejected and the sweep continues; source/configuration/code changes require a
fresh `--output`. One original-record staging file uses about 25.2 GB for the
current dataset and is removed on completion. Encoded whole-corpus copies are
not retained for every case; actual per-posting gzip sizes and query winners
are retained. This full-corpus sweep is substantially more work than encoding
64k centroids. Past result files are not included; rerun the comparison to produce
measurements.

For a smaller preflight on the same corpus, use a separate output and fewer
queries/formats (the complete source corpus is still scanned):

```bash
.venv/bin/python -u benchmark_cluster_quantization.py \
  --source clusters_balanced_l2_96k_final \
  --output cluster_quantization_preflight --queries 100 \
  --methods lvq6 sq6 pq32x4 gguf_q4_0 fp8_e4m3_row
```

List the default 20 with `--list-methods`, or the complete grid with
`--all-methods --list-methods`. To run the full grid, add
`--all-methods --output cluster_quantization_8k_full` to the main command.
Regenerate plots without rerunning
the benchmark using `.venv/bin/python -m balanced_l2.plot_posting_quantization
cluster_quantization_8k`. `--cpu` is intended for small synthetic test fixtures.
These are noisy vector replay measurements; real-image and browser/network
latencies need separate deployment testing.

## Joint cluster-count and quantization comparison

`benchmark_joint_quantization.py` compares exactly 64,000 / 80,000 / 96,000 /
128,000 clusters. For each count it independently varies centroid and posting
codecs over **FP16, LVQ8, LVQ6, and LVQ5**: 16 codec pairs per count, 64 total.
The default is **100,000 shared queries, noise 0.025, and 4/8/16/32/64 probes**,
giving **320 measurements**. Inputs default to the current `vectors` directory.

The runner processes one count at a time:

1. Train one balanced squared-L2 model on the shared seeded sample and assign
   the entire normalized corpus using the original learned centroids. Source
   vectors are read and normalized in blocks directly from the original files.
   Save one FP16 clustered dataset directly to its final compressed cluster
   files, without a second raw staging copy.
2. Encode the centroids four ways and route the same queries with each decoded
   index. All variants preserve centroid norms and use L2 routing.
3. Encode each posting four ways, measuring actual gzip9 bytes and decoded-vector
   cosine search across all four centroid routes. Membership stays fixed; each
   posting codec is encoded/decoded once per posting, not once per codec pair.
   The saved FP16 cluster set supplies all tests; quantized posting payloads are
   temporary in memory and do not create additional complete datasets on disk.
4. Save all 16 pairs and five probe counts durably, then delete that count's
   complete generated build (postings, centroids, assignments, routes, and winner
   scratch files) **before training the next count**.

Shared exhaustive full-corpus truth is computed once in original source order.
Recall compares image ID/page after actual posting-vector reranking; routing
coverage is reported separately. Results also contain tie-aware recall, agreement
with the FP16 postings for the same centroid codec, mean/p95 posting downloads,
initial centroid bytes, and full posting storage. All centroid codecs start from
the same exported FP16 centroid values. LVQ5 centroid support is for this
benchmark; it does not change the deployed browser centroid decoder.
Reported `decode_seconds` covers binary codec decoding, excluding gzip inflation.

Run from `src`:

```bash
systemd-run --user --scope --unit=joint-quantization-comparison \
  -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=512M \
  .venv/bin/python -u benchmark_joint_quantization.py \
  --output joint_quantization_comparison
```

Repeat the identical command to resume. Source snapshots, producer versions,
checksums, shared truth, codec routes, and posting checkpoints are validated.
Completed counts reuse retained metrics and stay deleted. Shared normalization,
original-record copies, and raw export chunks are no longer written. Only the
shared training sample is staged (about **3.84 GB**), and it is removed after all
counts finish. One saved clustered dataset uses roughly **22–25 GB**; centroid,
assignment, query, and resume artifacts add a few GB. The temporary disk peak is
estimated at **30–35 GB for the current corpus**; allow **35–40 GB free**. The
runner checks a conservative bound before staging and retains one cluster count
at a time. Original vector files are read-only and are outside this scratch-space
estimate.

Runs created by the earlier full-corpus staging version need a fresh `--output`
directory. Generated checkpoints and measurements are not included in this
source export.

`results.csv`, `results.json`, and `metrics_h<count>.json` retain measurements and
provenance. `plots/index.html` includes a combined frontier and one four-panel
figure per cluster count: each panel fixes the centroid codec and compares all
four posting codecs, with probe labels and initial centroid download size.
PNG, SVG, and PDF are saved. The main download axis remains posting bytes per
query; the initial index download is annotated separately.

Use `--list-cases` to inspect the 64 configurations without starting work.
`--heads`, `--queries`, `--probes`, and `--output` allow a separate run.
`--cpu` is restricted to tiny synthetic fixtures and uses ordinary CPU k-means;
full comparisons use the GPU balanced-L2 trainer. Real-image/browser latency
needs separate testing from these vector replay measurements.
