# Search release: 96k centroids

The selected index uses **96,000 LVQ6 L2 centroids** and whole-file **Brotli-11**.
The verified index is 26,039,981 bytes, below the 26,214,400-byte Pages limit.
The browser decodes it natively; no decoder library or Cloudflare Worker is needed.

The published repository has sibling `src/` and `client/` directories. Run Python
pipeline commands below from `src/`, with `.venv/`, post-rebalance `vectors/` and
`meta/` there. For an older checkout with `client/` inside `src/`, replace
`../client` in these commands with `client`.

## Final build

From `src/`:

```bash
systemd-run --user --scope \
  -p MemoryHigh=8G -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=400% \
  bash build_final.sh
```

A fresh build samples the current `vectors/` and trains centroids. An existing
build resumes from its validated `clusters_balanced_l2_96k_final/native/`
checkpoint, so a completed build does not resample or retrain. Old comparison
directories and their centroid checkpoints are not required.

The script exports FP16, LVQ6 and LVQ8 postings, validates the complete dataset,
and prepares the client for local data testing.

The output is `clusters_balanced_l2_96k_final/`. Repeat the identical command
after interruption: saved stages and outputs are checked before reuse. Existing
unrelated outputs are refused. The script publishes nothing and uploads nothing.

The pipeline avoids full-corpus FP32 staging, removes any owned training sample
after the heads checkpoint, removes the temporary FP16 centroid export after
Brotli validation, and deletes export spool shards as each finishes. Native heads
and assignments remain as small restart artifacts. All three final posting
formats are required and stay on disk.

## Local client

`prepare_client_release.py` runs at the end of the build. It checks every
posting filename/count/size, the centroid identity, the lossless Brotli payload,
the size limit and metadata files before replacing client data symlinks.
It writes `search-release.js` last.

Local data preparation generates `localOnly: true`. The prepared public client
uses the verified Cloudflare URLs with `localOnly: false`, including on localhost.
The main thread and retrieval worker load the same configuration. The frontend
build does not copy the large dataset into `dist/`. Preparing and building the
client does not deploy the live frontend.

To replace the Cloudflare configuration with a local-only dataset for testing,
run from `src/`:

```bash
.venv/bin/python prepare_client_release.py \
  --data clusters_balanced_l2_96k_final --meta meta --client ../client
```

Restore the verified public release configuration before building a frontend
release.

Serve the source client from the repository root:

```bash
python3 client/scripts/serve-local.py --port 8765
```

Open `http://127.0.0.1:8765/` and refresh after preparing a dataset.
This server sets cross-origin isolation headers for CPU inference threads and
the Brotli header needed for local centroid files. A plain Python HTTP server
does not set those headers.

With local data prepared, run from `src/` to check all four presets and metadata
against three real source vectors:

```bash
systemd-run --user --scope \
  -p MemoryHigh=768M -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% \
  timeout --kill-after=2s 120s node --max-old-space-size=512 \
  ../client/scripts/check-quality-search.mjs
```

The smoke check exercises the selected Brotli index and all posting codecs.
It is not a broad recall measurement. `check-local-data.mjs` additionally checks
up to fifteen source vectors with FP16 postings. Pass both paths explicitly
from `src/`:

```bash
node ../client/scripts/check-local-data.mjs ../client/data vectors
```

## Image inference

The client uses the CPU-only ONNX Runtime pair matching Transformers.js 4.2.0,
retaining the same q8 DINO model, preprocessing and normalized CLS embedding.
`image-inference.js` uses half the logical CPU count, rounded up and capped at
six threads in an isolated context; unavailable isolation or CPU information
falls back to one. Explicit thread settings are retained. Update both runtime
URLs together when updating Transformers.

Two serial inference warmups run before the model is marked ready. Centroid
preloading starts on DOMContentLoaded alongside the library/model loading,
without waiting for either; image selection and searches share that same load.

The console reports centroid download/decode timing and the actual inference
threads, isolation and CPU information. Continuous-loop inference measurements
do not predict interactive latency.

Run a fresh real-image comparison from `src/`, serially with GPU disabled and
bounded memory:

```bash
systemd-run --user --scope \
  -p MemoryHigh=2G -p MemoryMax=3G -p MemorySwapMax=0 \
  -p TasksMax=256 -p CPUQuota=600% \
  .venv/bin/python ../client/scripts/benchmark-inference.py \
  --galleries /path/to/galleries
```

The gallery path must contain numbered subdirectories with source images.
The command samples real source images, records original dimensions, hashes,
decode/resize time and separate model execution time, and checks the embeddings.
It fails when no gallery images are available. Results are written to
`src/build/inference-execution/results.json`. Add `--reverse`
to check timing drift.
It uses `client/scripts/browser-benchmark.py`, with browser reports and traces in
`src/build/browser-benchmarks/`; no model, browser, or benchmark variants run in
parallel. The model-ready console log reports actual threads, isolation,
and CPU information to diagnose differences between browsers.

To measure the normal app instead of the isolated model, add `--full-app` to
the command above. It uses the actual app model and `getSearchEmbedding`, fresh
object URLs/canvases, and one-second gaps between 18 requests. It does not run a
second model or retrieve postings. Detailed output is written under the
`src/build/inference-execution/`. No previous timing reports
are required.

Each search now prints `modelPreprocessing`, `modelExecution` and
`modelPipelineOther` separately; they are components of `modelInference`.
Image canvases request `willReadFrequently: true` because the processor reads
their pixels. With DevTools closed, run a search, then open the console and copy
the complete timing snapshot without expanding console objects:

```js
copy(JSON.stringify(lastSearchTiming, null, 2))
```

## Presets and routing

The 96k presets are:

- Budget: LVQ6 × 16 — 1.71 MB mean query download, 94.59% tie-aware recall.
- Balanced/default: LVQ8 × 32 — 4.33 MB, 98.17%.
- High: LVQ8 × 64 — 8.59 MB, 98.97%.
- Very high: FP16 × 64 — 16.79 MB, 99.74%.

These estimates come from a 100,000-query replay at noise 0.025, rescaled to the
final posting sizes. Historical comparison outputs are not included.
Manual probe counts interpolate size estimates only; recall is not interpolated.
Model/index startup, metadata and result images are separate transfers.

Routing uses CPU WebAssembly SIMD with Float32 centroid storage and Float64
accumulation, with four workers as fallback. Scalar routing remains available
if workers cannot start. Image embeddings normalize the 384-value CLS token
alone, matching the producer.

## Upload first, publish the client last

Run from `src/`. Create the dedicated **Pages** project once:

```bash
wrangler pages project create reverse-image-data-96k --production-branch main --force
```

`--force` here selects Pages rather than Wrangler's migration of new projects to
Workers. It is not used when uploading batches. If the project already exists,
skip creation. Then upload **only the data**:

```bash
systemd-run --user --scope \
  -p MemoryHigh=8G -p MemoryMax=10G -p MemorySwapMax=0 -p CPUQuota=200% \
  env NODE_OPTIONS=--max-old-space-size=4096 \
  .venv/bin/python -u deploy_data.py \
  --data clusters_balanced_l2_96k_final --meta meta \
  --project reverse-image-data-96k --client ../client
```

Repeat the upload command after interruption to resume. `--dry-run` validates the
local inventory and prints its size without uploading or changing the client.
The checkpoint is `src/build/data-deploy.json` relative to the repository root.
Preserve it, its inventories and `src/build/uploaded-routes.json` locally; they
are deployment state, not public source files.

The uploader stages one batch at a time using hardlinks, uploads sequentially,
and records immutable deployment URLs. This keeps every existing live data URL
available. The complete upload contains:

- All 96,000 files in each of `clusters/`, `clusters_lvq6/`, `clusters_lvq8/`.
- All current `meta/` files, including metadata batch 11.
- `centroids.lvq6.bin.br`, its data deployment's `_headers`, and `posting-codecs.json`.

The gzip centroid file is retained for backend identity checks; it exceeds the
Pages asset limit and is not the hosted index. Brotli is served with
`Content-Encoding: br` and `Cache-Control: no-transform`. It needs no additional
decoder library or Cloudflare Worker.

There are 96 posting batches (all three codecs together), 12 metadata batches,
and one index/manifest batch. Verification checks the uploaded inventory, every
asset's availability, size and cross-origin headers, plus the complete index and
manifest hashes. It does not redownload and rehash all posting bodies.
When Pages omits the size on HEAD, verification checks GET headers and closes
the connection before reading the body. If GET also omits the size, it streams
that asset to verify its byte count and hash.
Verification reconnects and retries transient network failures and temporary
HTTP errors up to five attempts, with 1, 2, 4 and 8 second delays. Size, hash,
encoding and CORS mismatches still stop the upload.
The verifier requests Brotli for the `.br` index so its compressed bytes can be
hash-checked; the browser decodes that HTTP response natively.

Only after every batch verifies, the uploader writes
`build/uploaded-routes.json` and updates the **local**
client to those Cloudflare URLs with `localOnly: true`. It never builds or
deploys the frontend. Per-batch route arrays select each immutable deployment;
legacy `{id}` / `{batch}` URL templates remain supported.

The reviewed public release configuration uses those same verified Cloudflare
URLs with `localOnly: false`. Publish that frontend last, after local testing.
Old open tabs must retain access to their original dataset. No live frontend
deployment is part of preparing this clean source release.

## Wire format and checks

LVQ6 has a 16-byte header, FP16 global mean, per-row FP16 ranges and packed six-bit
codes. Version 2 ranks L2 centroids with
`query · centroid - 0.5 * ||centroid||²`; version 1 preserves legacy IP routing.
The legacy index remains bundled for hosted fallback. Eight-part manifests are
still supported, but the selected 96k index is served as one Brotli file.

Run small CPU checks serially from `src/`:

```bash
.venv/bin/python -m unittest balanced_l2.tests.test_pipeline \
  balanced_l2.tests.test_brotli_export balanced_l2.tests.test_prepare_client_release
node --test --test-concurrency=1 ../client/test/*.test.js
npm --prefix ../client run build
```
