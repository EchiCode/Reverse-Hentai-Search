<h1 align="center">WWW ’27 Demo Submission</h1>

<p align="center"><strong>The ACM Web Conference 2027</strong></p>

<p align="center">An adaptation of this project using a different dataset has been submitted to the Demonstrations track.</p>


# Reverse Hentai Search

A static, browser-side reverse image search engine for manga panels.
The browser computes a DINO-vits16 embedding on the CPU, ranks 96,000 balanced
L2 centroids with WebAssembly SIMD, then downloads selected vector postings and
metadata from Cloudflare Pages. Query images stay in the browser.

[Live site](https://echicode.com) · [Release and pipeline details](client/scripts/README.md)

```text
client/  Browser source, assets, tests and build tools
src/     Python embedding, metadata, clustering and data deployment tools
```

The prepared client uses the fully uploaded 96k dataset through immutable
Cloudflare URLs. Dataset files, local credentials, dependencies, generated builds
and benchmark results are excluded from Git. The frontend is plain JavaScript,
HTML and CSS.

## Run locally

From the repository root, use Python 3 to serve the client with the isolation
headers required for multithreaded CPU inference:

```bash
python3 client/scripts/serve-local.py --port 8765
```

Open <http://127.0.0.1:8765/>. The first load downloads the model, library/runtime
and centroid index in parallel. Postings, metadata and result images require
network access as searches need them.

## Test and build the client

Use a current Node.js version with `node --test` support:

```bash
cd client
npm ci
npm test
npm run build
```

Publish `client/dist/` as the static frontend when ready. Keep `_headers` in the
published output. Building or pushing this repository does not upload a dataset.

## Python pipeline

Run pipeline commands from `src/`. The working pipeline uses Python 3.13;
NumPy and Brotli are required for index export and release validation. Training
uses NVIDIA cuVS/CuPy; image embedding uses PyTorch and Transformers. Scraping,
metadata tools and optional codec comparisons have additional dependencies
listed in their imports. A new machine needs a compatible environment installed
before running these tools.

`main.py` generates embeddings, `clustering.py` builds a resumable index,
`balanced_l2/export_posting_quantization.py` exports posting codecs,
`prepare_client_release.py` validates and configures a local dataset, and
`deploy_data.py` uploads and verifies static data batches. With this sibling
layout, pass `--client ../client` to the latter two tools. `build_final.sh`
selects the sibling client automatically.

Scraper credentials are read from the `API_KEY` and `USER_TOKEN` environment
variables; do not commit their values. Source galleries, vectors, metadata and
native build checkpoints are local inputs rather than repository contents.

Upload and verify the complete dataset before switching the frontend to it.
The [release instructions](client/scripts/README.md) cover final builds,
bounded-memory commands, presets and data deployment. The current release is
96,000 LVQ6 centroids with Brotli; postings can switch between LVQ6, LVQ8 and FP16.

## Support

[Buy Me a Coffee](https://www.buymeacoffee.com/echicode)

## 📜 License

This project is licensed under the **(AGPLv3) GNU Affero General Public License version 3**.
See the full license in the [LICENSE](./LICENSE) file.

- Non-commercial use is allowed under the terms of the AGPLv3, with required attribution (credit).  
- For commercial use, please contact the author to obtain a separate commercial license.

You can reach me at [echicode@gmail.com] for commercial licensing inquiries.
