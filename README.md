<h1 align="center">WWW ’27 Demo Submission</h1>

<p align="center"><strong>The ACM Web Conference 2027</strong></p>

<p align="center">A cleaned-up version of this project using a different dataset was submitted to the Demonstrations track.</p>


# 🔞 Reverse Hentai Search

A fully client-side reverse image search engine for hentai/manga.  
Drop in a panel → get the source. Runs entirely in your browser. No servers. No tracking. No shame.

Live Demo: https://echicode.com

---

## 🔍 What It Does

- 🖼️ Reverse-search hentai panels using DINO-vit16 embeddings  
- 🧠 Uses client-side clustering to scale to 10k galleries, which amounts to ~xxx images
- 🪶 Lazy loads metadata (tags, source links, affiliate offers) per cluster  
- 💻 Works fully offline after load  
- 🔐 Privacy-respecting by design  

---

## 🧱 Project Structure

```
reverse-hentai-search/
├── client/           # Frontend (React + TS + Vite)
├── scripts/          # Python tools to generate embeddings, clusters, and metadata
└── README.md
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/echicode/reverse-hentai-search
cd reverse-hentai-search/client
npm install
npm run dev
```

To build for production:

```bash
npm run build
```

Place generated data files (embeddings, cluster IDs, metadata chunks) in the appropriate folder.

---

## 🛠 Preprocessing Workflow (Python)

To build your own search dataset:

1. Extract images and save to galleries; check format compatible with the output of scripts/scrape.py.
- Either run a simple clustering via 
```scripts/main_just_hash.py --no-download``
- Otherwise, for more finetuning, call ```scripts/hash/embed_existing_galleries.py``` and build the clustering via ``scripts/clusters```, with optional modifing the clustering.
2. The full result, clustered and all, is in clusters. Change the folder in the client to this one.

All scripts are modular — you can plug in your own galleries or tweak the embedding pipeline.

---

## 💾 Models

Uses [Xenova's `dinov2-vits16`](https://huggingface.co/Xenova/dino-vits16) ONNX model for WebAssembly-based embedding generation in-browser.

Python pipeline uses the same model (but from OpenAI) for embedding compatibility.

---

## 🔐 Privacy

- No cookies, no telemetry, no external API calls  
- All matching and computation is done locally  
- Metadata is lazy-loaded from static JSON chunks  

---

## 💸 Support

[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/echicode)

Fuel the code. Feed the lust.

---

## 📜 License

This project is licensed under the **(AGPLv3) GNU Affero General Public License version 3**.
See the full license in the [LICENSE](./LICENSE) file.

- Non-commercial use is allowed under the terms of the AGPLv3, with required attribution (credit).  
- For commercial use, please contact the author to obtain a separate commercial license.

You can reach me at [echicode@gmail.com] for commercial licensing inquiries.
