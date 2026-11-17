import os
import orjson
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import shutil
import gzip
import struct
from contextlib import suppress
from operator import index

from balanced_l2.common import iter_record_batches


def embed_existing_galleries(
    gallery_root="galleries",
    output_dir="vectors",
    batch_size=128,  # be careful the batching is per gallery, so for say 1024 it will work for most before crashing at ~50k...
    save_every=1_000_000_000,
    resize=(256, 256),
    model_name="facebook/dino-vits16",
    offset = 0,
    device=None,
    delete_gallery=True
):
    """Skip committed pages on restart.

    Use a fresh output directory if the model, resize, or existing image contents change.
    """
    batch_size = index(batch_size)
    offset = index(offset)
    if batch_size <= 0 or offset < 0:
        raise ValueError("batch_size must be positive and offset must be non-negative")
    os.makedirs(output_dir, exist_ok=True)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    all_gallery_folders = sorted(
        [f for f in os.listdir(gallery_root) if os.path.isdir(os.path.join(gallery_root, f)) and f.isdigit()],
        key=lambda x: int(x)
    )
    gallery_ids = [int(gallery_id) for gallery_id in all_gallery_folders]
    if any(code <= 0 or code > 2147483647 for code in gallery_ids):
        raise ValueError("Gallery IDs must fit positive int32 records")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("Duplicate numeric gallery IDs")
    if delete_gallery:
        output_path = os.path.realpath(output_dir)
        for gallery_id in all_gallery_folders:
            gallery_path = os.path.realpath(os.path.join(gallery_root, gallery_id))
            if os.path.commonpath((gallery_path, output_path)) == gallery_path:
                raise ValueError("output_dir must be outside galleries that will be deleted")

    model = None

    # Automatically determine batch number from existing files
    existing_batches = [
        f for f in os.listdir(output_dir) if f.startswith("dino_vectors_batch_") and f.endswith(".bin")
    ]
    existing_ids = [
        int(f.replace("dino_vectors_batch_", "").replace(".bin", ""))
        for f in existing_batches if f.replace("dino_vectors_batch_", "").replace(".bin", "").isdigit()
    ]
    existing_files = [
        os.path.join(output_dir, name) for name in existing_batches
        if name.replace("dino_vectors_batch_", "").replace(".bin", "").isdigit()
    ]
    batch_num = max(existing_ids, default=-1) + 1+offset
    batch_orig = batch_num

    db = []
    processed = 0
    saved_pages = None if any(os.path.getsize(path) for path in existing_files) else {}
    dimensions = None
    pending_deletions = []

    def save_batch():
        nonlocal db, batch_num
        out_path = os.path.join(output_dir, f"dino_vectors_batch_{batch_num}.bin")
        temporary_path = out_path + ".tmp"
        try:
            with open(temporary_path, "wb") as f:
                for entry in db:
                    code = int(entry['code'])
                    page = int(entry['page'])
                    vector = np.array(entry['vector'], dtype=np.float16)
                    f.write(struct.pack('<i', code))
                    f.write(struct.pack('<h', page))
                    f.write(vector.tobytes())
            os.replace(temporary_path, out_path)
        except BaseException:
            with suppress(OSError):
                os.remove(temporary_path)
            raise
        for entry in db:
            saved_pages.setdefault(int(entry['code']), set()).add(entry['page'])
        print(f"\u2705 Saved batch {batch_num} with {len(db)} vectors \u2192 {out_path}")
        db = []
        batch_num += 1
        for gallery_path in pending_deletions:
            shutil.rmtree(gallery_path)
        pending_deletions.clear()

    for gallery_id in tqdm(all_gallery_folders, desc="Embedding galleries"):
        gallery_path = os.path.join(gallery_root, gallery_id)
        entries = os.listdir(gallery_path)
        image_files = sorted([
            os.path.join(gallery_path, f)
            for f in entries
            if f.lower().endswith((".webp",".png", ".jpg", ".jpeg"))
            and os.path.isfile(os.path.join(gallery_path, f))
        ])
        gallery_failed = len(image_files) != len(entries)

        try:
            page_ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in image_files]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric page filename in gallery {gallery_id}") from exc
        if any(page <= 0 or page > 32767 for page in page_ids):
            raise ValueError(f"Page IDs in gallery {gallery_id} must be between 1 and 32767")
        if len(set(page_ids)) != len(page_ids):
            raise ValueError(f"Duplicate page IDs in gallery {gallery_id}")

        if saved_pages is None:
            dimensions = index(AutoConfig.from_pretrained(model_name).hidden_size)
            if dimensions <= 0:
                raise ValueError("Model embedding dimension must be positive")
            saved_pages = {}
            for records in iter_record_batches(existing_files, dimensions, batch_size=65536):
                if ((records['code'] <= 0).any() or (records['page'] <= 0).any()
                        or not np.isfinite(records['vec']).all()
                        or not records['vec'].any(axis=1).all()):
                    raise ValueError("Invalid committed embedding records")
                wanted = np.isin(records['code'], gallery_ids)
                for code, page in zip(records['code'][wanted], records['page'][wanted]):
                    saved_pages.setdefault(int(code), set()).add(int(page))

        if not image_files:
            if delete_gallery and saved_pages.get(int(gallery_id)) and not os.listdir(gallery_path):
                os.rmdir(gallery_path)
            continue

        committed = saved_pages.get(int(gallery_id), set())
        remaining = [(path, page) for path, page in zip(image_files, page_ids)
                     if page not in committed]
        if not remaining:
            if delete_gallery and not gallery_failed:
                shutil.rmtree(gallery_path)
            continue
        image_files, page_ids = zip(*remaining)

        for batch_idx in range(0, len(image_files), batch_size):
            batch_files = image_files[batch_idx:batch_idx + batch_size]
            imgs = []
            meta = []

            for img_path, page in zip(batch_files, page_ids[batch_idx:batch_idx + batch_size]):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(resize, Image.LANCZOS)
                    imgs.append(img)
                    meta.append((gallery_id, page))
                except Exception:
                    gallery_failed = True
                    continue

            if not imgs:
                continue

            if model is None:
                model = AutoModel.from_pretrained(model_name).to(device).eval()
                processor = AutoImageProcessor.from_pretrained(model_name, backend="torchvision")
            inputs = processor(images=imgs, return_tensors="pt").to(device)

            with torch.no_grad():
                features = model(**inputs).last_hidden_state[:, 0]  # Use CLS token
                if features.ndim != 2 or features.shape[0] != len(meta) or not features.shape[1]:
                    raise ValueError("Model must return one nonempty embedding per image")
                if dimensions is None:
                    dimensions = features.shape[1]
                elif features.shape[1] != dimensions:
                    raise ValueError("Model embedding dimensions differ from committed records")
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                features = features.cpu().numpy()
                if not np.isfinite(features).all():
                    raise ValueError("Model returned non-finite image embeddings")

            for (code, page), vec in zip(meta, features):
                db.append({"code": code, "page": page, "vector": vec.tolist()})
                processed += 1

            if len(db) >= save_every:
                save_batch()
        if delete_gallery and not gallery_failed:
            if db:
                pending_deletions.append(gallery_path)
            else:
                shutil.rmtree(gallery_path)

    if db:
        save_batch()

    print(f"🎉 Done! Embedded {processed} images across {batch_num-batch_orig} total batches.")
    return batch_num

if __name__ == '__main__':
    embed_existing_galleries()
