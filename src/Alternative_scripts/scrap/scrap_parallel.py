import os
import time

import orjson
import cloudscraper
import requests
from urllib3.util import Retry
from PIL import Image, ImageFile
from io import BytesIO

from torch.fx.experimental import debug
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import base64
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

API_KEY = os.environ.get("API_KEY")
USER_TOKEN = os.environ.get("USER_TOKEN")
scraper = None

def creat_scraper():
    user_token = (USER_TOKEN or "").strip()
    credential = user_token or (API_KEY or "").strip()
    if "\r" in credential or "\n" in credential:
        raise ValueError("API_KEY/USER_TOKEN must be a single-line value")
    # Js2Py 0.74 has an unpatched sandbox escape (CVE-2024-28397).
    session = cloudscraper.create_scraper(
        interpreter="native",
        browser={"browser": "chrome", "platform": "linux", "desktop": True, "mobile": False},
        debug=False,
        enable_stealth=False,
        min_request_interval=0,
        max_concurrent_requests=float("inf"),
        rotate_tls_ciphers=False,
        auto_refresh_on_403=False,
        session_refresh_interval=float("inf"),
    )
    session.headers["Referer"] = "https://nhentai.net/"
    # HTTP status retries are handled by the gallery/archive code.
    session.get_adapter("https://").max_retries = Retry(
        total=3,
        status=0,
        other=0,
        backoff_factor=0.5,
        allowed_methods=("GET", "HEAD", "OPTIONS"),
        raise_on_status=False,
    )
    if credential:
        scheme = "User" if user_token else "Key"
        session.headers["Authorization"] = f"{scheme} {credential}"
    return session

OUTPUT_DIR = "galleries"
LOG_DIR = "logging"
METADATA_PATH = "metadata.json"  # Saved in main folder
IMAGE_SIZE = (256, 256)
loaded = frozenset(np.load('loaded.npy'))
# loaded = frozenset([])

def _save_image(source, idx, size=IMAGE_SIZE, gallery_folder=None):
    if source.mode in ("RGBA", "LA") or "transparency" in source.info:
        img = source.convert("RGBA")
    else:
        img = source.convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    if gallery_folder:
        os.makedirs(gallery_folder, exist_ok=True)
        out_path = os.path.join(gallery_folder, f"{idx + 1:03d}.webp")
        img.save(out_path, format='WebP')
    return img


def download_image(media_id, idx, page_info, size=IMAGE_SIZE, gallery_folder=None):
    ext = page_info["path"].split('.')[-1]
    img_url = f"https://i.nhentai.net/galleries/{media_id}/{idx + 1}.{ext}"
    try:
        with scraper.get(img_url, timeout=15) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                raise ValueError(f"URL did not return an image (Content-Type: {content_type})")
            with Image.open(BytesIO(response.content)) as source:
                return _save_image(source, idx, size, gallery_folder)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download image {img_url}: {e}") from e

def _download_images(gallery_id, gallery_data, size, gallery_folder, max_img_workers):
    pages = gallery_data["pages"]
    imgs = [None] * len(pages)
    with ThreadPoolExecutor(max_workers=max_img_workers) as executor:
        futures = {
            executor.submit(download_image, gallery_data["media_id"], i, page, size, gallery_folder): i
            for i, page in enumerate(pages)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                imgs[idx] = future.result()
            except Exception as e:
                executor.shutdown(wait=True, cancel_futures=True)
                raise RuntimeError(f"Image {idx + 1} download failed: {e}") from e
    return imgs


def download_and_extract_metadata(gallery_id, output_dir=OUTPUT_DIR, size=IMAGE_SIZE, max_img_workers=1, tries=0, image_downloader=None):
    if gallery_id in loaded:
        return None, f"✅ Gallery {gallery_id} was already downloaded, so it was skipped",False

    try:
        api_url = f"https://nhentai.net/api/v2/galleries/{gallery_id}"
        response = scraper.get(api_url, timeout=30)

        if response.status_code==429:
            print('retrying',gallery_id)
            if tries<2:
                response.close()
                time.sleep(2)
                return download_and_extract_metadata(gallery_id, output_dir, size, max_img_workers, tries + 1, image_downloader)
        response.close()
        response.raise_for_status()
    except Exception as e:
        with open("failed.txt", "a") as f:
            f.write(f"{gallery_id}\n")
        return None, f"❌ Failed to fetch gallery metadata {gallery_id}: {e}",True

    try:
        gallery_data = response.json()
        if not isinstance(gallery_data, dict):
            raise ValueError("Gallery metadata must be a JSON object")
    except Exception as e:
        with open("failed.txt", "a") as f:
            f.write(f"{gallery_id}\n")
        return None, f"❌ Failed to parse JSON for gallery {gallery_id}: {e}",True

    response.close()

    media_id = gallery_data.get("media_id")
    if media_id is None:
        with open("failed.txt", "a") as f:
            f.write(f"{gallery_id}\n")
        return None, f"❌ Gallery {gallery_id} missing media_id",True

    image_info = gallery_data.get("pages", [])
    if not image_info:
        with open("failed.txt", "a") as f:
            f.write(f"{gallery_id}\n")
        return None, f"❌ Gallery {gallery_id} has no images info",True

    full_output = os.path.join(output_dir, str(gallery_id))
    os.makedirs(full_output, exist_ok=True)

    try:
        downloader = image_downloader or _download_images
        imgs = downloader(gallery_id, gallery_data, size, full_output, max_img_workers)
        if not imgs:
            raise ValueError("Gallery downloader returned no images")
    except Exception as e:
        print(e)
        with open("failed.txt", "a") as f:
            f.write(f"{gallery_id}\n")
        return None, f"❌ Gallery {gallery_id} download failed: {e}", True


    try:
        # Create binary thumbnail from first image
        tumb = None
        if imgs[0]:
            thumb_img = imgs[0]# .resize((196, 196), Image.LANCZOS)
            buffer = BytesIO()
            thumb_img.save(buffer, format='WebP', quality=80, lossless=False)
            buffer.seek(0)
            tumb_b = buffer.read()  # Store as binary data
            tumb = base64.b64encode(tumb_b).decode('utf-8')

        # Titles: prefer pretty, then English, then Japanese
        title = (
            gallery_data.get("title", {}).get("pretty")
            or gallery_data.get("title", {}).get("english")
            or gallery_data.get("title", {}).get("japanese")
            or "Untitled"
        )

        # Extract tags and characters with counts
        tags = gallery_data.get("tags", [])
        all_tags = [(tag["name"], tag.get('count', 0)) for tag in tags if tag.get("type") == "tag"]
        all_chars = [(tag["name"], tag.get('count', 0)) for tag in tags if tag.get("type") == "character"]
        top_tags = [t[0] for t in sorted(all_tags, key=lambda x: -x[1])[:5]]
        top_chars = [t[0] for t in sorted(all_chars, key=lambda x: -x[1])[:5]]

        meta = {
            "id": gallery_id,
            "name": title.strip(),
            "top_tags": top_tags,
            "top_chars": top_chars,
            "thumbnail": tumb,  # Store path to binary thumbnail file
        }
    except Exception as e:
        with open("failed.txt", "a") as f:
            f.write(f"{gallery_id}\n")
        return None, f"❌ Failed to process gallery metadata {gallery_id}: {e}", True

    del imgs
    del response

    return meta, f"✅ Gallery {gallery_id} downloaded",False


def _read_gallery_ids(path):
    with open(path, "r") as f:
        ids = {int(line) for line in f if line.strip()}
    if any(gid <= 0 for gid in ids):
        raise ValueError(f"Gallery IDs must be positive integers: {path}")
    return ids


def _load_metadata():
    metadata = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            metadata = orjson.loads(f.read())
    if not isinstance(metadata, list):
        raise ValueError(f"Expected a metadata list: {METADATA_PATH}")
    if any(not isinstance(entry, dict) or type(entry.get("id")) is not int or entry["id"] <= 0
           for entry in metadata):
        raise ValueError(f"Invalid gallery ID in metadata: {METADATA_PATH}")
    return metadata


import gc
def batch_download_parallel_safe(start=0, end=1000, max_workers=1, batch_id="default", image_downloader=None):
    global scraper
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    downloaded_file = os.path.join(log_dir, "downloaded.txt")
    downloaded_set = set()

    if os.path.exists(downloaded_file):
        downloaded_set = _read_gallery_ids(downloaded_file)


    to_download = set(range(start, end)) - downloaded_set
    to_download = to_download - loaded

    existing_metadata = _load_metadata()
    to_download -= {entry["id"] for entry in existing_metadata}

    print(f"⏳ {len(to_download)} galleries to download")

    metadata = []
    failed_galleries = []
    G = []
    scraper = creat_scraper()
    try:
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
                total=len(to_download), desc=f"Batch {batch_id} Downloading galleries"
            ) as pbar:
                futures = {executor.submit(download_and_extract_metadata, gid, OUTPUT_DIR, IMAGE_SIZE, image_downloader=image_downloader): gid for gid in to_download}
                for future in as_completed(futures):
                    gid = futures[future]
                    meta, result, fail = future.result()
                    # messages.append(result)
                    if meta:
                        metadata.append(meta)
                        G.append(gid)
                    else:
                        if fail:
                            failed_galleries.append(gid)

                        # messages.append(result)
                    pbar.update(1)
        else:
            pbar = tqdm(total=len(to_download), desc=f"Batch {batch_id} Downloading galleries")

            for gid in to_download:
                meta, result, fail = download_and_extract_metadata(gid, OUTPUT_DIR, IMAGE_SIZE, image_downloader=image_downloader)

                if meta:
                    metadata.append(meta)
                    G.append(gid)
                else:
                    if fail:
                        failed_galleries.append(gid)

                pbar.update(1)

            pbar.close()

        existing_metadata.extend(metadata)
        temporary_metadata = f"{METADATA_PATH}.retry.tmp"
        with open(temporary_metadata, "wb") as f:
            f.write(orjson.dumps(existing_metadata))
        os.replace(temporary_metadata, METADATA_PATH)

        with open(downloaded_file, "a") as f:
            f.writelines(f"{gid}\n" for gid in G)

        if len(failed_galleries)>0:
            time.sleep(0.25)

        for gid in failed_galleries:
            shutil.rmtree(os.path.join(OUTPUT_DIR, str(gid)), ignore_errors=True)

        print(f"✅ Appended {len(metadata)} new entries to {METADATA_PATH}")
    finally:
        scraper.cookies.clear()
        scraper.close()
        del scraper

    del metadata
    del existing_metadata

    gc.collect()
    return len(G)




def download_failed(max_workers=3, batch_id="default", image_downloader=None, failed_file=None):
    global scraper
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    if failed_file is None:
        failed_file = "failed_.txt"
    if os.path.exists(failed_file):
        to_download = _read_gallery_ids(failed_file)
    else:
        return 0

    downloaded_file = os.path.join(log_dir, "downloaded.txt")
    downloaded_set = set()

    if os.path.exists(downloaded_file):
        downloaded_set = _read_gallery_ids(downloaded_file)


    to_download = to_download - downloaded_set
    to_download = to_download - loaded

    existing_metadata = _load_metadata()
    to_download -= {entry["id"] for entry in existing_metadata}

    print(f"⏳ {len(to_download)} galleries to download")

    metadata = []
    failed_galleries = []
    G = []
    scraper = creat_scraper()
    try:
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
                total=len(to_download), desc=f"Batch {batch_id} Downloading galleries"
            ) as pbar:
                futures = {executor.submit(download_and_extract_metadata, gid, OUTPUT_DIR, IMAGE_SIZE, image_downloader=image_downloader): gid for gid in to_download}
                for future in as_completed(futures):
                    gid = futures[future]
                    meta, result, fail = future.result()
                    # messages.append(result)
                    if meta:
                        metadata.append(meta)
                        G.append(gid)
                    else:
                        if fail:
                            failed_galleries.append(gid)

                        # messages.append(result)
                    pbar.update(1)
        else:
            pbar = tqdm(total=len(to_download), desc=f"Batch {batch_id} Downloading galleries")

            for gid in to_download:
                meta, result, fail = download_and_extract_metadata(gid, OUTPUT_DIR, IMAGE_SIZE, image_downloader=image_downloader)

                if meta:
                    metadata.append(meta)
                    G.append(gid)
                else:
                    if fail:
                        failed_galleries.append(gid)

                pbar.update(1)

            pbar.close()

        existing_metadata.extend(metadata)
        temporary_metadata = f"{METADATA_PATH}.retry.tmp"
        with open(temporary_metadata, "wb") as f:
            f.write(orjson.dumps(existing_metadata))
        os.replace(temporary_metadata, METADATA_PATH)

        with open(downloaded_file, "a") as f:
            f.writelines(f"{gid}\n" for gid in G)

        if len(failed_galleries)>0:
            time.sleep(0.25)

        for gid in failed_galleries:
            shutil.rmtree(os.path.join(OUTPUT_DIR, str(gid)), ignore_errors=True)

        print(f"✅ Appended {len(metadata)} new entries to {METADATA_PATH}")
    finally:
        scraper.cookies.clear()
        scraper.close()
        del scraper

    del metadata
    del existing_metadata

    gc.collect()
    return len(G)

def clean_in_for(max_workers=1, image_downloader=None):
    for _ in range(1):
        if os.path.exists("failed_.txt"):
            raise FileExistsError("Pending failed_.txt must be retried before replacing it")
        os.rename("failed.txt", "failed_.txt")
        download_failed(max_workers=max_workers, image_downloader=image_downloader)
        os.remove("failed_.txt")

        time.sleep(2)


if __name__ == "__main__":
    # clean_in_for()
    download_failed(max_workers=1)
    # Example usage
    # batch_download_parallel_safe(611060, 611061, max_workers=1)
