"""ZIP variant of scrap.py, using its credentials, output paths, and batch entrypoints."""

import os
import time
from functools import partial
from tempfile import TemporaryFile
from urllib.parse import urlsplit
from zipfile import ZipFile

import requests
from urllib3.util import Retry

import scrap


def download_zip_images(gallery_id, gallery_data, size, gallery_folder):
    api_url = f"https://nhentai.net/api/v2/galleries/{gallery_id}/download"
    for attempt in range(3):
        with scrap.scraper.post(api_url, params={"format": "zip"}, timeout=30) as response:
            if response.status_code == 429 and attempt < 2:
                delay = Retry().get_retry_after(response)
            else:
                response.raise_for_status()
                archive_url = response.json()["url"]
                break
        time.sleep(2 if delay is None else delay)

    target = urlsplit(archive_url)
    if target.scheme != "https" or not target.hostname or target.username or target.password:
        raise ValueError("Archive API must return an HTTPS URL without embedded credentials")

    with TemporaryFile() as archive_file:
        # The temporary URL grants access without forwarding our API credentials to the CDN.
        with requests.get(archive_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                archive_file.write(chunk)
        archive_file.seek(0)
        with ZipFile(archive_file) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir()
                and member.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif"))
            ]
            pages = gallery_data["pages"]
            if len(members) != len(pages):
                raise ValueError(f"Archive contains {len(members)} image pages; expected {len(pages)}")
            by_name = {os.path.basename(member.filename): member for member in members}
            if len(by_name) != len(members):
                raise ValueError("Archive contains duplicate image filenames")
            page_names = [os.path.basename(page["path"]) for page in pages]
            if len(set(page_names)) == len(members) and all(name in by_name for name in page_names):
                members = [by_name[name] for name in page_names]
            else:
                numbers = [os.path.splitext(os.path.basename(member.filename))[0] for member in members]
                if not all(number.isdecimal() for number in numbers):
                    raise ValueError("Archive pages must match metadata filenames or have numbered filenames")
                numbers = [int(number) for number in numbers]
                if len(set(numbers)) != len(members):
                    raise ValueError("Archive contains duplicate page numbers")
                members = [member for _, member in sorted(zip(numbers, members))]

            imgs = []
            for index, member in enumerate(members):
                with archive.open(member) as image_file, scrap.Image.open(image_file) as source:
                    imgs.append(scrap._save_image(source, index, size, gallery_folder))
            return imgs


download_and_extract_metadata = partial(scrap.download_and_extract_metadata, image_downloader=download_zip_images)
batch_download_parallel_safe = partial(scrap.batch_download_parallel_safe, image_downloader=download_zip_images)
download_failed = partial(scrap.download_failed, image_downloader=download_zip_images)
clean_in_for = partial(scrap.clean_in_for, image_downloader=download_zip_images)


if __name__ == "__main__":
    download_failed()
    # Example: batch_download_parallel_safe(1, 10)
