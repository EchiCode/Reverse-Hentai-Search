import gc
import os
import orjson
from tqdm import tqdm
from merger import load_metadata

INPUT_NEW = "metadata.json"
INPUT_OLD = "temp"
OUTPUT = "metadata_clean.json"

seen = set()

with open(OUTPUT, "wb") as fout:
    fout.write(b"[")

    first = True

    # Process metadata.json
    with open(INPUT_NEW, "rb") as fin:
        for item in tqdm(orjson.loads(fin.read()), desc="new"):
            id_ = item["id"]

            if id_ in seen:
                continue

            t = item.get("thumbnail", "")
            if len(t) <= 100:
                continue

            seen.add(id_)

            if not first:
                fout.write(b",")

            fout.write(orjson.dumps(item))
            first = False

    gc.collect()

    # Process old metadata files one by one
    old_files = [
        os.path.join(INPUT_OLD, f)
        for f in os.listdir(INPUT_OLD)
    ]

    for path in tqdm(old_files, desc="old files"):
        records = load_metadata(path)

        for item in records:
            id_ = item["id"]

            if id_ in seen:
                continue

            t = item.get("thumbnail", "")
            if len(t) <= 100:
                continue

            seen.add(id_)

            if not first:
                fout.write(b",")

            fout.write(orjson.dumps(item))
            first = False

        # Important: free per-file memory
        del records
        gc.collect()

    fout.write(b"]")