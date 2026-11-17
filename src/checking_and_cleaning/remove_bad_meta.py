import gzip
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import numpy as np

from merger import read_metadata_entry

A = set(np.load("loaded_new.npy"))
B = set(np.load("vec_codes.npy"))
bad = A-B

removed = 0
for path in tqdm(Path("meta_new").glob("meta_*.bin.gz")):
    data = gzip.decompress(path.read_bytes())
    stream = BytesIO(data)
    kept = bytearray()
    while stream.tell() < len(data):
        start = stream.tell()
        entry, used = read_metadata_entry(stream)
        if entry is None or used != stream.tell() - start:
            raise ValueError(f"Incomplete metadata entry in {path}")
        if entry["id"] in bad:
            removed += 1
        else:
            kept.extend(data[start:stream.tell()])

    if len(kept) == len(data):
        continue
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_bytes(gzip.compress(kept))
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)

print(f"Removed {removed} entries from meta_new")
