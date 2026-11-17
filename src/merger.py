import struct
import gzip
import json
import numpy as np
import os
import re
from copy import deepcopy as copy
from tqdm import tqdm

def write_string(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<H', len(b)))
    f.write(b)

def write_string_list(f, lst):
    f.write(struct.pack('<B', len(lst)))
    for s in lst:
        write_string(f, s)

def write_metadata_entry(f, entry):
    f.write(struct.pack('<i', int(entry['id'])))
    write_string(f, entry.get('name', ''))
    write_string_list(f, entry.get('top_tags', []))
    write_string_list(f, entry.get('top_chars', []))
    thumb_64 = entry.get('thumbnail', '')
    thumb = thumb_64.encode('utf-8')
    f.write(struct.pack('<I', len(thumb)))
    f.write(thumb)

def string_to_bytes(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<H', len(b)) + b

def string_list_to_bytes(lst: list[str]) -> bytes:
    chunks = [struct.pack('<B', len(lst))]
    for s in lst:
        chunks.append(string_to_bytes(s))
    return b''.join(chunks)

def metadata_entry_to_bytes(entry: dict) -> bytes:
    chunks = [struct.pack('<i', int(entry['id']))]
    chunks.append(string_to_bytes(entry.get('name', '')))
    chunks.append(string_list_to_bytes(entry.get('top_tags', [])))
    chunks.append(string_list_to_bytes(entry.get('top_chars', [])))
    thumb = entry.get('thumbnail', '').encode('utf-8')
    chunks.append(struct.pack('<I', len(thumb)))
    chunks.append(thumb)
    return b''.join(chunks)

def read_string(f):
    len_bytes = f.read(2)
    if not len_bytes or len(len_bytes) < 2:
        return '', 0
    l = struct.unpack('<H', len_bytes)[0]
    s = f.read(l).decode('utf-8')
    return s, 2 + l

def read_string_list(f):
    count_bytes = f.read(1)
    if not count_bytes or len(count_bytes) < 1:
        return [], 1
    count = struct.unpack('<B', count_bytes)[0]
    lst = []
    total = 1
    for _ in range(count):
        s, l = read_string(f)
        lst.append(s)
        total += l
    return lst, total

def read_metadata_entry(f):
    code_bytes = f.read(4)
    if not code_bytes or len(code_bytes) < 4:
        return None, 0
    code = struct.unpack('<i', code_bytes)[0]
    name, l1 = read_string(f)
    top_tags, l2 = read_string_list(f)
    top_chars, l3 = read_string_list(f)
    thumb_len_bytes = f.read(4)
    if not thumb_len_bytes or len(thumb_len_bytes) < 4:
        return None, 0
    thumb_len = struct.unpack('<I', thumb_len_bytes)[0]
    thumb = f.read(thumb_len)
    thumb = thumb.decode('utf-8')
    total = 4 + l1 + l2 + l3 + 4 + thumb_len
    return {'id': code, 'name': name, 'top_tags': top_tags, 'top_chars': top_chars, 'thumbnail': thumb}, total


import os

def append_vectors(old='vectors_old', new='vectors', save='vectors_new'):
    os.makedirs(save, exist_ok=True)

    # Sort files to maintain order
    old_files = sorted(os.listdir(old))
    new_files = sorted(os.listdir(new))

    # Copy old files as-is
    for idx, f in enumerate(old_files, start=1):
        old_path = os.path.join(old, f)
        save_path = os.path.join(save, f"{idx}.bin")
        with open(old_path, 'rb') as fin, open(save_path, 'wb') as fout:
            fout.write(fin.read())

    # Append new files after old files
    offset = len(old_files)
    for idx, f in enumerate(new_files, start=1):
        new_path = os.path.join(new, f)
        save_path = os.path.join(save, f"{offset + idx}.bin")
        with open(new_path, 'rb') as fin, open(save_path, 'ab') as fout:
            fout.write(fin.read())

    print(f"All old and new vectors saved in '{save}' folder in sequential order.")

from operator import itemgetter

def load_metadata(path):
    res = []
    with gzip.open(path, "rb") as f_in:
        while True:
            entry, used = read_metadata_entry(f_in)
            if entry is None or used == 0:
                break
            res.append(entry)
    res_ = sorted(res,key=itemgetter('id'))
    return res_

import orjson

def append_meta_only_last(old='meta_old', new='metadata.json', save='meta_new', chunk_size=500):
    os.makedirs(save, exist_ok=True)

    # Extract last file, might have fewer than chunk_size entries
    old_files = os.listdir(old)
    keys = []
    for f in old_files:
        f1 = f[5:]
        f2 = f1[:-7]
        keys.append(int(f2))

    I = np.argsort(keys)
    old_files = list(np.array(old_files)[I])
    last = old_files.pop()

    last_path = os.path.join(old, last)
    last_data = load_metadata(last_path)

    # Copy old files as-is
    for f in old_files:
        old_path = os.path.join(old, f)
        save_path = os.path.join(save, f)
        with open(old_path, 'rb') as fin, open(save_path, 'wb') as fout:
            fout.write(fin.read())

    # Load new metadata JSON with orjson
    with open(new, 'rb') as fin:
        new_meta = orjson.loads(fin.read())

    # Merge old and new metadata
    all_meta = last_data + new_meta

    # Ensure entries are sorted by id
    all_meta.sort(key=lambda x: x["id"])

    if not all_meta:
        return

    current_idx = None
    f = None

    for entry in all_meta:
        expected_idx = entry["id"] // chunk_size
        if expected_idx != current_idx:
            if f is not None:
                f.close()
            current_idx = expected_idx
            save_path = os.path.join(save, f"meta_{current_idx}.bin.gz")
            f = gzip.open(save_path, 'wb')
        write_metadata_entry(f, entry)

    if f is not None:
        f.close()

    print(f"Old meta copied and new entries saved as .bin.gz in id-ranged chunks of {chunk_size} in '{save}'.")

def append_meta(old='meta', new='metadata.json', save='meta_new', chunk_size=250):
    # Load new metadata JSON once
    with open(new, 'rb') as fin:
        data = orjson.loads(fin.read())

    old_files = [os.path.join(old, f) for f in os.listdir(old)]
    for path in tqdm(old_files):
        data+= load_metadata(path)

    all_entries = list(data)
    print(len(all_entries))
    all_entries.sort(key=lambda e: e['id'])

    os.makedirs(save, exist_ok=True)
    current_idx = None
    f = None
    buffer = []

    for entry in tqdm(all_entries):
        expected_idx = entry["id"] // chunk_size
        if expected_idx != current_idx:
            # flush previous buffer
            if f is not None:
                f.write(b''.join(buffer))
                f.close()
            current_idx = expected_idx
            save_path = os.path.join(save, f"meta_{current_idx}.bin.gz")
            f = gzip.open(save_path, 'wb')
            buffer = []

        buffer.append(metadata_entry_to_bytes(entry))

    # flush last buffer
    if f is not None:
        f.write(b''.join(buffer))
        f.close()


if __name__ == "__main__":
    # arr = [load_metadata('meta_new/meta_'+str(i)+'.bin.gz') for i in range(401)]
    # print([len(a) for a in arr])

    append_vectors()
    append_meta()
