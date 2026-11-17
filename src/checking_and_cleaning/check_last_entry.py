import gzip
import struct

import gzip
import struct

def load_meta_codes(path):
    """Load a .bin.gz metadata chunk and return a list of codes only."""
    codes = []
    with gzip.open(path, 'rb') as f:
        while True:
            code_bytes = f.read(4)
            if not code_bytes:
                break  # EOF
            code = struct.unpack('<i', code_bytes)[0]
            codes.append(code)

            # Skip the rest of the entry
            name_len_bytes = f.read(2)
            if not name_len_bytes:
                break
            name_len = struct.unpack('<H', name_len_bytes)[0]
            f.seek(name_len, 1)  # skip name

            # Skip top_tags
            tags_count_bytes = f.read(1)
            tags_count = struct.unpack('<B', tags_count_bytes)[0]
            for _ in range(tags_count):
                tag_len = struct.unpack('<H', f.read(2))[0]
                f.seek(tag_len, 1)

            # Skip top_chars
            chars_count_bytes = f.read(1)
            chars_count = struct.unpack('<B', chars_count_bytes)[0]
            for _ in range(chars_count):
                char_len = struct.unpack('<H', f.read(2))[0]
                f.seek(char_len, 1)

            # Skip thumbnail
            thumb_len = struct.unpack('<I', f.read(4))[0]
            f.seek(thumb_len, 1)

    return codes


if __name__ == '__main__':
    A = load_meta_codes('meta_1216.bin.gz')
    A = sorted(A)
    print(A[-1])
    print(len(A)+A[0])