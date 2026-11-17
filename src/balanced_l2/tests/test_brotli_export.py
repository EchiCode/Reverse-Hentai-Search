import gzip
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import brotli

from balanced_l2.export import export_centroids_brotli


class BrotliExportTests(unittest.TestCase):
    def test_whole_index_is_exact_and_static_headers_apply_only_to_brotli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = struct.pack('<8sHHI', b'LVQ6IDX\0', 2, 4, 2) + bytes(8 + 2 * 7)
            (root / 'centroids.lvq6.bin.gz').write_bytes(gzip.compress(raw))
            result = export_centroids_brotli(root)
            encoded = (root / result['centroid_brotli_index']).read_bytes()
            self.assertEqual(brotli.decompress(encoded), raw)
            self.assertEqual(result['centroid_brotli_raw_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertIn('/centroids.lvq6.bin.br\n  Content-Encoding: br', (root / '_headers').read_text())

    def test_oversize_index_does_not_publish_a_partial_brotli_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = struct.pack('<8sHHI', b'LVQ6IDX\0', 2, 4, 2) + bytes(22)
            (root / 'centroids.lvq6.bin.gz').write_bytes(gzip.compress(raw))
            with patch('brotli.compress', return_value=bytes(25 * 1024 ** 2 + 1)):
                with self.assertRaisesRegex(ValueError, 'Pages limit'):
                    export_centroids_brotli(root)
            self.assertFalse((root / 'centroids.lvq6.bin.br').exists())


if __name__ == '__main__':
    unittest.main()
