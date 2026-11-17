"""Source assets resolve in the development and exported GitHub layouts."""

from pathlib import Path
import tempfile
import unittest

from balanced_l2.common import source_path


class SourcePathTests(unittest.TestCase):
    def test_client_assets_and_source_fingerprints_work_in_both_layouts(self):
        for nested in (True, False):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'src'
                source.mkdir()
                (source / 'benchmark.py').write_text('benchmark')
                client = source / 'client' if nested else root / 'client'
                scripts = client / 'scripts'
                scripts.mkdir(parents=True)
                encoder = scripts / 'encode-centroids.py'
                encoder.write_text('encoder')
                self.assertEqual(source_path(source, 'client'), client)
                self.assertEqual(source_path(source, 'client/data'), client / 'data')
                self.assertEqual(source_path(source, 'client/scripts/encode-centroids.py').read_text(), 'encoder')
                self.assertEqual(source_path(source, 'benchmark.py').read_text(), 'benchmark')
                if nested:
                    (root / 'client').mkdir()
                    self.assertEqual(source_path(source, 'client'), client)


if __name__ == '__main__':
    unittest.main()
