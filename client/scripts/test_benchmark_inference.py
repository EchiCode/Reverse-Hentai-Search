import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image


spec = importlib.util.spec_from_file_location('gallery_benchmark', Path(__file__).with_name('benchmark-inference.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class GalleryInputsTest(unittest.TestCase):
    def test_samples_real_pages_across_galleries_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            galleries, site = root / 'galleries', root / 'site'
            galleries.mkdir()
            site.mkdir()
            for gallery in (1, 2, 5, 8, 9):
                folder = galleries / str(gallery)
                folder.mkdir()
                for page in range(1, 6):
                    Image.new('RGB', (12, 17), (gallery, page, 0)).save(folder / f'{page:03d}.png')
            (galleries / 'thumbnails').mkdir()
            inputs = benchmark.prepare_images(galleries, site)
            self.assertEqual([item['image'] for item in inputs],
                             [f'{gallery}/{page:03d}.png' for gallery in (1, 5, 9) for page in (1, 3, 5)])
            for item in inputs:
                original = Path(item['source']).read_bytes()
                self.assertEqual((item['width'], item['height']), (12, 17))
                self.assertEqual(item['sha256'], hashlib.sha256(original).hexdigest())
                self.assertEqual((site / item['url'].lstrip('/')).read_bytes(), original)

    def test_empty_gallery_root_cannot_fall_back_to_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'No gallery folders'):
                benchmark.prepare_images(root, root)


if __name__ == '__main__':
    unittest.main()
