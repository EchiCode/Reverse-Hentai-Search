"""Tiny filesystem checks; no model, browser, server or GPU is started."""

import brotli
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import prepare_client_release as release


class PrepareClientReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='client-release-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.meta, self.client = (self.root / name for name in ('build', 'meta', 'client'))
        for path in (self.data, self.meta, self.client):
            path.mkdir()
        self.link_dir = self.client / 'data'
        self.link_dir.mkdir()
        self.config = self.client / 'search-release.js'
        self.config.write_text('globalThis.SEARCH_RELEASE = null;\n')
        self.raw = (struct.pack('<8sHHI', b'LVQ6IDX\0', 2, 384, 3) + bytes(384 * 2 + 3 * 2)
                    + struct.pack('<e', .01) * 3 + bytes(3 * 288))
        compressed = gzip.compress(self.raw, mtime=0)
        (self.data / 'centroids.lvq6.bin.gz').write_bytes(compressed)
        (self.data / 'centroids.lvq6.bin.br').write_bytes(brotli.compress(self.raw))
        self.manifest = {'schema_version': 1, 'head_count': 3, 'dimensions': 384,
                         'centroid_sha256': hashlib.sha256(compressed).hexdigest(), 'formats': {}}
        for format_name, directory in release.FORMATS.items():
            folder = self.data / directory
            folder.mkdir()
            sizes = []
            for cid in range(3):
                content = gzip.compress(bytes([cid]), mtime=0)
                (folder / f'cluster_{cid}.bin.gz').write_bytes(content)
                sizes.append(len(content))
            self.manifest['formats'][format_name] = {'path': directory, 'gzip_bytes': sizes,
                                                     'total_gzip_bytes': sum(sizes)}
        self._manifest()
        for mid in (0, 2, 2788):
            (self.meta / f'meta_{mid}.bin.gz').write_bytes(gzip.compress(bytes([mid % 256]), mtime=0))

    def _manifest(self):
        (self.data / 'posting-codecs.json').write_text(json.dumps(self.manifest))

    def _prepare(self, **kwargs):
        return release.prepare(self.data, self.meta, self.client, **kwargs)

    def test_complete_release_replaces_dangling_links_and_publishes_bound_config(self):
        for name in ('centroids.lvq6.bin.gz', 'clusters', 'clusters_lvq6', 'clusters_lvq8', 'meta', 'posting-codecs.json'):
            (self.link_dir / name).symlink_to(self.root / 'deleted' / name)
        result = self._prepare()
        self.assertEqual(result['headCount'], 3)
        self.assertEqual(result['centroidSha256'], self.manifest['centroid_sha256'])
        self.assertEqual(result['centroidRawSha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertTrue(result['localOnly'])
        self.assertEqual(result['centroids'], 'data/centroids.lvq6.bin.br')
        self.assertEqual(set(result['clusters']), set(release.FORMATS))
        for directory in release.FORMATS.values():
            self.assertEqual((self.link_dir / directory).resolve(), self.data / directory)
        self.assertEqual((self.link_dir / 'meta').resolve(), self.meta)
        self.assertEqual((self.link_dir / 'centroids.lvq6.bin.br').resolve(), self.data / 'centroids.lvq6.bin.br')
        configured = json.loads(self.config.read_text().split(' = ', 1)[1].rstrip(';\n'))
        self.assertEqual(configured, result)
        self.assertFalse(list(self.link_dir.glob('.*.tmp')))

    def test_incomplete_or_corrupt_data_never_updates_config_or_links(self):
        original = self.config.read_text()
        missing = self.data / 'clusters_lvq8/cluster_2.bin.gz'
        saved = missing.read_bytes()
        missing.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete posting files'):
            self._prepare()
        self.assertEqual(self.config.read_text(), original)
        self.assertEqual(list(self.link_dir.iterdir()), [])
        missing.write_bytes(saved)
        self.manifest['formats']['fp16']['gzip_bytes'][0] += 1
        self._manifest()
        with self.assertRaises(ValueError):
            self._prepare()
        self.manifest['formats']['fp16']['gzip_bytes'][0] -= 1
        self._manifest()
        (self.data / 'centroids.lvq6.bin.br').write_bytes(brotli.compress(b'different'))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self._prepare()
        self.assertEqual(self.config.read_text(), original)
        self.assertEqual(list(self.link_dir.iterdir()), [])

    def test_physical_client_entries_are_preserved(self):
        physical = self.link_dir / 'clusters'
        physical.mkdir()
        (physical / 'keep.txt').write_text('keep')
        original = self.config.read_text()
        with self.assertRaisesRegex(ValueError, 'physical client data'):
            self._prepare()
        self.assertEqual((physical / 'keep.txt').read_text(), 'keep')
        self.assertEqual(self.config.read_text(), original)

    def test_remote_routes_are_explicitly_bound_https_and_local_until_requested(self):
        local = self._prepare()
        routes = {**local, 'centroids': 'https://release.example/centroids.lvq6.bin.br',
                  'formats': 'https://release.example/posting-codecs.json',
                  'clusters': {name: f'https://v1-b{{batch}}.example/{name}/cluster_{{id}}.bin.gz'
                               for name in release.FORMATS},
                  'meta': 'https://v1-m{batch}.example/meta_{id}.bin.gz'}
        routes.pop('localOnly')
        self.assertTrue(self._prepare(routes=routes)['localOnly'])
        routes['localOnly'] = False
        self.assertFalse(self._prepare(routes=routes)['localOnly'])
        good = self.config.read_text()
        for key, value in [('centroidSha256', '0' * 64), ('centroids', 'http://release.example/file.br'),
                           ('meta', 'https://release.example/meta_{other}.bin.gz')]:
            invalid = {**routes, key: value}
            with self.assertRaises(ValueError):
                self._prepare(routes=invalid)
            self.assertEqual(self.config.read_text(), good)

    def test_remote_route_arrays_cover_all_batch_slots_and_validate_every_url(self):
        local = self._prepare()
        routes = {**local, 'centroids': 'https://release.example/centroids.lvq6.bin.br',
                  'formats': 'https://release.example/posting-codecs.json',
                  'clusters': {name: [f'https://deploy-{name}.example/cluster_{{id}}.bin.gz']
                               for name in release.FORMATS},
                  'meta': [f'https://deploy-meta-{batch}.example/meta_{{id}}.bin.gz'
                           for batch in range(12)]}
        routes.pop('localOnly')
        result = self._prepare(routes=routes)
        self.assertTrue(result['localOnly'])
        self.assertEqual(result['clusters'], routes['clusters'])
        self.assertEqual(result['meta'], routes['meta'])
        good = self.config.read_text()
        invalid_entries = [
            ('centroids', [routes['centroids']]), ('formats', [routes['formats']]),
            ('clusters', {**routes['clusters'], 'lvq8': []}),
            ('clusters', {**routes['clusters'], 'lvq8': routes['clusters']['lvq8'] * 2}),
            ('meta', routes['meta'][:-1]), ('meta', routes['meta'] + [routes['meta'][-1]]),
            ('meta', routes['meta'][:-1] + [None]),
            ('meta', routes['meta'][:-1] + ['https://release.example/meta_{other}.bin.gz']),
        ]
        for unsafe in ('http://release.example/meta_{id}.bin.gz',
                       'https://user@release.example/meta_{id}.bin.gz',
                       'https://release.example/%2e%2e/meta_{id}.bin.gz',
                       'https://release.example/meta_{id}.bin.gz?query=1'):
            invalid_entries.append(('meta', routes['meta'][:-1] + [unsafe]))
        for key, value in invalid_entries:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self._prepare(routes={**routes, key: value})
                self.assertEqual(self.config.read_text(), good)

        # Two posting batches are needed as soon as the head count exceeds 1000.
        larger = {**local, 'headCount': 1001}
        two_batches = {**routes, 'headCount': 1001,
                       'clusters': {name: urls * 2 for name, urls in routes['clusters'].items()}}
        self.assertEqual(release._remote_routes(two_batches, larger, 12)['clusters'], two_batches['clusters'])
        with self.assertRaisesRegex(ValueError, 'batch count'):
            release._remote_routes({**routes, 'headCount': 1001}, larger, 12)

    def test_local_server_sets_brotli_only_for_existing_br_files(self):
        root = Path(__file__).resolve().parents[2]
        client = root / 'client'
        if not client.is_dir():
            client = root.parent / 'client'
        script = client / 'scripts/serve-local.py'
        spec = importlib.util.spec_from_file_location('local_release_server', script)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        for name, expected in [('centroids.lvq6.bin.br', True), ('centroids.lvq6.bin.gz', False), ('missing.br', False)]:
            handler = object.__new__(server.LocalHandler)
            handler.directory, handler.path = str(self.data), '/' + name
            headers = {}
            handler.send_header = lambda key, value: headers.__setitem__(key, value)
            with patch.object(server.SimpleHTTPRequestHandler, 'end_headers'):
                handler.end_headers()
            self.assertEqual(headers.get('Content-Encoding') == 'br', expected)
            self.assertEqual(headers['Cross-Origin-Embedder-Policy'], 'require-corp')


if __name__ == '__main__':
    unittest.main()
