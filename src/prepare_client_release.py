"""Bind a complete local dataset to the source client without publishing it."""

import argparse
import brotli
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from urllib.parse import unquote, urlsplit
import uuid


FORMATS = {'fp16': 'clusters', 'lvq6': 'clusters_lvq6', 'lvq8': 'clusters_lvq8'}
LIMIT = 25 * 1024 ** 2
DEFAULT_CLIENT = Path(__file__).resolve().parent / 'client'
if not DEFAULT_CLIENT.is_dir():
    DEFAULT_CLIENT = DEFAULT_CLIENT.parent.parent / 'client'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _remote_routes(routes, release, meta_batch_count):
    if not isinstance(routes, dict):
        raise ValueError('Routes must be a JSON object')
    for name in ('headCount', 'centroidSha256', 'centroidRawSha256'):
        if routes.get(name) != release[name]:
            raise ValueError(f'Remote routes identify a different dataset: {name}')
    if not isinstance(routes.get('clusters'), dict) or set(routes['clusters']) != set(FORMATS):
        raise ValueError('Routes must include every posting format')
    for name in ('centroids', 'formats', 'meta'):
        if name not in routes:
            raise ValueError(f'Missing remote route: {name}')
    entries = [('centroids', routes['centroids'], set(), None),
               ('formats', routes['formats'], set(), None),
               ('meta', routes['meta'], {'id', 'batch'}, meta_batch_count)]
    entries += [(name, value, {'id', 'batch'}, (release['headCount'] + 999) // 1000)
                for name, value in routes['clusters'].items()]
    for name, value, allowed, batch_count in entries:
        if isinstance(value, list) and (batch_count is None or len(value) != batch_count):
            raise ValueError(f'Invalid remote route batch count: {name}')
        for template in value if isinstance(value, list) else [value]:
            if not isinstance(template, str) or not template:
                raise ValueError(f'Invalid remote route: {name}')
            tokens = set(re.findall(r'\{([^{}]+)\}', template))
            if tokens - allowed or ('id' in allowed and 'id' not in tokens):
                raise ValueError(f'Invalid route placeholders: {name}')
            expanded = template.replace('{id}', '0').replace('{batch}', '0')
            if '{' in expanded or '}' in expanded:
                raise ValueError(f'Invalid route placeholders: {name}')
            parsed = urlsplit(expanded)
            if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or '\\' in expanded
                    or '..' in unquote(parsed.path).split('/') or parsed.port == 0):
                raise ValueError(f'Remote routes require HTTPS URLs without credentials or parent paths: {name}')
    local_only = routes.get('localOnly', True)
    if not isinstance(local_only, bool):
        raise ValueError('localOnly must be a boolean')
    return {**release, **{name: routes[name] for name in ('centroids', 'formats', 'clusters', 'meta')},
            'localOnly': local_only}


def prepare(data, meta='meta', client=DEFAULT_CLIENT, *, routes=None):
    """Validate all files first, update only dataset symlinks, then publish config last."""
    data, meta = Path(data).resolve(), Path(meta).resolve()
    original_client = Path(client)
    if original_client.is_symlink():
        raise ValueError('Client directory must not be a symlink')
    client = original_client.resolve()
    manifest_path = data / 'posting-codecs.json'
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get('formats'), dict):
        raise ValueError('Expected a posting manifest object with every format')
    heads = manifest.get('head_count')
    if (manifest.get('schema_version') != 1 or isinstance(heads, bool) or not isinstance(heads, int)
            or heads <= 0 or manifest.get('dimensions') != 384):
        raise ValueError('Expected a complete 384-dimensional posting manifest')
    gzip_path, br_path = data / 'centroids.lvq6.bin.gz', data / 'centroids.lvq6.bin.br'
    compressed = gzip_path.read_bytes()
    centroid_hash = _hash(compressed)
    if manifest.get('centroid_sha256') != centroid_hash:
        raise ValueError('Centroid gzip does not match the posting manifest')
    raw = gzip.decompress(compressed)
    if (len(raw) != 16 + 384 * 2 + heads * (4 + 288)
            or struct.unpack_from('<8sHHI', raw) != (b'LVQ6IDX\0', 2, 384, heads)):
        raise ValueError('Invalid LVQ6 L2 centroid header or payload size')
    for index, (value,) in enumerate(struct.iter_unpack('<e', raw[16:16 + 2 * (384 + 2 * heads)])):
        if not math.isfinite(value) or index >= 384 + heads and value <= 0:
            raise ValueError('Invalid LVQ6 centroid metadata')
    brotli_data = br_path.read_bytes()
    if not 0 < len(brotli_data) <= LIMIT:
        raise ValueError('Brotli centroid index exceeds the 25 MiB asset limit')
    if brotli.decompress(brotli_data) != raw:
        raise ValueError('Brotli centroid payload differs from the posting index')
    for format_name, directory in FORMATS.items():
        entry = manifest.get('formats', {}).get(format_name, {})
        if not isinstance(entry, dict):
            raise ValueError(f'Incomplete posting format: {format_name}')
        sizes = entry.get('gzip_bytes')
        if (entry.get('path') != directory or not isinstance(sizes, list) or len(sizes) != heads
                or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes)
                or entry.get('total_gzip_bytes', sum(sizes)) != sum(sizes)):
            raise ValueError(f'Incomplete posting format: {format_name}')
        folder = data / directory
        actual = set()
        for path in folder.iterdir():
            match = re.fullmatch(r'cluster_(0|[1-9][0-9]*)\.bin\.gz', path.name)
            if not match or path.is_symlink() or not path.is_file():
                raise ValueError(f'Unexpected posting entry: {path}')
            cid = int(match[1])
            if cid >= heads or path.stat().st_size != sizes[cid]:
                raise ValueError(f'Posting size or ID differs from the manifest: {path}')
            actual.add(cid)
        if len(actual) != heads:
            raise ValueError(f'Incomplete posting files: {format_name}')
    metadata_count, max_meta_id = 0, -1
    for path in meta.iterdir():
        match = re.fullmatch(r'meta_(0|[1-9][0-9]*)\.bin\.gz', path.name)
        if (not match
                or path.is_symlink() or not path.is_file() or path.stat().st_size <= 0):
            raise ValueError(f'Unexpected or empty metadata entry: {path}')
        metadata_count += 1
        max_meta_id = max(max_meta_id, int(match[1]))
    if not metadata_count:
        raise ValueError('Metadata directory is empty')
    release = {'headCount': heads, 'centroidSha256': centroid_hash, 'centroidRawSha256': _hash(raw),
               'centroids': 'data/centroids.lvq6.bin.br', 'formats': 'data/posting-codecs.json',
               'clusters': {name: f'data/{directory}/cluster_{{id}}.bin.gz' for name, directory in FORMATS.items()},
               'meta': 'data/meta/meta_{id}.bin.gz', 'localOnly': True}
    if routes is not None:
        release = _remote_routes(routes, release, max_meta_id // 250 + 1)
    data_link_dir = client / 'data'
    config_path = client / 'search-release.js'
    if data_link_dir.is_symlink() or data_link_dir.exists() and not data_link_dir.is_dir():
        raise ValueError('Client data directory must be a real directory')
    if config_path.is_symlink() or config_path.exists() and not config_path.is_file():
        raise ValueError('Client release config must be a regular file')
    targets = {'centroids.lvq6.bin.gz': gzip_path, 'centroids.lvq6.bin.br': br_path,
               'posting-codecs.json': manifest_path, 'meta': meta,
               **{directory: data / directory for directory in FORMATS.values()}}
    for name in targets:
        destination = data_link_dir / name
        if destination.exists() and not destination.is_symlink():
            raise ValueError(f'Refusing to overwrite a physical client data entry: {destination}')
    client.mkdir(parents=True, exist_ok=True)
    data_link_dir.mkdir(exist_ok=True)
    for name, target in targets.items():
        temporary = data_link_dir / f'.{name}.{uuid.uuid4().hex}.tmp'
        try:
            temporary.symlink_to(target, target_is_directory=target.is_dir())
            os.replace(temporary, data_link_dir / name)
        finally:
            temporary.unlink(missing_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=client, prefix='.search-release-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('// Generated after complete local dataset validation. Publishing the site is a separate step.\n')
            stream.write('globalThis.SEARCH_RELEASE = ' + json.dumps(release, indent=2) + ';\n')
        os.replace(temporary, config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return release


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--meta', default=Path('meta'), type=Path)
    parser.add_argument('--client', default=DEFAULT_CLIENT, type=Path)
    parser.add_argument('--routes', type=Path,
                        help='Explicit HTTPS routes for an uploaded release; no remote verification is performed')
    args = parser.parse_args()
    routes = json.loads(args.routes.read_text()) if args.routes else None
    print(json.dumps(prepare(args.data, args.meta, args.client, routes=routes), indent=2))
