"""Upload dataset batches to Cloudflare Pages, then update only the local client."""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import http.client
import json
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

import prepare_client_release


FORMATS = prepare_client_release.FORMATS
LIMIT = prepare_client_release.LIMIT
CHECKPOINT = Path('build/data-deploy.json')
ROUTES = Path('build/uploaded-routes.json')
INVENTORY = 'upload-inventory.json'
HEADERS = '''/*
  Access-Control-Allow-Origin: *
  Cross-Origin-Resource-Policy: cross-origin
  Cache-Control: public, max-age=31536000, immutable, no-transform

/centroids.lvq6.bin.br
  Content-Encoding: br
  Content-Type: application/octet-stream
'''


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Refusing to replace a symlink: {path}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stat(path, size=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular asset: {path}')
    stat = path.stat()
    if not 0 < stat.st_size <= LIMIT or size is not None and stat.st_size != size:
        raise ValueError(f'Asset size differs or exceeds 25 MiB: {path}')
    return stat.st_size, stat.st_mtime_ns


def make_plan(data, meta):
    """Stat the complete dataset without hashing the large posting/metadata payloads."""
    data, meta = Path(data).resolve(), Path(meta).resolve()
    manifest_path = data / 'posting-codecs.json'
    manifest = json.loads(manifest_path.read_text())
    heads = manifest.get('head_count')
    if (manifest.get('schema_version') != 1 or manifest.get('dimensions') != 384
            or isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0):
        raise ValueError('Expected a complete 384-dimensional posting manifest')
    fingerprint = hashlib.sha256()
    total_bytes, total_files = 0, 0

    def record(name, size, mtime):
        nonlocal total_bytes, total_files
        fingerprint.update(f'{name}\0{size}\0{mtime}\n'.encode())
        total_bytes += size
        total_files += 1

    mtimes = {}
    for name, directory in FORMATS.items():
        entry = manifest.get('formats', {}).get(name, {})
        sizes = entry.get('gzip_bytes')
        if (entry.get('path') != directory or not isinstance(sizes, list) or len(sizes) != heads
                or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes)
                or entry.get('total_gzip_bytes', sum(sizes)) != sum(sizes)):
            raise ValueError(f'Incomplete posting format: {name}')
        folder = data / directory
        if sum(1 for _ in folder.iterdir()) != heads:
            raise ValueError(f'Unexpected posting file count: {folder}')
        mtimes[name] = array('Q')
        for cid, size in enumerate(sizes):
            path = folder / f'cluster_{cid}.bin.gz'
            actual_size, mtime = _stat(path, size)
            mtimes[name].append(mtime)
            record(f'{directory}/{path.name}', actual_size, mtime)

    metadata = []
    for path in meta.iterdir():
        match = re.fullmatch(r'meta_(0|[1-9][0-9]*)\.bin\.gz', path.name)
        if not match:
            raise ValueError(f'Unexpected metadata entry: {path}')
        size, mtime = _stat(path)
        metadata.append((int(match[1]), size, mtime))
    if not metadata:
        raise ValueError('Metadata directory is empty')
    metadata.sort()
    for mid, size, mtime in metadata:
        record(f'meta/meta_{mid}.bin.gz', size, mtime)

    index_files = []
    for name in ('centroids.lvq6.bin.br', 'posting-codecs.json'):
        size, mtime = _stat(data / name)
        sha = _hash_file(data / name)
        record(name, size, mtime)
        fingerprint.update(sha.encode())
        index_files.append((name, size, mtime, sha))
    stats = json.loads((data / 'stats.json').read_text())
    if (stats.get('centroid_brotli_bytes') != index_files[0][1]
            or stats.get('centroid_brotli_sha256') != index_files[0][3]):
        raise ValueError('Brotli centroid index differs from build statistics')
    identity = {'headCount': heads, 'centroidSha256': manifest.get('centroid_sha256'),
                'centroidRawSha256': stats.get('centroid_brotli_raw_sha256')}
    if any(not isinstance(identity[key], str) or not re.fullmatch('[0-9a-f]{64}', identity[key])
           for key in ('centroidSha256', 'centroidRawSha256')):
        raise ValueError('Missing centroid dataset hashes')
    fingerprint.update(json.dumps(identity, sort_keys=True).encode())
    posting_batches = (heads + 999) // 1000
    metadata_batches = metadata[-1][0] // 250 + 1
    release = fingerprint.hexdigest()
    summary = {'release': release, 'head_count': heads, 'posting_batches': posting_batches,
               'metadata_batches': metadata_batches, 'batches': posting_batches + metadata_batches + 1,
               'total_files': total_files, 'total_bytes': total_bytes,
               'index_bytes': index_files[0][1]}
    return {'data': data, 'meta': meta, 'manifest': manifest, 'mtimes': mtimes,
            'metadata': metadata, 'index_files': index_files, 'identity': identity, 'summary': summary}


def _batches(plan):
    for batch in range(plan['summary']['posting_batches']):
        yield f'p{batch:03d}'
    for batch in range(plan['summary']['metadata_batches']):
        yield f'm{batch:02d}'
    yield 'index'


def _files(plan, batch):
    """Yield only one batch's upload paths and its frozen local stat values."""
    if batch.startswith('p'):
        start = int(batch[1:]) * 1000
        stop = min(start + 1000, plan['summary']['head_count'])
        for name, directory in FORMATS.items():
            for cid in range(start, stop):
                filename = f'cluster_{cid}.bin.gz'
                yield (f'{name}/{filename}', plan['data'] / directory / filename,
                       plan['manifest']['formats'][name]['gzip_bytes'][cid], plan['mtimes'][name][cid])
    elif batch.startswith('m'):
        for mid, size, mtime in plan['metadata']:
            if mid // 250 == int(batch[1:]):
                filename = f'meta_{mid}.bin.gz'
                yield filename, plan['meta'] / filename, size, mtime
    else:
        for name, size, mtime, _ in plan['index_files']:
            yield name, plan['data'] / name, size, mtime


def _unchanged(files):
    for _, source, size, mtime in files:
        if _stat(source, size) != (size, mtime):
            raise ValueError(f'Asset changed during deployment: {source}')


def _stage(stage, files):
    inventory = {}
    _unchanged(files)
    for name, source, size, _ in files:
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, destination)
        inventory[name] = {'bytes': size, 'sha256': _hash_file(destination)}
    _unchanged(files)
    (stage / '_headers').write_text(HEADERS)
    (stage / '404.html').write_text('Dataset asset not found.\n')
    _write_json(stage / INVENTORY, inventory)
    return inventory


def _deploy(stage, project, branch):
    log_path = CHECKPOINT.resolve().parent / 'cloudflare-wrangler.log'
    environment = {**os.environ, 'CI': 'true', 'NO_COLOR': '1', 'WRANGLER_LOG_PATH': str(log_path)}
    # An isolated cwd also prevents Wrangler from discovering/building repository functions.
    result = subprocess.run(['wrangler', 'pages', 'deploy', str(stage), '--project-name', project,
                             '--branch', branch, '--install-skills=false'], cwd=stage,
                            env=environment, text=True, capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(f'Wrangler upload failed:\n{(result.stdout + result.stderr)[-4000:]}')
    urls = re.findall(r'https://[0-9a-f]{8,32}\.' + re.escape(project) + r'\.pages\.dev\b',
                      result.stdout + result.stderr)
    if not urls:
        raise RuntimeError('Wrangler did not return an immutable deployment URL; checkpoint was not advanced')
    return urls[-1]


def _response(connection, method, path, entry=None, collect=False, headers_only=False):
    for attempt in range(5):
        try:
            return _response_once(connection, method, path, entry, collect, headers_only)
        except (ConnectionError, TimeoutError, http.client.HTTPException) as error:
            connection.close()
            if attempt == 4:
                raise
            delay = 2 ** attempt
            print(f'{method} {path}: {error}; retry {attempt + 2}/5 in {delay}s', flush=True)
            time.sleep(delay)


def _response_once(connection, method, path, entry=None, collect=False, headers_only=False):
    connection.request(method, path, headers={'Accept-Encoding': 'br' if path.endswith('.br') else 'identity'})
    response = connection.getresponse()
    try:
        if response.status in (408, 429, 500, 502, 503, 504):
            raise http.client.HTTPException(f'Temporary HTTP {response.status}: {method} {path}')
        if response.status != 200 or response.getheader('Access-Control-Allow-Origin') != '*':
            raise ValueError(f'Remote asset unavailable or missing CORS: {method} {path} ({response.status})')
        content_length = response.getheader('Content-Length')
        if entry is not None:
            if content_length is not None and content_length != str(entry['bytes']):
                raise ValueError(f'Remote size differs: {path} (expected {entry["bytes"]}, got {content_length})')
            expected_encoding = 'br' if path.endswith('.br') else ''
            if (response.getheader('Content-Encoding') or '').lower() != expected_encoding:
                raise ValueError(f'Remote encoding differs: {path}')
        if method == 'HEAD':
            response.read()
            return content_length is not None
        if headers_only and content_length is not None:
            # Pages omits length on HEAD; inspect GET headers without draining every posting.
            connection.close()
            return None
        digest, chunks, length = hashlib.sha256(), [], 0
        limit = entry['bytes'] if entry is not None else 4 * 1024 * 1024
        while chunk := response.read(min(1024 * 1024, limit + 1 - length)):
            length += len(chunk)
            if length > limit:
                raise ValueError(f'Remote body too large: {path}')
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        if entry is not None and (length != entry['bytes'] or digest.hexdigest() != entry['sha256']):
            raise ValueError(f'Remote body hash differs: {path}')
        return b''.join(chunks) if collect else None
    finally:
        response.close()


def _verify(url, inventory, workers):
    hostname = urlsplit(url).hostname
    connections = [http.client.HTTPSConnection(hostname, timeout=45) for _ in range(workers)]
    available = Queue()
    for connection in connections:
        available.put(connection)
    executor = None
    try:
        remote = json.loads(_response(connections[0], 'GET', '/' + INVENTORY, collect=True))
        if remote != inventory:
            raise ValueError(f'Remote upload inventory differs: {url}')

        def check(item):
            name, entry = item
            connection = available.get()
            try:
                size_known = _response(connection, 'HEAD', '/' + name, entry)
                if name in ('centroids.lvq6.bin.br', 'posting-codecs.json'):
                    _response(connection, 'GET', '/' + name, entry)
                elif not size_known:
                    _response(connection, 'GET', '/' + name, entry, headers_only=True)
            finally:
                available.put(connection)

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(check, item) for item in inventory.items()]
        for future in as_completed(futures):
            future.result()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for connection in connections:
            connection.close()


def deploy(data, meta, project, client=prepare_client_release.DEFAULT_CLIENT, *, workers=4, dry_run=False):
    if not 1 <= workers <= 4 or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', project):
        raise ValueError('Expected a Pages project name and 1–4 verification workers')
    plan = make_plan(data, meta)
    print(json.dumps({**plan['summary'], 'project': project}, indent=2), flush=True)
    if dry_run:
        return plan['summary']
    binding = {'project': project, 'release': plan['summary']['release'],
               'data': str(plan['data']), 'meta': str(plan['meta'])}
    state = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {**binding, 'batches': {}}
    if any(state.get(key) != value for key, value in binding.items()):
        raise ValueError(f'Checkpoint identifies a different dataset/project: {CHECKPOINT}')
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    inventories = CHECKPOINT.parent / ('data-deploy-inventories-' + binding['release'][:12])
    batch_keys = list(_batches(plan))
    for number, batch in enumerate(batch_keys, 1):
        saved = state['batches'].get(batch)
        if saved and saved.get('verified'):
            print(f'[{number}/{len(batch_keys)}] {batch}: already verified', flush=True)
            continue
        files = list(_files(plan, batch))
        sidecar = inventories / f'{batch}.json'
        if saved:
            if _hash_file(sidecar) != saved['inventory_sha256']:
                raise ValueError(f'Local upload inventory changed: {sidecar}')
            inventory = json.loads(sidecar.read_text())
            _unchanged(files)
        else:
            print(f'[{number}/{len(batch_keys)}] {batch}: staging/uploading {len(files)} assets', flush=True)
            with tempfile.TemporaryDirectory(prefix='.cloudflare-stage-', dir=plan['data']) as temporary:
                stage = Path(temporary).resolve()
                inventory = _stage(stage, files)
                _write_json(sidecar, inventory)
                url = _deploy(stage, project, f'r{binding["release"][:12]}-{batch}')
                _unchanged(files)
            saved = {'url': url, 'inventory_sha256': _hash_file(sidecar), 'verified': False}
            state['batches'][batch] = saved
            _write_json(CHECKPOINT, state)
        print(f'[{number}/{len(batch_keys)}] {batch}: verifying {len(inventory)} assets', flush=True)
        _verify(saved['url'], inventory, workers)
        saved['verified'] = True
        _write_json(CHECKPOINT, state)

    # A source edit during a long upload must never switch the client to a mixed release.
    if make_plan(data, meta)['summary']['release'] != binding['release']:
        raise ValueError('Dataset changed during deployment; local client was not updated')
    urls = {key: state['batches'][key]['url'] for key in batch_keys}
    routes = {**plan['identity'], 'localOnly': True,
              'centroids': urls['index'] + '/centroids.lvq6.bin.br',
              'formats': urls['index'] + '/posting-codecs.json',
              'clusters': {name: [urls[f'p{batch:03d}'] + f'/{name}/cluster_{{id}}.bin.gz'
                                  for batch in range(plan['summary']['posting_batches'])] for name in FORMATS},
              'meta': [urls[f'm{batch:02d}'] + '/meta_{id}.bin.gz'
                       for batch in range(plan['summary']['metadata_batches'])]}
    _write_json(ROUTES, routes)
    prepare_client_release.prepare(data, meta, client, routes=routes)
    state['complete'] = True
    _write_json(CHECKPOINT, state)
    print(f'All batches verified. Local client updated; data routes: {ROUTES}. No frontend deployed.', flush=True)
    return routes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('clusters_balanced_l2_96k_final'))
    parser.add_argument('--meta', type=Path, default=Path('meta'))
    parser.add_argument('--project', default='reverse-image-data-96k')
    parser.add_argument('--client', type=Path, default=prepare_client_release.DEFAULT_CLIENT)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    deploy(args.data, args.meta, args.project, args.client, workers=args.workers, dry_run=args.dry_run)
