import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import deploy_data as upload


class DataDeployTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.meta = self.root / 'data', self.root / 'meta'
        self.data.mkdir()
        self.meta.mkdir()
        manifest = {'schema_version': 1, 'dimensions': 384, 'head_count': 3,
                    'centroid_sha256': 'a' * 64, 'formats': {}}
        for name, directory in upload.FORMATS.items():
            (self.data / directory).mkdir()
            for cid in range(3):
                (self.data / directory / f'cluster_{cid}.bin.gz').write_bytes(b'posting')
            manifest['formats'][name] = {'path': directory, 'gzip_bytes': [7] * 3,
                                         'total_gzip_bytes': 21}
        (self.data / 'posting-codecs.json').write_text(json.dumps(manifest))
        index = b'compressed centroid fixture'
        (self.data / 'centroids.lvq6.bin.br').write_bytes(index)
        (self.data / 'stats.json').write_text(json.dumps({
            'centroid_brotli_bytes': len(index),
            'centroid_brotli_sha256': hashlib.sha256(index).hexdigest(),
            'centroid_brotli_raw_sha256': 'b' * 64}))
        for mid in (0, 2788):
            (self.meta / f'meta_{mid}.bin.gz').write_bytes(b'metadata')
        self.checkpoint = self.root / 'build' / 'data-deploy.json'
        self.routes = self.root / 'build' / 'uploaded-routes.json'
        for name, value in [('CHECKPOINT', self.checkpoint), ('ROUTES', self.routes)]:
            replacement = patch.object(upload, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def run_upload(self, **kwargs):
        return upload.deploy(self.data, self.meta, 'fixture-project', self.root / 'client', **kwargs)

    def fake_deploy(self, stage, project, branch):
        self.assertTrue((stage / '_headers').is_file())
        self.assertTrue((stage / '404.html').is_file())
        self.assertFalse((stage / 'functions').exists())
        inventory = json.loads((stage / upload.INVENTORY).read_text())
        self.assertLessEqual(len(inventory), 9)
        self.assertTrue(all(not path.startswith('clusters') for path in inventory))
        return f'https://{hashlib.sha256(branch.encode()).hexdigest()[:8]}.{project}.pages.dev'

    def test_dry_run_validates_without_staging_network_or_client_writes(self):
        with patch.object(upload, '_deploy') as deploy, patch.object(upload, '_verify') as verify, \
                patch.object(upload.prepare_client_release, 'prepare') as prepare:
            plan = self.run_upload(dry_run=True)
        self.assertEqual(plan['head_count'], 3)
        self.assertEqual(plan['posting_batches'], 1)
        self.assertEqual(plan['metadata_batches'], 12)
        self.assertEqual(plan['batches'], 14)
        self.assertEqual(plan['total_files'], 13)
        expected = sum(path.stat().st_size for folder in upload.FORMATS.values()
                       for path in (self.data / folder).iterdir())
        expected += sum(path.stat().st_size for path in self.meta.iterdir())
        expected += sum((self.data / name).stat().st_size
                        for name in ('posting-codecs.json', 'centroids.lvq6.bin.br'))
        self.assertEqual(plan['total_bytes'], expected)
        deploy.assert_not_called()
        verify.assert_not_called()
        prepare.assert_not_called()
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse(self.routes.exists())
        self.assertFalse((self.root / 'client').exists())
        self.assertEqual(list(self.data.glob('.cloudflare-stage-*')), [])

    def test_all_batches_verified_before_local_client_routes_change(self):
        calls = []

        def verify(url, inventory, workers):
            self.assertFalse(self.routes.exists())
            self.assertEqual(workers, 4)
            calls.append((url, inventory))

        def prepare(data, meta, client, *, routes):
            self.assertEqual(len(calls), 14)
            self.assertTrue(all(item['verified'] for item in json.loads(self.checkpoint.read_text())['batches'].values()))
            self.assertTrue(routes['localOnly'])

        with patch.object(upload, '_deploy', side_effect=self.fake_deploy) as deploy, \
                patch.object(upload, '_verify', side_effect=verify), \
                patch.object(upload.prepare_client_release, 'prepare', side_effect=prepare):
            routes = self.run_upload()
        self.assertEqual(deploy.call_count, 14)
        self.assertEqual(set(routes['clusters']), set(upload.FORMATS))
        for name, templates in routes['clusters'].items():
            self.assertEqual(len(templates), 1)
            self.assertTrue(templates[0].endswith(f'/{name}/cluster_{{id}}.bin.gz'))
        self.assertEqual(len(routes['meta']), 12)
        self.assertIn('meta_2788.bin.gz', routes['meta'][11].replace('{id}', '2788'))
        self.assertEqual(list(calls[-2][1]), ['meta_2788.bin.gz'])
        self.assertEqual(set(calls[-1][1]), {'centroids.lvq6.bin.br', 'posting-codecs.json'})
        self.assertTrue(routes['centroids'].endswith('.br'))
        self.assertTrue(json.loads(self.checkpoint.read_text())['complete'])
        self.assertEqual(list(self.data.glob('.cloudflare-stage-*')), [])

    def test_failed_verification_resumes_uploaded_batch_without_reuploading(self):
        with patch.object(upload, '_deploy', side_effect=self.fake_deploy) as first_upload, \
                patch.object(upload, '_verify', side_effect=ValueError('temporarily unavailable')), \
                patch.object(upload.prepare_client_release, 'prepare') as prepare:
            with self.assertRaisesRegex(ValueError, 'temporarily unavailable'):
                self.run_upload()
        first_upload.assert_called_once()
        prepare.assert_not_called()
        saved = json.loads(self.checkpoint.read_text())['batches']['p000']
        self.assertFalse(saved['verified'])
        self.assertFalse(self.routes.exists())
        self.assertFalse((self.root / 'client').exists())
        with patch.object(upload, '_deploy', side_effect=self.fake_deploy) as remaining, \
                patch.object(upload, '_verify') as verify, \
                patch.object(upload.prepare_client_release, 'prepare') as prepare:
            routes = self.run_upload()
        self.assertEqual(remaining.call_count, 13)
        self.assertEqual(verify.call_count, 14)
        self.assertTrue(routes['clusters']['lvq6'][0].startswith(saved['url'] + '/'))
        prepare.assert_called_once()
        with patch.object(upload, '_deploy') as already_uploaded, patch.object(upload, '_verify') as verified, \
                patch.object(upload.prepare_client_release, 'prepare'):
            self.run_upload()
        already_uploaded.assert_not_called()
        verified.assert_not_called()

    def test_checkpoint_cannot_be_reused_for_another_dataset_or_project(self):
        with patch.object(upload, '_deploy', side_effect=self.fake_deploy), \
                patch.object(upload, '_verify', side_effect=ValueError('offline')):
            with self.assertRaisesRegex(ValueError, 'offline'):
                self.run_upload()
        with patch.object(upload, '_deploy') as deploy:
            with self.assertRaisesRegex(ValueError, 'different dataset/project'):
                upload.deploy(self.data, self.meta, 'another-project')
            (self.meta / 'meta_2788.bin.gz').write_bytes(b'new metadata payload')
            with self.assertRaisesRegex(ValueError, 'different dataset/project'):
                self.run_upload()
        deploy.assert_not_called()

    def test_local_manifest_sizes_and_asset_limits_fail_before_upload(self):
        asset = self.data / 'clusters_lvq8' / 'cluster_2.bin.gz'
        asset.write_bytes(b'wrong length')
        with patch.object(upload, '_deploy') as deploy:
            with self.assertRaisesRegex(ValueError, 'Asset size differs'):
                self.run_upload()
        deploy.assert_not_called()
        asset.write_bytes(b'posting')
        with patch.object(upload, 'LIMIT', 6):
            with self.assertRaisesRegex(ValueError, 'exceeds 25 MiB'):
                self.run_upload(dry_run=True)

    def test_wrangler_uses_staging_cwd_and_only_immutable_deployment_url(self):
        stage = self.root / 'stage'
        stage.mkdir()
        result = subprocess.CompletedProcess([], 0,
            'https://r123456-p000.fixture-project.pages.dev\n'
            'https://1234abcd.fixture-project.pages.dev\n', '')
        with patch.object(upload.subprocess, 'run', return_value=result) as run:
            url = upload._deploy(stage, 'fixture-project', 'r123456-p000')
        self.assertEqual(url, 'https://1234abcd.fixture-project.pages.dev')
        self.assertEqual(run.call_args.kwargs['cwd'], stage)
        self.assertEqual(run.call_args.args[0][0:3], ['wrangler', 'pages', 'deploy'])
        self.assertIn(str(stage), run.call_args.args[0])
        self.assertTrue(run.call_args.kwargs['env']['WRANGLER_LOG_PATH'].endswith('cloudflare-wrangler.log'))


class FakeResponse:
    def __init__(self, headers, body=b'', status=200):
        self.status = status
        self.headers = {'Access-Control-Allow-Origin': '*', **headers}
        self.body = io.BytesIO(body)
        self.reads = []
        self.closed = False

    def getheader(self, name):
        return next((value for key, value in self.headers.items() if key.lower() == name.lower()), None)

    def read(self, size=-1):
        self.reads.append(size)
        return self.body.read(size)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    def request(self, method, path, headers):
        self.requests.append((method, path, headers))

    def getresponse(self):
        return next(self.responses)

    def close(self):
        self.closed = True


class DataResponseTest(unittest.TestCase):
    def setUp(self):
        self.payload = b'compressed posting fixture'
        self.entry = {'bytes': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()}
        self.path = '/fp16/cluster_3.bin.gz'

    def test_head_missing_length_requests_fallback_but_known_length_does_not(self):
        for headers, expected in [({}, False), ({'Content-Length': str(len(self.payload))}, True)]:
            with self.subTest(headers=headers):
                response = FakeResponse(headers)
                connection = FakeConnection([response])
                self.assertIs(upload._response(connection, 'HEAD', self.path, self.entry), expected)
                self.assertEqual(connection.requests, [('HEAD', self.path, {'Accept-Encoding': 'identity'})])
                self.assertTrue(response.closed)

    def test_wrong_explicit_head_length_still_fails(self):
        connection = FakeConnection([FakeResponse({'Content-Length': str(len(self.payload) + 1)})])
        with self.assertRaises(ValueError):
            upload._response(connection, 'HEAD', self.path, self.entry)

    def test_brotli_head_and_get_advertise_br_and_hash_encoded_body(self):
        path = '/centroids.lvq6.bin.br'
        headers = {'Content-Length': str(len(self.payload)), 'Content-Encoding': 'br'}
        connection = FakeConnection([FakeResponse(headers), FakeResponse(headers, self.payload)])
        self.assertTrue(upload._response(connection, 'HEAD', path, self.entry))
        self.assertEqual(upload._response(connection, 'GET', path, self.entry, collect=True), self.payload)
        self.assertEqual(connection.requests, [('HEAD', path, {'Accept-Encoding': 'br'}),
                                               ('GET', path, {'Accept-Encoding': 'br'})])

    def test_brotli_still_requires_encoding_and_matching_encoded_body_hash(self):
        path = '/centroids.lvq6.bin.br'
        for headers, body in [({'Content-Length': str(len(self.payload))}, self.payload),
                              ({'Content-Encoding': 'br'}, b'x' * len(self.payload))]:
            with self.subTest(headers=headers):
                connection = FakeConnection([FakeResponse(headers, body)])
                with patch.object(upload.time, 'sleep') as sleep:
                    with self.assertRaises(ValueError):
                        upload._response(connection, 'GET', path, self.entry)
                self.assertEqual(connection.requests, [('GET', path, {'Accept-Encoding': 'br'})])
                sleep.assert_not_called()

    def test_get_headers_only_validates_length_without_reading_body(self):
        response = FakeResponse({'Content-Length': str(len(self.payload))}, self.payload)
        connection = FakeConnection([response])
        upload._response(connection, 'GET', self.path, self.entry, headers_only=True)
        self.assertEqual(response.reads, [])
        self.assertTrue(response.closed)
        self.assertTrue(connection.closed)

    def test_get_headers_only_wrong_length_still_fails(self):
        response = FakeResponse({'Content-Length': str(len(self.payload) - 1)}, self.payload)
        with self.assertRaises(ValueError):
            upload._response(FakeConnection([response]), 'GET', self.path, self.entry, headers_only=True)
        self.assertEqual(response.reads, [])

    def test_get_missing_length_streams_and_hashes_body_even_in_headers_only_mode(self):
        for headers_only in (False, True):
            with self.subTest(headers_only=headers_only):
                response = FakeResponse({}, self.payload)
                upload._response(FakeConnection([response]), 'GET', self.path, self.entry,
                                 headers_only=headers_only)
                self.assertTrue(response.reads)
                self.assertTrue(response.closed)

    def test_get_missing_length_rejects_corrupt_short_or_oversize_body(self):
        for body in (b'x' * len(self.payload), self.payload[:-1], self.payload + b'x'):
            with self.subTest(body=body):
                response = FakeResponse({}, body)
                with self.assertRaises(ValueError):
                    upload._response(FakeConnection([response]), 'GET', self.path, self.entry,
                                     headers_only=True)

    def test_missing_length_fallback_keeps_cors_and_encoding_checks(self):
        for headers in ({'Access-Control-Allow-Origin': 'https://wrong.example'}, {'Content-Encoding': 'gzip'}):
            with self.subTest(headers=headers):
                with self.assertRaises(ValueError):
                    upload._response(FakeConnection([FakeResponse(headers, self.payload)]),
                                     'GET', self.path, self.entry, headers_only=True)

    def test_verify_gets_headers_only_for_missing_length_and_hashes_index_assets(self):
        inventory = {'fp16/cluster_3.bin.gz': self.entry, 'fp16/cluster_4.bin.gz': self.entry,
                     'centroids.lvq6.bin.br': self.entry, 'posting-codecs.json': self.entry}
        connection = FakeConnection()

        def response(connection, method, path, entry=None, collect=False, headers_only=False):
            if path == '/' + upload.INVENTORY:
                return json.dumps(inventory).encode()
            if method == 'HEAD':
                return path == '/fp16/cluster_4.bin.gz'

        with patch.object(upload.http.client, 'HTTPSConnection', return_value=connection), \
                patch.object(upload, '_response', side_effect=response) as request:
            upload._verify('https://1234abcd.fixture.pages.dev', inventory, 1)
        gets = {call.args[2]: call.kwargs for call in request.call_args_list
                if call.args[1] == 'GET' and call.args[2] != '/' + upload.INVENTORY}
        self.assertEqual(set(gets), {'/fp16/cluster_3.bin.gz', '/centroids.lvq6.bin.br', '/posting-codecs.json'})
        self.assertTrue(gets['/fp16/cluster_3.bin.gz'].get('headers_only'))
        self.assertFalse(gets['/centroids.lvq6.bin.br'].get('headers_only', False))
        self.assertFalse(gets['/posting-codecs.json'].get('headers_only', False))
        self.assertEqual(sum(call.args[1] == 'HEAD' for call in request.call_args_list), len(inventory))
        self.assertTrue(connection.closed)


class DataResponseRetryTest(unittest.TestCase):
    def setUp(self):
        self.payload = b'compressed posting fixture'
        self.entry = {'bytes': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()}
        self.path = '/fp16/cluster_3.bin.gz'

    def test_transient_disconnect_timeout_and_http_errors_retry_with_closed_connection(self):
        failures = [upload.http.client.RemoteDisconnected('closed'), TimeoutError('timed out'),
                    ConnectionResetError('reset'), upload.http.client.IncompleteRead(b'partial', 20),
                    upload.http.client.HTTPException('transient HTTP failure')]
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                connection = FakeConnection()
                with patch.object(upload, '_response_once', side_effect=[failure, True]) as request, \
                        patch.object(upload.time, 'sleep') as sleep, \
                        patch.object(connection, 'close', wraps=connection.close) as close, \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = upload._response(connection, 'HEAD', self.path, self.entry)
                self.assertIs(result, True)
                self.assertEqual(request.call_count, 2)
                close.assert_called_once()
                sleep.assert_called_once_with(1)

    def test_exhausted_retry_budget_stops_after_five_attempts_and_four_backoffs(self):
        connection = FakeConnection()
        with patch.object(upload, '_response_once', side_effect=TimeoutError('still unavailable')) as request, \
                patch.object(upload.time, 'sleep') as sleep, \
                patch.object(connection, 'close', wraps=connection.close) as close, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(TimeoutError, 'still unavailable'):
                upload._response(connection, 'HEAD', self.path, self.entry)
        self.assertEqual(request.call_count, 5)
        self.assertEqual(close.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4, 8])

    def test_temporary_http_status_retries_but_missing_asset_stops(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                connection = FakeConnection([
                    FakeResponse({'Access-Control-Allow-Origin': None}, status=status),
                    FakeResponse({'Content-Length': str(len(self.payload))})])
                with patch.object(upload.time, 'sleep') as sleep, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(upload._response(connection, 'HEAD', self.path, self.entry))
                self.assertEqual(len(connection.requests), 2)
                sleep.assert_called_once_with(1)
        connection = FakeConnection([FakeResponse({}, status=404)])
        with patch.object(upload.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, '404'):
                upload._response(connection, 'HEAD', self.path, self.entry)
        self.assertEqual(len(connection.requests), 1)
        sleep.assert_not_called()

    def test_validation_failure_is_not_retried(self):
        with patch.object(upload, '_response_once', side_effect=ValueError('Remote size differs')) as request, \
                patch.object(upload.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'Remote size differs'):
                upload._response(FakeConnection(), 'HEAD', self.path, self.entry)
        request.assert_called_once()
        sleep.assert_not_called()

    def test_partial_body_read_restarts_complete_request_and_hash_state(self):
        class InterruptedResponse(FakeResponse):
            def read(self, size=-1):
                if self.reads:
                    raise upload.http.client.IncompleteRead(b'', len(self.body.getvalue()) - 3)
                return super().read(min(size, 3))

        interrupted = InterruptedResponse({}, self.payload)
        complete = FakeResponse({}, self.payload)
        connection = FakeConnection([interrupted, complete])
        with patch.object(upload.time, 'sleep') as sleep, \
                patch.object(connection, 'close', wraps=connection.close) as close, \
                contextlib.redirect_stdout(io.StringIO()):
            result = upload._response(connection, 'GET', self.path, self.entry, collect=True)
        self.assertEqual(result, self.payload)
        self.assertEqual([method for method, _, _ in connection.requests], ['GET', 'GET'])
        self.assertTrue(interrupted.closed)
        self.assertTrue(complete.closed)
        close.assert_called_once()
        sleep.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()
