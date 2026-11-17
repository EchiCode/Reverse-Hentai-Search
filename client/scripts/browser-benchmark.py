import functools
import gzip
import http.server
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import urllib.request

import websocket

CLIENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLIENT_ROOT.parent / 'src'
if not PROJECT_ROOT.is_dir():
    PROJECT_ROOT = CLIENT_ROOT.parent
ROOT = Path(os.environ.get('RANK_CLIENT_ROOT', str(CLIENT_ROOT)))
OUT = PROJECT_ROOT / 'build/browser-benchmarks'
OUT.mkdir(parents=True, exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if os.environ.get('RANK_PRODUCTION') and path.split('?',1)[0] == '/data/centroids.lvq6.bin.gz':
            return str(ROOT / 'centroids.lvq6.bin.gz')
        if path.startswith('/__rank__/'):
            return str(Path(__file__).resolve().parent / path.split('?',1)[0].removeprefix('/__rank__/'))
        return super().translate_path(path)

    def do_GET(self):
        if os.environ.get('RANK_DISABLE_SIMD') and self.path.split('?',1)[0] == '/centroid-simd.wasm':
            self.send_error(404, 'SIMD disabled by CPU fallback test')
            return
        if os.environ.get('RANK_PRODUCTION') and self.path.startswith('/data/clusters/cluster_'):
            body=gzip.compress(b'')
            self.send_response(200)
            self.send_header('Content-Type','application/gzip')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
            return
        if self.path == '/__rank_bench__':
            body=b'<!doctype html><title>Centroid benchmark</title><script src="/centroid-simd.js"></script><script src="/centroid-worker-pool.js"></script><script src="/search.js"></script><body>Centroid benchmark</body>'
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):
        pass

    def end_headers(self):
        if not os.environ.get('RANK_DISABLE_SHARED'):
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
resources, logs, sessions, pending = {}, [], {}, {}
counter = 0
profile = tempfile.TemporaryDirectory(prefix='web-startup-chrome-', dir=os.environ.get('RANK_PROFILE_ROOT') or None)
chrome_log = (OUT / (os.environ.get('RANK_REPORT','startup') + '.chrome.log')).open('w')
proc = subprocess.Popen([os.environ.get('RANK_BROWSER_BINARY', '/usr/bin/google-chrome'), '--headless=new', '--no-sandbox',
    '--disable-gpu', '--disable-gpu-compositing', '--disable-gpu-rasterization',
    '--disable-features=WebGPU,Vulkan,WebNN',
    '--disable-dev-shm-usage', '--disable-background-networking', '--disable-extensions',
    '--disable-component-update', '--disable-sync', '--no-first-run', '--no-default-browser-check',
    '--remote-debugging-port=0', '--remote-allow-origins=*', '--user-data-dir=' + profile.name,
    'about:blank'], stdout=chrome_log, stderr=chrome_log)
ws = None


def send(method, params=None, session=None):
    global counter
    counter += 1
    payload = {'id': counter, 'method': method, 'params': params or {}}
    if session:
        payload['sessionId'] = session
    ws.send(json.dumps(payload))
    return counter


def receive(timeout=.2):
    ws.settimeout(timeout)
    try:
        msg = json.loads(ws.recv())
    except websocket.WebSocketTimeoutException:
        return
    if 'id' in msg:
        pending[msg['id']] = msg
        return
    method, p, session = msg.get('method'), msg.get('params', {}), msg.get('sessionId', '')
    if method == 'Target.attachedToTarget':
        child = p['sessionId']
        sessions[child] = p['targetInfo']
        send('Network.enable', {}, child)
        send('Runtime.enable', {}, child)
        # Dataset payloads are replaced with measured 64k/128k sizes after this common-startup audit.
        if not os.environ.get('RANK_PRODUCTION'):
            send('Network.setBlockedURLs', {'urls': ['*centroids.lvq6.bin.gz', '*posting-codecs.json']}, child)
        send('Target.setAutoAttach', {'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True}, child)
    elif method == 'Network.requestWillBeSent':
        key = session + ':' + p['requestId']
        old = resources.get(key)
        resources[key] = dict(url=p['request']['url'], method=p['request']['method'],
            type=p.get('type'), initiator=p.get('initiator', {}).get('type'),
            redirects=(old.get('redirects', []) + [old['url']] if old else []),
            encoded_body_bytes=0, decoded_body_bytes=0, session=session)
    elif method and method.startswith('Network.') and 'requestId' in p:
        key = session + ':' + p['requestId']
        r = resources.setdefault(key, {})
        if method == 'Network.responseReceived':
            response = p['response']
            r.update(url=response['url'],status=response['status'], mime=response['mimeType'], headers=response['headers'],
                disk_cache=response.get('fromDiskCache', False), service_worker=response.get('fromServiceWorker', False))
        elif method == 'Network.dataReceived':
            r['encoded_body_bytes'] = r.get('encoded_body_bytes', 0) + p.get('encodedDataLength', 0)
            r['decoded_body_bytes'] = r.get('decoded_body_bytes', 0) + p.get('dataLength', 0)
        elif method == 'Network.loadingFinished':
            r.update(finished=True, encoded_total_bytes=p['encodedDataLength'])
        elif method == 'Network.loadingFailed':
            r.update(failed=p.get('errorText'), blocked=p.get('blockedReason'))
    elif method == 'Runtime.consoleAPICalled':
        values = [a.get('value', a.get('description', '')) for a in p['args']]
        logs.append({'type': p['type'], 'values': values})
    elif method == 'Runtime.exceptionThrown':
        logs.append({'exception': p['exceptionDetails']})


def command(method, params=None, session=None, timeout=10):
    ident = send(method, params, session)
    deadline = time.monotonic() + timeout
    while ident not in pending and time.monotonic() < deadline:
        receive()
    return pending.pop(ident, {'error': 'timeout'})


try:
    active = Path(profile.name) / 'DevToolsActivePort'
    deadline = time.monotonic() + 20
    while not active.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    port, path = active.read_text().splitlines()[:2]
    ws = websocket.create_connection('ws://127.0.0.1:' + port + path, timeout=10, origin='http://localhost')
    command('Target.setAutoAttach', {'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True})
    deadline = time.monotonic() + 10
    while not any(t['type'] == 'page' for t in sessions.values()) and time.monotonic() < deadline:
        receive()
    page = next(s for s,t in sessions.items() if t['type'] == 'page')
    command('Page.enable', {}, page)
    candidate_script = os.environ.get('RANK_SCRIPT')
    url = f'http://127.0.0.1:{server.server_port}/' + ('__rank_bench__' if candidate_script and not os.environ.get('RANK_EXPERIMENT_APP') else '')
    command('Page.navigate', {'url': url}, page)
    started, last_status, ready_since = time.monotonic(), None, None
    next_check = started
    while time.monotonic() - started < 180:
        receive()
        if time.monotonic() >= next_check:
            if candidate_script:
                result = command('Runtime.evaluate', {'expression':'typeof rankCentroids', 'returnByValue':True},page)
                if result.get('result',{}).get('result',{}).get('value') == 'function':
                    break
                next_check=time.monotonic()+.2
                continue
            result = command('Runtime.evaluate', {'expression': "JSON.stringify({status:document.getElementById('status')?.textContent,modelReady:typeof dinoModel!=='undefined'&&!!dinoModel,searchReady:typeof searchSettingsReady!=='undefined'&&searchSettingsReady})", 'returnByValue': True}, page)
            status = result.get('result', {}).get('result', {}).get('value')
            if status != last_status:
                print('STATUS', round(time.monotonic()-started,1), status, flush=True)
                last_status = status
            if status:
                decoded = json.loads(status)
                if decoded.get('modelReady') and ready_since is None:
                    ready_since = time.monotonic()
                if ready_since and time.monotonic() - ready_since > 8:
                    break
                if 'Failed to load' in decoded.get('status', ''):
                    break
            next_check = time.monotonic() + 3
    baseline_expression = r'''(() => {
        function unrolled(query, index, probes) {
            const values = index.metric === 'l2' ? new Float32Array(query) : query;
            const best = new TopK(Math.min(index.count, Math.max(1, Math.floor(probes))));
            for (let id=0, offset=0; id<index.count; id++, offset+=index.vectorLength) {
                let score=0, i=0;
                for (;i+3<index.vectorLength;i+=4) {
                    score += values[i]*index.vectors[offset+i];
                    score += values[i+1]*index.vectors[offset+i+1];
                    score += values[i+2]*index.vectors[offset+i+2];
                    score += values[i+3]*index.vectors[offset+i+3];
                }
                for (;i<index.vectorLength;i++) score += values[i]*index.vectors[offset+i];
                if (index.bias) score += index.bias[id];
                if (best.canImprove(score)) best.add(score,{id,score});
            }
            return best.values();
        }
        let seed=42;
        const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296-.5)*.1;
        const vectors=new Float32Array(128000*384),bias=new Float64Array(128000);
        for(let id=0,offset=0;id<128000;id++) {
            let norm=0;
            for(let d=0;d<384;d++,offset++) { vectors[offset]=random();norm+=vectors[offset]*vectors[offset]; }
            bias[id]=-.5*norm;
        }
        const queries=Array.from({length:3},()=>Float32Array.from({length:384},random));
        const results=[];
        for(const count of [64000,80000,96000,128000]) {
            const index={vectors,bias,count,vectorLength:384,metric:'l2'};
            for(const query of queries) {
                const a=rankCentroids(query,index,16),b=unrolled(query,index,16);
                if(JSON.stringify(a)!==JSON.stringify(b))throw Error('Routing correctness mismatch');
            }
            const times={baseline:[],unrolled:[]};
            for(let repeat=0;repeat<8;repeat++) {
                for(const name of repeat%2?['unrolled','baseline']:['baseline','unrolled']) {
                    const start=performance.now();
                    const result=(name==='baseline'?rankCentroids:unrolled)(queries[repeat%3],index,16);
                    if(result.length!==16)throw Error('Wrong probe count');
                    times[name].push(performance.now()-start);
                }
            }
            for(const values of Object.values(times))values.sort((a,b)=>a-b);
            results.push({count,baseline_ms:times.baseline,unrolled_ms:times.unrolled});
        }
        return {note:'Synthetic Float32 centroids, L2 bias, actual rankCentroids code, 16 probes; desktop Chrome only. Not retrieval accuracy or end-to-end latency.',dimensions:384,queries:3,results};
    })()'''
    expression=Path(candidate_script).read_text() if candidate_script else baseline_expression
    benchmark = command('Runtime.evaluate', {'returnByValue': True,'awaitPromise':True,'expression':expression},page,timeout=180)
    if os.environ.get('RANK_PRODUCTION'):
        worker_states=[]
        for session,info in list(sessions.items()):
            if info['type']=='worker' and '/search-worker.js' in info.get('url',''):
                state=command('Runtime.evaluate', {'expression':"JSON.stringify(typeof centroids==='undefined'||!centroids?null:{count:centroids.count,backend:centroids.simd?'simd':centroids.workerPool?.backend||'scalar',transferred:centroids.workerPool?.transferred||false})",'returnByValue':True},session)
                worker_states.append({'url':info.get('url'), 'state':state})
        benchmark['workerStates']=worker_states
    name=os.environ.get('RANK_REPORT','routing_browser_benchmark')
    (OUT / f'{name}.json').write_text(json.dumps(benchmark, indent=2))
    print('ROUTING', json.dumps(benchmark), flush=True)
    report = dict(url=url, elapsed_seconds=time.monotonic()-started,
        model_status=last_status, resources=list(resources.values()), logs=logs,
        note='Fresh isolated Chrome profile. Dataset index/manifest blocked intentionally; substitute benchmark index sizes. Local source assets served without compression; remote transfer sizes are browser-measured.')
    (OUT / (name + '.trace.json' if candidate_script else 'cold_browser_trace.json')).write_text(json.dumps(report, indent=2))
    site_resources = [r for r in resources.values() if r.get('url','').startswith(('http://127.0.0.1:','https://'))]
    print('TOTAL', sum(max(0,r.get('encoded_total_bytes',0)) for r in site_resources)/1e6, 'MB', flush=True)
    for r in resources.values():
        resource_url = r.get('url','').split('?')[0]
        if resource_url.startswith(('http://127.0.0.1:','https://')):
            print(round(r.get('encoded_total_bytes',0)/1e6,6), r.get('status'), resource_url, r.get('failed',''), flush=True)
finally:
    if ws:
        ws.close()
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    server.shutdown()
    chrome_log.close()
    profile.cleanup()
