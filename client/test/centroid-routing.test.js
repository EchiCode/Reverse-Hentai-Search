import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { gzipSync } from 'node:zlib';

const helper = readFileSync(new URL('../centroid-simd.js', import.meta.url), 'utf8');
const search = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
const wasm = readFileSync(new URL('../centroid-simd.wasm', import.meta.url));
const arrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

function payload() {
    const dimensions = 384, count = 3, lower = 16 + dimensions * 2;
    const delta = lower + count * 2, codes = delta + count * 2, rowBytes = dimensions * 6 / 8;
    const bytes = Buffer.alloc(codes + count * rowBytes);
    bytes.write('LVQ6IDX\0');
    bytes.writeUInt16LE(2, 8);
    bytes.writeUInt16LE(dimensions, 10);
    bytes.writeUInt32LE(count, 12);
    for (let row = 0; row < count; row++) {
        bytes.writeUInt16LE(0x3c00, delta + row * 2); // delta=1
        bytes[codes + row * rowBytes] = row + 1;
    }
    return bytes;
}

function client({ wasmEnabled = true, poolFactory, compile } = {}) {
    const requests = [], logs = [], bytes = payload();
    const wasmApi = Object.create(WebAssembly);
    wasmApi.compile = compile || WebAssembly.compile;
    const context = vm.createContext({
        AbortController, DecompressionStream, Response, TextDecoder, TransformStream,
        URL, performance, setTimeout, clearTimeout,
        WebAssembly: wasmEnabled ? wasmApi : undefined,
        location: { hostname: 'localhost', href: 'http://localhost/index.html' },
        console: { warn() {}, log(...args) { logs.push(args); }, debug() {} },
        async fetch(url) {
            requests.push(String(url));
            return new Response(String(url).includes('.wasm') ? wasm : gzipSync(bytes));
        }
    });
    vm.runInContext(`${helper}\n${search}\nglobalThis.api = {
        ensureCentroids, routeCentroids, rankCentroids, decodeCentroidsLVQ
    };`, context);
    if (poolFactory) context.createCentroidWorkerPool = index => poolFactory(index, context.api);
    const query = new Float32Array(384);
    query[0] = 2.1;
    return { api: context.api, requests, logs, query, bytes };
}

test('shared preload selects SIMD by default without starting fallback workers', async () => {
    let pools = 0;
    const { api, requests, logs, query, bytes } = client({ poolFactory() { pools++; throw new Error('Unexpected pool'); } });
    const [index, same] = await Promise.all([api.ensureCentroids(), api.ensureCentroids()]);
    assert.equal(index, same);
    assert.equal(index.simd.backend, 'simd');
    assert.equal(pools, 0);
    assert.equal(requests.filter(url => url.includes('centroids.lvq6')).length, 1);
    assert.equal(await api.ensureCentroids(), index);
    const readyLogs = logs.filter(([message]) => message.startsWith('[Index] Centroids loaded:'));
    assert.equal(readyLogs.length, 1);
    for (const field of ['totalMs', 'fetchMs', 'decodeAndInitMs']) {
        assert.ok(Number.isFinite(readyLogs[0][1][field]) && readyLogs[0][1][field] >= 0);
    }
    const reference = api.rankCentroids(query, api.decodeCentroidsLVQ(arrayBuffer(bytes)), 3);
    assert.equal(JSON.stringify(await api.routeCentroids(query, index, 3)), JSON.stringify(reference));
});

test('four-worker fallback recovers from transferred shards without redownloading the index', async () => {
    let fail = false, closes = 0, setups = 0;
    const { api, requests, query } = client({ wasmEnabled: false, poolFactory(index, api) {
        setups++;
        const shardReference = { ...index };
        return { backend: 'workers-4', transferred: true, close() { closes++; },
            async rank(query, probes, signal) {
                signal?.throwIfAborted();
                if (fail) throw new Error('Worker stopped');
                return api.rankCentroids(query, shardReference, probes);
            } };
    } });
    const index = await api.ensureCentroids();
    assert.equal(index.workerPool.backend, 'workers-4');
    assert.equal(index.vectors, null);
    assert.ok(index.recoveryBuffer);
    assert.equal((await api.routeCentroids(query, index, 1))[0].id, 1);
    const cancelled = new AbortController();
    cancelled.abort(new Error('cancel this query'));
    await assert.rejects(api.routeCentroids(query, index, 1, cancelled.signal), /cancel this query/);
    assert.equal(closes, 0);
    fail = true;
    assert.equal((await api.routeCentroids(query, index, 1))[0].id, 1);
    assert.equal(index.workerPool, null);
    assert.ok(index.vectors);
    assert.equal(index.recoveryBuffer, null);
    assert.equal(closes, 1);
    assert.equal(setups, 1);
    assert.equal(await api.ensureCentroids(), index);
    assert.equal(requests.length, 1);
});

test('a browser unable to start workers retains usable scalar routing', async () => {
    const { api, query } = client({ wasmEnabled: false, poolFactory() { throw new Error('Workers blocked'); } });
    const index = await api.ensureCentroids();
    assert.ok(index.vectors);
    assert.equal((await api.routeCentroids(query, index, 1))[0].id, 1);
});

test('cancelling one consumer does not cancel shared asynchronous SIMD initialization', async () => {
    let release;
    const ready = new Promise(resolve => { release = resolve; });
    const { api } = client({ compile: async bytes => { await ready; return WebAssembly.compile(bytes); } });
    const controller = new AbortController();
    const first = api.ensureCentroids(controller.signal), shared = api.ensureCentroids();
    controller.abort(new Error('cancel this consumer'));
    await assert.rejects(first, /cancel this consumer/);
    release();
    assert.equal((await shared).simd.backend, 'simd');
    assert.equal(await api.ensureCentroids(), await shared);
});
