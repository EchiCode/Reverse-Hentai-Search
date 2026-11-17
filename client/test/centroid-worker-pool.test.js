import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { Worker as ThreadWorker } from 'node:worker_threads';

const searchSource = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
const poolSource = readFileSync(new URL('../centroid-worker-pool.js', import.meta.url), 'utf8');
const workerSource = readFileSync(new URL('../centroid-rank-worker.js', import.meta.url), 'utf8');
const location = { href: 'https://example.test/client/search-worker.js?v=test', hostname: 'example.test' };
const globals = { URL, performance, Float32Array, Float64Array, SharedArrayBuffer, location };
const reference = vm.createContext(globals);
vm.runInContext(`${searchSource}\nglobalThis.referenceRank = rankCentroids;`, reference);
const plain = values => Array.from(values, ({ id, score }) => ({ id, score }));

function fixture(count = 37, vectorLength = 7, metric = 'l2') {
    let state = 42;
    const random = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296 - .5);
    const vectors = Float32Array.from({ length: count * vectorLength }, random);
    const bias = metric === 'l2' ? new Float64Array(count) : null;
    if (bias) for (let row = 0; row < count; row++) {
        let norm = 0;
        for (let dim = 0; dim < vectorLength; dim++) norm += vectors[row * vectorLength + dim] ** 2;
        bias[row] = -.5 * norm;
    }
    return { vectors, bias, count, vectorLength, metric };
}

function harness({ shared = true, failure } = {}) {
    const created = [], transfers = [], inits = [];
    class BrowserWorker {
        constructor(url) {
            const number = created.length;
            if (failure === 'constructor' && number === 2) throw new Error('Worker creation blocked');
            this.url = url;
            this.terminated = false;
            created.push(this);
            const bootstrap = `
                const { parentPort } = require('node:worker_threads');
                const vm = require('node:vm');
                globalThis.self = globalThis;
                globalThis.location = ${JSON.stringify(location)};
                globalThis.importScripts = () => vm.runInThisContext(${JSON.stringify(searchSource)}, {filename:'search.js'});
                self.postMessage = data => parentPort.postMessage(data);
                vm.runInThisContext(${JSON.stringify(workerSource)}, {filename:'centroid-rank-worker.js'});
                parentPort.on('message', data => {
                    if (${number === 0 && failure === 'init'} && data.type === 'init') throw new Error('Worker startup failed');
                    if (${number === 0 && failure === 'rank'} && data.type === 'rank') throw new Error('Worker ranking failed');
                    self.onmessage({data});
                });
            `;
            this.thread = new ThreadWorker(bootstrap, { eval: true });
            this.thread.on('message', data => this.onmessage?.({ data }));
            this.thread.on('error', error => this.onerror?.({ message: error.message, preventDefault() {} }));
        }
        postMessage(data, transfer = []) {
            transfers.push(transfer.reduce((bytes, buffer) => bytes + buffer.byteLength, 0));
            if (data.type === 'init') inits.push({
                buffer: data.index.vectors.buffer, rows: data.index.count, offset: data.offset
            });
            this.thread.postMessage(data, transfer);
        }
        terminate() { this.terminated = true; this.thread.terminate(); }
    }
    const context = vm.createContext({ ...globals, SharedArrayBuffer: shared ? SharedArrayBuffer : undefined, Worker: BrowserWorker });
    vm.runInContext(poolSource, context);
    return { create: context.createCentroidWorkerPool, created, transfers, inits };
}

test('four workers preserve exact Float64 scores, L2 query rounding, ties and global IDs', { timeout: 10000 }, async t => {
    for (const metric of ['l2', 'ip']) {
        const index = fixture(37, 7, metric), original = index.vectors;
        const api = harness(), loading = api.create(index);
        assert.equal(index.vectors, original, 'original remains available until every worker is ready');
        const pool = await loading;
        t.after(() => pool.close());
        assert.equal(pool.backend, 'workers-4');
        assert.equal(pool.transferred, false);
        assert.equal(api.created.length, 4);
        assert.ok(index.vectors.buffer instanceof SharedArrayBuffer);
        assert.ok(api.inits.every(init => init.buffer === index.vectors.buffer));
        assert.equal(api.transfers.reduce((a, b) => a + b, 0), 0);
        assert.ok(api.created.every(worker => worker.url.pathname.endsWith('/client/centroid-rank-worker.js')));
        const query = Float64Array.from({ length: 7 }, (_, i) => Math.sin(i * .7));
        for (const probes of [0, 1, 4, 16, 32, 64, 3.9]) {
            assert.deepEqual(plain(await pool.rank(query, probes)), plain(reference.referenceRank(query, index, probes)));
        }
    }
    const index = { vectors: new Float32Array(65540 * 3), bias: null, count: 65540, vectorLength: 3, metric: 'ip' };
    index.vectors[65539 * 3] = 1;
    const pool = await harness().create(index);
    t.after(() => pool.close());
    const result = await pool.rank([1, 0, 0], 16);
    assert.deepEqual(plain(result), plain(reference.referenceRank([1, 0, 0], index, 16)));
    assert.equal(result[0].id, 65539);
    assert.deepEqual(Array.from(result.slice(1), row => row.id), Array.from({ length: 15 }, (_, i) => i));
});

test('without shared memory, four disjoint transfers total one matrix and preserve startup fallback', { timeout: 10000 }, async t => {
    const index = fixture(), original = index.vectors, bias = index.bias;
    const bytes = original.byteLength + bias.byteLength;
    const api = harness({ shared: false }), loading = api.create(index);
    assert.equal(original.byteLength + bias.byteLength, bytes);
    const pool = await loading;
    t.after(() => pool.close());
    assert.equal(pool.transferred, true);
    assert.equal(index.vectors, original);
    assert.equal(index.bias, bias);
    assert.equal(original.byteLength + bias.byteLength, bytes, 'the caller can release originals after readiness');
    assert.equal(api.transfers.reduce((a, b) => a + b, 0), bytes);
    assert.equal(api.inits.reduce((rows, init) => rows + init.rows, 0), index.count);
    const query = Float64Array.from({ length: 7 }, (_, i) => Math.cos(i));
    assert.deepEqual(plain(await pool.rank(query, 32)), plain(reference.referenceRank(query, index, 32)));
});

test('concurrent queries remain independent and remove abort listeners after completion', { timeout: 10000 }, async t => {
    const index = fixture(), pool = await harness().create(index);
    t.after(() => pool.close());
    await Promise.all(Array.from({ length: 12 }, async (_, i) => {
        const controller = new AbortController();
        const query = Float64Array.from({ length: 7 }, (_, j) => Math.cos(i + j / 9));
        const result = await pool.rank(query, i + 1, controller.signal);
        assert.deepEqual(plain(result), plain(reference.referenceRank(query, index, i + 1)));
        assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
    }));
});

test('cancelling a query rejects immediately, ignores late replies and keeps the pool usable', { timeout: 10000 }, async t => {
    const index = fixture(), pool = await harness().create(index);
    t.after(() => pool.close());
    const query = new Float32Array(7).fill(.25), controller = new AbortController();
    const cancelled = pool.rank(query, 16, controller.signal);
    controller.abort();
    await assert.rejects(cancelled, error => error.name === 'AbortError');
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
    assert.deepEqual(plain(await pool.rank(query, 4)), plain(reference.referenceRank(query, index, 4)));
    await assert.rejects(pool.rank(query, 16, controller.signal), error => error.name === 'AbortError');
});

test('startup errors terminate all created workers and leave the original index intact', { timeout: 10000 }, async () => {
    for (const [failure, shared] of [['constructor', true], ['init', true], ['init', false]]) {
        const index = fixture(), original = index.vectors;
        const bytes = original.byteLength;
        const api = harness({ failure, shared });
        await assert.rejects(api.create(index), /blocked|startup failed/);
        assert.equal(index.vectors, original);
        assert.equal(index.vectors.byteLength, bytes);
        assert.ok(api.created.every(worker => worker.terminated));
    }
});

test('worker errors reject every pending query and prevent use of a failed pool', { timeout: 10000 }, async () => {
    const index = fixture(), api = harness({ failure: 'rank' }), pool = await api.create(index);
    const query = new Float32Array(7).fill(.25), controller = new AbortController();
    const results = await Promise.allSettled([
        pool.rank(query, 16, controller.signal), pool.rank(query, 4)
    ]);
    assert.ok(results.every(result => result.status === 'rejected' && /ranking failed/.test(result.reason.message)));
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
    assert.ok(api.created.every(worker => worker.terminated));
    await assert.rejects(pool.rank(query, 16), /ranking failed/);
});

test('closing a pool settles pending queries and query dimensions are checked', { timeout: 10000 }, async () => {
    const api = harness(), pool = await api.create(fixture());
    await assert.rejects(pool.rank([1], 4), /dimensions differ/);
    const controller = new AbortController(), pending = pool.rank(new Float32Array(7), 4, controller.signal);
    pool.close();
    pool.close();
    await assert.rejects(pending, /pool closed/);
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
    assert.ok(api.created.every(worker => worker.terminated));
});
