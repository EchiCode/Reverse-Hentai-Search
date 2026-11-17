import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const helperSource = readFileSync(new URL('../centroid-simd.js', import.meta.url), 'utf8');
const searchSource = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
const wasm = readFileSync(new URL('../centroid-simd.wasm', import.meta.url));
const module = await WebAssembly.compile(wasm);
const buffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

function client(overrides = {}) {
    const requests = [];
    const context = vm.createContext({
        AbortController, URL, performance, setTimeout, clearTimeout, WebAssembly,
        document: { currentScript: { src: 'https://example.test/client/centroid-simd.js?v=test' } },
        location: { href: 'https://example.test/client/index.html' },
        console: { warn() {}, log() {}, debug() {} },
        async fetch(url) {
            requests.push(String(url));
            return { ok: true, arrayBuffer: async () => buffer(wasm) };
        },
        ...overrides
    });
    vm.runInContext('globalThis.self = globalThis;', context);
    vm.runInContext(`${helperSource}\n${searchSource}\nglobalThis.simdTests = {
        loadCentroidSimd, createCentroidSimd, decodeCentroidsLVQ, rankCentroids
    };`, context);
    return { api: context.simdTests, requests };
}

function sameRanking(actual, expected) {
    assert.equal(actual.length, expected.length);
    for (let i = 0; i < actual.length; i++) {
        assert.equal(actual[i].id, expected[i].id);
        assert.ok(Object.is(actual[i].score, expected[i].score), `Score differs at rank ${i}`);
    }
}

function packIntoBackend(backend, rows) {
    const dimensions = rows.length / backend.count, blockSize = backend.blockSize;
    for (let id = 0; id < backend.count; id++) {
        const block = (id - id % blockSize) * dimensions;
        for (let dimension = 0; dimension < dimensions; dimension++)
            backend.vectors[block + dimension * blockSize + id % blockSize] = rows[id * dimensions + dimension];
    }
}

test('SIMD compilation is shared and failed or absent WebAssembly stays a supported fallback', async () => {
    const supported = client();
    const [first, second] = await Promise.all([supported.api.loadCentroidSimd(), supported.api.loadCentroidSimd()]);
    assert.ok(first instanceof WebAssembly.Module);
    assert.equal(first, second);
    assert.equal(await supported.api.loadCentroidSimd(), first);
    assert.equal(supported.requests.length, 1);
    assert.match(supported.requests[0], /centroid-simd\.wasm/);

    let failures = 0;
    const unsupported = client({ fetch: async () => { failures++; throw new Error('blocked by CSP/network'); } });
    assert.equal(await unsupported.api.loadCentroidSimd(), null);
    assert.equal(await unsupported.api.loadCentroidSimd(), null);
    assert.equal(failures, 1);

    let compiles = 0;
    const deniedWasm = Object.create(WebAssembly);
    deniedWasm.compile = async () => { compiles++; throw new Error('SIMD disabled or Wasm blocked by CSP'); };
    const denied = client({ WebAssembly: deniedWasm });
    assert.equal(await denied.api.loadCentroidSimd(), null);
    assert.equal(await denied.api.loadCentroidSimd(), null);
    assert.equal(compiles, 1);
    assert.equal(denied.requests.length, 1);

    const unavailable = client({ WebAssembly: undefined });
    assert.equal(await unavailable.api.loadCentroidSimd(), null);
    assert.equal(unavailable.requests.length, 0);

    const incompatible = await WebAssembly.compile(Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0]));
    assert.equal(supported.api.createCentroidSimd(incompatible, 1), null);
});

test('SIMD dots and ordered routes exactly retain Float64 scalar arithmetic, ties and partial blocks', () => {
    const { api } = client();
    let seed = 42;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32 - .5);
    for (const count of [1, 3, 8, 17, 65]) {
        const dimensions = 384, rows = Float32Array.from({ length: count * dimensions }, random);
        rows.set([1, 0, 0, 0, 1e20, 1, -1e20, 1]);
        if (count >= 3) {
            rows.set(rows.subarray(dimensions * 2, dimensions * 3), dimensions);
            rows[dimensions * 2 + 383] += 2 ** -13;
        }
        const backend = api.createCentroidSimd(module, count, dimensions);
        assert.ok(backend);
        packIntoBackend(backend, rows);
        const queries = [Array(dimensions).fill(1),
            Array.from({ length: dimensions }, (_, i) => i % 2 ? -(1 + 2 ** -25) : 1 + 2 ** -25),
            Array.from({ length: dimensions }, random)];
        for (const metric of ['ip', 'l2']) {
            const bias = metric === 'l2' ? new Float64Array(count) : null;
            if (bias) for (let id = 0; id < count; id++) {
                let squared = 0;
                for (let i = 0; i < dimensions; i++) squared += rows[id * dimensions + i] ** 2;
                bias[id] = -.5 * squared;
            }
            const reference = { vectors: rows, bias, count, vectorLength: dimensions, metric };
            const accelerated = { ...reference, vectors: backend.vectors, blockSize: backend.blockSize, simd: backend };
            for (const query of queries) {
                const dots = backend.scores(query, metric), values = metric === 'l2' ? new Float32Array(query) : query;
                for (let id = 0; id < count; id++) {
                    let expected = 0;
                    for (let d = 0; d < dimensions; d++) expected += values[d] * rows[id * dimensions + d];
                    assert.ok(Object.is(dots[id], expected), `Raw dot differs: ${count}/${metric}/${id}`);
                }
                for (const probes of [1, 4, 16, 32, 128]) {
                    const expected = api.rankCentroids(query, reference, probes);
                    sameRanking(api.rankCentroids(query, accelerated, probes), expected);
                    sameRanking(api.rankCentroids(query, { ...accelerated, simd: null }, probes), expected);
                }
            }
        }
        assert.throws(() => backend.scores([1, 2], 'l2'), /dimensions|length/);
    }
    assert.equal(api.createCentroidSimd(module, 3, 7), null);
});

test('zero-filled SIMD padding cannot become a centroid and exact zero ties keep lower IDs', () => {
    const { api } = client(), count = 3, dimensions = 384;
    const backend = api.createCentroidSimd(module, count, dimensions);
    packIntoBackend(backend, new Float32Array(count * dimensions).fill(-1));
    const index = { vectors: backend.vectors, count, vectorLength: dimensions,
        blockSize: backend.blockSize, simd: backend, metric: 'ip', bias: null };
    const actual = api.rankCentroids(new Float32Array(dimensions).fill(1), index, 128);
    assert.deepEqual(Array.from(actual, item => item.id), [0, 1, 2]);
    backend.vectors.fill(-0);
    const zero = api.rankCentroids(new Float32Array(dimensions).fill(1), index, 128);
    assert.deepEqual(Array.from(zero, item => item.id), [0, 1, 2]);
    for (const item of zero) assert.ok(Object.is(item.score, 0));
});

function encodedIndex(bits, version) {
    const dimensions = 384, count = 17, rowBytes = dimensions * bits / 8;
    const lowerOffset = 16 + dimensions * 2, deltaOffset = lowerOffset + count * 2;
    const codesOffset = deltaOffset + count * 2, bytes = Buffer.alloc(codesOffset + rowBytes * count);
    bytes.write(`LVQ${bits}IDX\0`);
    bytes.writeUInt16LE(version, 8);
    bytes.writeUInt16LE(dimensions, 10);
    bytes.writeUInt32LE(count, 12);
    for (let d = 0; d < dimensions; d++) bytes.writeUInt16LE(0xfbff, 16 + 2 * d); // -65504 mean
    for (let id = 0; id < count; id++) {
        bytes.writeUInt16LE(0x7bff, lowerOffset + id * 2); // +65504 lower
        bytes.writeUInt16LE(0x0400, deltaOffset + id * 2); // 2^-14 delta
        for (let d = 0; d < dimensions; d++) {
            const code = (31 * d + 17 * id) % (1 << bits);
            for (let bit = 0; bit < bits; bit++) if (code & (1 << bit)) {
                const position = d * bits + bit;
                bytes[codesOffset + id * rowBytes + (position >>> 3)] |= 1 << (position & 7);
            }
        }
    }
    return bytes;
}

test('direct LVQ decode into SIMD memory preserves Float32 rounding and literal Float64 norms', () => {
    const { api } = client();
    for (const bits of [6, 8]) for (const version of [1, 2]) {
        const payload = encodedIndex(bits, version);
        const original = api.decodeCentroidsLVQ(buffer(payload), 384);
        const accelerated = api.decodeCentroidsLVQ(buffer(payload), 384, module);
        assert.ok(accelerated.simd);
        assert.equal(accelerated.vectors.buffer, accelerated.simd.vectors.buffer);
        const blockSize = accelerated.blockSize;
        for (let id = 0; id < original.count; id++) {
            const block = (id - id % blockSize) * 384;
            for (let d = 0; d < 384; d++) assert.ok(Object.is(
                accelerated.vectors[block + d * blockSize + id % blockSize], original.vectors[id * 384 + d]));
            if (original.bias) assert.ok(Object.is(accelerated.bias[id], original.bias[id]));
        }
        if (version === 2) assert.equal(original.vectors[1], 31 * 2 ** -14);
        for (const query of [new Float32Array(384).fill(1), Float64Array.from({ length: 384 }, (_, i) => i % 2 ? 2 ** -25 : 1)])
            for (const probes of [1, 4, 16, 32]) sameRanking(api.rankCentroids(query, accelerated, probes), api.rankCentroids(query, original, probes));
    }
});
