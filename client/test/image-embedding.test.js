import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = ['image-inference.js', 'search.js', 'search-quality.js', 'app.js'].map(file => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')).join('\n');
function client(output) {
    const context = vm.createContext({
        AbortController, performance,
        console: { log() {}, debug() {}, warn() {} },
        document: {
            addEventListener() {},
            createElement(tag) {
                assert.equal(tag, 'canvas');
                return { getContext(kind, options) {
                    assert.equal(kind, '2d');
                    assert.equal(options?.willReadFrequently, true);
                    return { drawImage() {} };
                } };
            },
        },
        Image: class {
            naturalWidth = 256;
            naturalHeight = 256;
            set src(value) { if (value) queueMicrotask(() => this.onload?.()); }
            async decode() {}
        },
        async model(canvas, options) {
            assert.equal(canvas.width, 256);
            assert.equal(canvas.height, 256);
            assert.equal(options.pool, false);
            return output;
        },
    });
    vm.runInContext(`${source}\ndinoModel = model;
        globalThis.tests = { getSearchEmbedding, rankCentroids, l2Normalize, createSearchDiagnostics };`, context);
    return context.tests;
}

test('image queries normalize CLS alone and route to the unit-vector L2 centroid', async () => {
    const raw = new Float32Array(197 * 384).fill(100);
    raw.fill(0, 0, 384);
    raw[0] = 3;
    raw[1] = 4;
    const api = client({ dims: [1, 197, 384], data: raw });
    const embedding = await api.getSearchEmbedding('image:test', new AbortController().signal, api.createSearchDiagnostics());
    assert.equal(embedding.length, 384);
    assert.equal(embedding[0], 0.6);
    assert.equal(embedding[1], 0.8);
    assert.ok(Math.abs(Math.hypot(...embedding) - 1) < 1e-12);
    assert.ok(embedding.slice(2).every(value => value === 0));

    const vectors = new Float32Array(2 * 384);
    vectors[0] = 0.6;
    vectors[1] = 0.8;
    vectors[384] = 0.06;
    vectors[385] = 0.08;
    const index = { vectors, count: 2, vectorLength: 384, metric: 'l2', bias: new Float64Array([-0.5, -0.005]) };
    const previousQuery = api.l2Normalize(Array.from(raw)).slice(0, 384);
    assert.equal(api.rankCentroids(previousQuery, index, 1)[0].id, 1);
    assert.equal(api.rankCentroids(embedding, index, 1)[0].id, 0);
});

test('image queries reject missing, short, wrong-dimension, non-finite and zero CLS outputs', async () => {
    const invalid = [
        {},
        { dims: [1, 384], data: new Float32Array(383).fill(1) },
        { dims: [1, 768], data: new Float32Array(768).fill(1) },
        { dims: [1, 384], data: new Float32Array(384).fill(NaN) },
        { dims: [1, 384], data: new Float32Array(384) },
    ];
    for (const output of invalid) {
        const api = client(output);
        await assert.rejects(api.getSearchEmbedding('image:test', new AbortController().signal,
            api.createSearchDiagnostics()), /valid embedding/);
    }
});
