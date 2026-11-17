import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { gzipSync } from 'node:zlib';

const source = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
const workerSource = readFileSync(new URL('../search-worker.js', import.meta.url), 'utf8');
const arrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const floatBytes = values => Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString('hex');

function client(fetch, overrides = {}) {
    const context = vm.createContext({
        AbortController, DecompressionStream, Response, TextDecoder, TransformStream,
        URL, performance, setTimeout, clearTimeout, fetch,
        location: { hostname: 'localhost' }, console: { log() {}, debug() {}, warn() {} },
        ...overrides
    });
    vm.runInContext(`${source}\nglobalThis.postingTests = {
        decodeSearchCluster, searchDataUrl, searchByClusters, retrieveSimilarImages, createSearchDiagnostics
    };`, context);
    return { api: context.postingTests, context };
}

// Literal outputs of balanced_l2.posting_quantization.encode_posting/decode_posting.
// Seven dimensions exercise six-bit padding and unaligned per-row headers.
const golden = [
    {
        bits: 6,
        payload: 'TFZRUE9TVAAHAAYBBAAAAB8dsSN8KLYQZqIAAAqvZQAAAAEApLZkJCTwY+ZxAMoAAAADAEi4NSSUD4TVnwAvAQAABQBmriYfT1I4CvADlAEAAAgAZq4mH09SOArwAw==',
        float32hex: '005f543e00cfccbe0054333f00503abb0094653e000897be0058cfbe00514ebe005dff3e000800bf0058fd3b00684fbe000b003f0096febe00901f3c0040b5bc0010f6bc00a0daba00c02ebd00c0ccbd005a6b3e00901f3c0040b5bc0010f6bc00a0daba00c02ebd00c0ccbd005a6b3e'
    },
    {
        bits: 8,
        payload: 'TFZRUE9TVAAHAAgBBAAAAB8dsSN8KLYQZqIAAAqvZQAAAAEApLZWHJAA/2KbGx7KAAAAAwBIuCgcUfsAhFb/JC8BAAAFAGauERc9JhQ6KgD/lAEAAAgAZq4RFz0mFDoqAP8=',
        float32hex: '007f4c3e00cfccbe8022333f00c0023a00466a3e00f799be00c6cbbe00394ebe004dff3e000800bf00609b3a005c4bbe00ecff3e0080ffbe0024283c00e49ebc0058f9bc00402d3a00e623bd00c0ccbdc07b6b3e0024283c00e49ebc0058f9bc00402d3a00e623bd00c0ccbdc07b6b3e'
    }
];

test('LVQ6/LVQ8 posting vectors reproduce the benchmark Float32 bytes and record identities', () => {
    const { api } = client();
    for (const fixture of golden) {
        const decoded = api.decodeSearchCluster(arrayBuffer(Buffer.from(fixture.payload, 'base64')), 7, `lvq${fixture.bits}`);
        assert.equal(decoded.count, 4);
        assert.equal(decoded.vectorLength, 7);
        assert.deepEqual(Array.from(decoded.codes), [101, 202, 303, 404]);
        assert.deepEqual(Array.from(decoded.pages), [1, 3, 5, 8]);
        assert.equal(floatBytes(decoded.vectors), fixture.float32hex);
        for (let row = 0; row < decoded.count; row++) {
            let squared = 0;
            for (const value of decoded.vectors.subarray(row * 7, (row + 1) * 7)) squared += value * value;
            assert.equal(decoded.inverseNorms[row], Math.fround(1 / Math.sqrt(squared)));
        }
    }
});

test('LVQ posting decoder rejects malformed payloads, nonfinite state, padding, and format mismatches', () => {
    const { api } = client(), original = Buffer.from(golden[0].payload, 'base64');
    const mutations = [
        bytes => bytes.write('BADPOST!'),
        bytes => bytes.writeUInt16LE(8, 8),
        bytes => bytes.writeUInt8(8, 10),
        bytes => bytes.writeUInt8(2, 11),
        bytes => bytes.writeUInt32LE(500, 12),
        bytes => bytes.writeUInt16LE(0x7c00, 16),
        bytes => bytes.writeUInt16LE(0x7c00, 36),
        bytes => bytes.writeUInt16LE(0, 38),
        bytes => bytes.writeUInt16LE(0x7e00, 38),
        bytes => bytes[45] |= 0x80
    ];
    for (const mutate of mutations) {
        const bytes = Buffer.from(original);
        mutate(bytes);
        assert.throws(() => api.decodeSearchCluster(arrayBuffer(bytes), 7, 'lvq6'), /LVQ/);
    }
    for (const bytes of [original.subarray(0, 15), original.subarray(0, -1), Buffer.concat([original, Buffer.alloc(1)])]) {
        assert.throws(() => api.decodeSearchCluster(arrayBuffer(bytes), 7, 'lvq6'), /LVQ/);
    }
    assert.throws(() => api.decodeSearchCluster(arrayBuffer(original), 7, 'lvq8'), /LVQ/);
    assert.throws(() => api.decodeSearchCluster(arrayBuffer(original), 7, 'fp32'), /quantization/);
});

test('LVQ mean restoration rounds only after addition, including odd-byte rows', () => {
    const { api } = client();
    for (const bits of [6, 8]) {
        const bytes = Buffer.alloc(29);
        bytes.write('LVQPOST\0');
        bytes.writeUInt16LE(1, 8);
        bytes.writeUInt8(bits, 10);
        bytes.writeUInt8(1, 11);
        bytes.writeUInt32LE(1, 12);
        bytes.writeUInt16LE(0xfbff, 16); // mean = -65504
        bytes.writeInt32LE(1, 18);
        bytes.writeInt16LE(1, 22);
        bytes.writeUInt16LE(0x7bff, 24); // lower = 65504
        bytes.writeUInt16LE(0x0400, 26); // delta = 2^-14
        bytes[28] = 1;
        const decoded = api.decodeSearchCluster(arrayBuffer(bytes), 1, `lvq${bits}`);
        assert.equal(decoded.vectors[0], 2 ** -14);
        assert.equal(decoded.inverseNorms[0], 2 ** 14);
    }
});

test('explicit full-precision decoding does not misidentify a record beginning with the LVQ magic', () => {
    const { api } = client(), bytes = Buffer.from('LVQPOST\0');
    const decoded = api.decodeSearchCluster(arrayBuffer(bytes), 1, 'fp16');
    assert.equal(decoded.count, 1);
    assert.equal(decoded.codes[0], bytes.readInt32LE(0));
    assert.equal(decoded.pages[0], bytes.readInt16LE(4));
});

test('quantized posting URLs are distinct by codec and preserve legacy full-precision and metadata paths', () => {
    for (const hostname of ['localhost', '127.0.0.1', '[::1]', 'example.test']) {
        const { api } = client(undefined, { location: { hostname } });
        assert.equal(api.searchDataUrl('clusters', 7999, 'lvq6'), 'data/clusters_lvq6/cluster_7999.bin.gz');
        assert.equal(api.searchDataUrl('clusters', 7999, 'lvq8'), 'data/clusters_lvq8/cluster_7999.bin.gz');
        assert.equal(api.searchDataUrl('clusters', 7999, 'fp16'), api.searchDataUrl('clusters', 7999));
        assert.equal(api.searchDataUrl('meta', 62500, 'lvq6'), api.searchDataUrl('meta', 62500));
        assert.throws(() => api.searchDataUrl('clusters', 0, 'lvq4'), /quantization/);
    }
});

function centroid() {
    const bytes = Buffer.alloc(16 + 768 + 4 + 288);
    bytes.write('LVQ6IDX\0');
    bytes.writeUInt16LE(2, 8);
    bytes.writeUInt16LE(384, 10);
    bytes.writeUInt32LE(1, 12);
    bytes.writeUInt16LE(0x3400, 16 + 768 + 2); // .25 step
    bytes[16 + 768 + 4] = 4; // Unit vector on coordinate zero.
    return bytes;
}

function posting(quantization, code) {
    if (quantization === 'fp16') {
        const bytes = Buffer.alloc(774);
        bytes.writeInt32LE(code);
        bytes.writeInt16LE(2, 4);
        bytes.writeUInt16LE(0x3c00, 6);
        return bytes;
    }
    const bits = Number(quantization.slice(3)), prefix = 16 + 768;
    const bytes = Buffer.alloc(prefix + 10 + 384 * bits / 8);
    bytes.write('LVQPOST\0');
    bytes.writeUInt16LE(384, 8);
    bytes.writeUInt8(bits, 10);
    bytes.writeUInt8(1, 11);
    bytes.writeUInt32LE(1, 12);
    bytes.writeInt32LE(code, prefix);
    bytes.writeInt16LE(2, prefix + 4);
    bytes.writeUInt16LE(0x3400, prefix + 8); // .25 step
    bytes[prefix + 10] = 4;
    return bytes;
}

test('fallback search uses the requested posting codec and keeps caches separate across format switches', async () => {
    const requests = [], ids = { fp16: 100, lvq6: 200, lvq8: 300 };
    const { api } = client(async url => {
        requests.push(url);
        const format = url.includes('clusters_lvq6') ? 'lvq6' : url.includes('clusters_lvq8') ? 'lvq8' : 'fp16';
        return new Response(gzipSync(url.includes('centroids') ? centroid() : posting(format, ids[format])));
    });
    const query = new Float32Array(384); query[0] = 1;
    for (const quantization of ['lvq6', 'lvq8', 'fp16', 'lvq6']) {
        const result = await api.retrieveSimilarImages(query, {
            signal: new AbortController().signal, threshold: .9, clusters: 1, quantization
        });
        assert.equal(result.totalCompared, 1);
        assert.equal(result.results[0].code, ids[quantization]);
        assert.equal(result.results[0].page, 2);
        assert.equal(result.results[0].similarity, 1);
    }
    assert.deepEqual(requests, ['data/centroids.lvq6.bin.gz',
        'data/clusters_lvq6/cluster_0.bin.gz', 'data/clusters_lvq8/cluster_0.bin.gz', 'data/clusters/cluster_0.bin.gz']);
    await assert.rejects(api.searchByClusters(query, {
        signal: new AbortController().signal, threshold: .9, clusters: 1, quantization: 'unknown'
    }), /quantization/);
});

test('retrieval worker requests preserve codec choices and use the current worker version', async () => {
    const sent = [], urls = [];
    class Worker {
        constructor(url) { urls.push(url.href); }
        postMessage(message) {
            sent.push(message);
            queueMicrotask(() => this.onmessage({ data: { id: message.id, result: { results: [], totalCompared: 0 } } }));
        }
    }
    const { api } = client(undefined, {
        Worker, document: { currentScript: { src: 'https://example.test/search.js' }, baseURI: 'https://example.test/' }
    });
    for (const quantization of ['lvq6', 'lvq8', 'fp16']) {
        await api.retrieveSimilarImages([1], { signal: new AbortController().signal, clusters: 8, threshold: .8, quantization });
    }
    assert.equal(urls[0], 'https://example.test/search-worker.js?v=20261117-echicode-data');
    assert.deepEqual(sent.map(message => message.payload.quantization), ['lvq6', 'lvq8', 'fp16']);
});

test('worker-side retrieval decodes LVQ6 and LVQ8 postings with their selected codecs', async () => {
    for (const quantization of ['lvq6', 'lvq8']) {
        const requests = [], responses = [];
        const { context } = client(async url => {
            requests.push(url);
            return new Response(gzipSync(url.includes('centroids') ? centroid() : posting(quantization, 123)));
        }, { self: { postMessage(message) { responses.push(message); } }, importScripts() {} });
        vm.runInContext(workerSource, context);
        const embedding = new Float32Array(384); embedding[0] = 1;
        await context.self.onmessage({ data: { id: 1, type: 'search', payload: {
            embedding, threshold: .9, clusters: 1, quantization
        } } });
        assert.equal(responses[0].error, undefined);
        assert.equal(responses[0].result.results[0].code, 123);
        assert.equal(requests[1], `data/clusters_${quantization}/cluster_0.bin.gz`);
    }
});
