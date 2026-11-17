import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { gzipSync } from 'node:zlib';

const source = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
const url = 'https://example.test/release/centroid-parts.json';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

function fixture(count = 8, native = false) {
    const raw = Buffer.alloc(16 + 8 + 32 * 7);
    raw.write('LVQ6IDX\0');
    raw.writeUInt16LE(2, 8);
    raw.writeUInt16LE(4, 10);
    raw.writeUInt32LE(32, 12);
    for (let i = 0; i < 32; i++) {
        raw.writeUInt16LE(0xbc00, 24 + i * 2);
        raw.writeUInt16LE(0x3400, 88 + i * 2);
    }
    for (let i = 152; i < raw.length; i++) raw[i] = (i * 17 + (i >>> 3)) & 255;
    const compressed = gzipSync(raw), files = new Map();
    const stored = native ? raw : compressed;
    const parts = Array.from({ length: count }, (_, i) => {
        const bytes = stored.subarray(Math.floor(i * stored.length / count),
            Math.floor((i + 1) * stored.length / count));
        const file = `part-${i}`;
        files.set(new URL(file, url).href, bytes);
        return native ? { file, raw_bytes: bytes.length, compressed_bytes: bytes.length, sha256: hash(bytes) }
            : { file, bytes: bytes.length, sha256: hash(bytes) };
    });
    return { raw, files, manifest: native ? { schema_version: 2, compression: 'http', raw_bytes: raw.length,
        raw_sha256: hash(raw), source_gzip_sha256: hash(compressed), compressed_bytes: raw.length, parts }
        : { schema_version: 1, gzip_bytes: compressed.length, sha256: hash(compressed), parts } };
}

function client(fetch) {
    const context = vm.createContext({ fetch, URL, Response, DecompressionStream, TransformStream,
        AbortController, TextDecoder, performance, crypto: webcrypto,
        location: { href: 'https://example.test/', hostname: 'example.test' },
        console: { log() {}, warn() {}, debug() {} } });
    vm.runInContext(`${source}\nglobalThis.api = { fetchGzipBinary, decodeCentroidsLVQ, rankCentroids };`, context);
    return context.api;
}

test('parallel parts reassemble the local unsplit file and decode identical vectors', async () => {
    for (const [count, native] of [[2, false], [4, false], [8, false], [8, true]]) {
        const { raw, files, manifest } = fixture(count, native);
        let active = 0, peak = 0;
        const api = client(async request => {
            if (request === url) return Response.json(manifest);
            active++;
            peak = Math.max(peak, active);
            await new Promise(resolve => setTimeout(resolve, (count - active) * 2));
            active--;
            return new Response(files.get(request));
        });
        const timing = {};
        const actual = await api.fetchGzipBinary(url, undefined, timing);
        assert.deepEqual(Buffer.from(actual), raw);
        assert.equal(peak, count);
        assert.equal(timing.requestCount, count + 1);
        assert.equal(timing.gzipMode, native ? 'http-native' : 'streaming');
        const local = api.decodeCentroidsLVQ(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length), 4);
        const remote = api.decodeCentroidsLVQ(actual, 4);
        assert.deepEqual(Buffer.from(remote.vectors.buffer), Buffer.from(local.vectors.buffer));
        assert.deepEqual(Buffer.from(remote.bias.buffer), Buffer.from(local.bias.buffer));
        assert.equal(JSON.stringify(api.rankCentroids([1, 0, 0, 0], remote, 16)),
            JSON.stringify(api.rankCentroids([1, 0, 0, 0], local, 16)));
    }
});

test('native HTTP manifests reject invalid sizes, versions and source identity', async () => {
    for (const failure of ['schema', 'sourceHash', 'compressedTotal', 'partLimit', 'rawTotal', 'changed', 'reordered']) {
        const { files, manifest } = fixture(8, true);
        if (failure === 'schema') manifest.schema_version = 3;
        if (failure === 'sourceHash') manifest.source_gzip_sha256 = 'wrong';
        if (failure === 'compressedTotal') manifest.compressed_bytes++;
        if (failure === 'partLimit') manifest.parts[0].raw_bytes = 26 * 1024 * 1024;
        if (failure === 'rawTotal') manifest.raw_bytes++;
        if (failure === 'changed') files.get(new URL(manifest.parts[0].file, url).href)[0] ^= 1;
        if (failure === 'reordered') manifest.parts.reverse();
        const api = client(async request => request === url ? Response.json(manifest) : new Response(files.get(request)));
        await assert.rejects(api.fetchGzipBinary(url), /integrity check failed|Invalid centroid parts/);
    }
});

test('missing, changed and reordered parts fail before decoding', async () => {
    for (const failure of ['missing', 'changed', 'reordered', 'duplicate', 'path']) {
        const { files, manifest } = fixture();
        const first = new URL(manifest.parts[0].file, url).href;
        if (failure === 'missing') files.delete(first);
        if (failure === 'changed') files.get(first)[0] ^= 1;
        if (failure === 'reordered') manifest.parts.reverse();
        if (failure === 'duplicate') manifest.parts[1] = manifest.parts[0];
        if (failure === 'path') manifest.parts[0].file = '../part-0';
        const api = client(async request => request === url ? Response.json(manifest)
            : new Response(files.get(request), { status: files.has(request) ? 200 : 404 }));
        await assert.rejects(api.fetchGzipBinary(url), /HTTP 404|integrity check failed|Invalid centroid parts/);
    }
});

test('aborting a split download cancels every in-flight part', async () => {
    const { manifest } = fixture(), controller = new AbortController();
    let started = 0, cancelled = 0;
    const api = client(async (request, { signal }) => {
        if (request === url) return Response.json(manifest);
        return new Promise((resolve, reject) => {
            signal.addEventListener('abort', () => { cancelled++; reject(signal.reason); }, { once: true });
            if (++started === 8) controller.abort(new Error('cancel test'));
        });
    });
    await assert.rejects(api.fetchGzipBinary(url, controller.signal), /cancel test/);
    assert.equal(started, 8);
    assert.equal(cancelled, 8);
});
