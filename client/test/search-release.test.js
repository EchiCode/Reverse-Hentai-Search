import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
function fixture() {
    const raw = Buffer.alloc(16 + 384 * 2 + 3 * (4 + 288));
    raw.write('LVQ6IDX\0');
    raw.writeUInt16LE(2, 8);
    raw.writeUInt16LE(384, 10);
    raw.writeUInt32LE(3, 12);
    for (let i = 0; i < 3; i++) raw.writeUInt16LE(0x3c00, 16 + 384 * 2 + 3 * 2 + i * 2);
    return raw;
}
function client(hostname = 'localhost', localOnly = true, mutate = () => {}) {
    const raw = fixture();
    const release = { headCount: 3, centroidSha256: 'a'.repeat(64),
        centroidRawSha256: createHash('sha256').update(raw).digest('hex'), localOnly,
        centroids: 'https://new-data.test/centroids.lvq6.bin.br',
        formats: 'https://new-data.test/posting-codecs.json',
        clusters: Object.fromEntries(['fp16', 'lvq6', 'lvq8'].map(codec =>
            [codec, `https://release-b{batch}.test/${codec}/cluster_{id}.bin.gz`])),
        meta: 'https://release-meta-b{batch}.test/meta_{id}.bin.gz' };
    mutate(raw, release);
    const requests = [];
    const context = vm.createContext({ SEARCH_RELEASE: release, location: { hostname },
        crypto: webcrypto, URL, Response, AbortController, performance, TextDecoder,
        console: { log() {}, warn() {}, debug() {} },
        fetch: async url => {
            requests.push(url);
            assert.equal(url, release.centroids);
            // Fetch exposes the decoded body for HTTP Content-Encoding: br.
            return new Response(raw, { headers: { 'content-length': '100', 'content-encoding': 'br' } });
        } });
    vm.runInContext(source + '\nglobalThis.api = { searchDataUrl, ensureCentroids };', context);
    return { ...context.api, requests };
}

test('local testing uses the complete selected dataset and all posting formats', () => {
    const api = client();
    assert.equal(api.searchDataUrl('centroids'), 'https://new-data.test/centroids.lvq6.bin.br');
    assert.equal(api.searchDataUrl('formats'), 'https://new-data.test/posting-codecs.json');
    assert.equal(api.searchDataUrl('clusters', 95999, 'lvq8'), 'https://release-b95.test/lvq8/cluster_95999.bin.gz');
    assert.equal(api.searchDataUrl('meta', 697000), 'https://release-meta-b11.test/meta_2788.bin.gz');
});

test('immutable batch URLs select the correct posting and metadata deployment', () => {
    const api = client('localhost', true, (raw, release) => {
        release.headCount = 96000;
        release.clusters = Object.fromEntries(['fp16', 'lvq6', 'lvq8'].map(codec =>
            [codec, Array.from({ length: 96 }, (_, batch) =>
                `https://deploy-${codec}-${batch}.test/b{batch}/cluster_{id}.bin.gz`)]));
        release.meta = Array.from({ length: 12 }, (_, batch) =>
            `https://deploy-meta-${batch}.test/b{batch}/meta_{id}.bin.gz`);
    });
    for (const codec of ['fp16', 'lvq6', 'lvq8']) {
        for (const id of [0, 999, 1000, 95999]) {
            const batch = Math.floor(id / 1000);
            assert.equal(api.searchDataUrl('clusters', id, codec),
                `https://deploy-${codec}-${batch}.test/b${batch}/cluster_${id}.bin.gz`);
        }
    }
    for (const id of [0, 62499, 62500, 697000]) {
        const index = Math.floor(id / 250), batch = Math.floor(index / 250);
        assert.equal(api.searchDataUrl('meta', id),
            `https://deploy-meta-${batch}.test/b${batch}/meta_${index}.bin.gz`);
    }
});

test('local-only configuration preserves the old hosted dataset through upload staging', () => {
    const api = client('example.test');
    assert.equal(api.searchDataUrl('centroids'), 'centroids.lvq6.bin.gz');
    assert.equal(api.searchDataUrl('clusters', 4), 'https://batch-0.reverse-image-clusters-vers2.pages.dev/cluster_4.bin.gz');
    const hosted = client('example.test', false);
    assert.equal(hosted.searchDataUrl('clusters', 95999, 'lvq6'), 'https://release-b95.test/lvq6/cluster_95999.bin.gz');
});

test('one native Brotli request loads and caches the hash-pinned index', async () => {
    const api = client();
    const first = await api.ensureCentroids();
    assert.equal(first.count, 3);
    assert.equal(first.metric, 'l2');
    assert.equal(first, await api.ensureCentroids());
    assert.equal(api.requests.length, 1);
});

test('different centroid payloads or shapes cannot activate on the pinned release', async () => {
    const corrupt = client('localhost', true, raw => { raw[20] ^= 1; });
    await assert.rejects(corrupt.ensureCentroids(), /differs from the configured release/);
    const wrongCount = client('localhost', true, (raw, release) => { release.headCount = 4; });
    await assert.rejects(wrongCount.ensureCentroids(), /shape differs/);
});
