import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { brotliDecompressSync, gunzipSync } from 'node:zlib';

const clientDirectory = fileURLToPath(new URL('..', import.meta.url));
const dataset = process.argv[2] || path.join(clientDirectory, 'data');
const httpBase = /^https?:\/\//.test(dataset) ? new URL(dataset.endsWith('/') ? dataset : `${dataset}/`) : null;
const dataDirectory = httpBase ? null : path.resolve(dataset);
const requests = [];
async function loadFile(relative) {
    if (!httpBase) return readFile(path.join(dataDirectory, relative));
    const response = await fetch(new URL(relative, httpBase));
    assert.ok(response.ok, `HTTP ${response.status}: ${relative}`);
    return Buffer.from(await response.arrayBuffer());
}
const manifest = JSON.parse(await loadFile('posting-codecs.json'));
const compressedCentroids = await loadFile('centroids.lvq6.bin.gz');
const centroidHash = createHash('sha256').update(compressedCentroids).digest('hex');
assert.equal(manifest.centroid_sha256, centroidHash);
const release = manifest.head_count === 96000 ? {
    headCount: manifest.head_count, centroidSha256: centroidHash,
    centroidRawSha256: createHash('sha256').update(gunzipSync(compressedCentroids)).digest('hex'),
    centroids: 'data/centroids.lvq6.bin.br', formats: 'data/posting-codecs.json', localOnly: true,
    clusters: Object.fromEntries(['fp16', 'lvq6', 'lvq8'].map(format => [format,
        `data/${format === 'fp16' ? 'clusters' : `clusters_${format}`}/cluster_{id}.bin.gz`])),
    meta: 'data/meta/meta_{id}.bin.gz'
} : null;
const context = vm.createContext({
    AbortController, DecompressionStream, Response, TextDecoder, TransformStream,
    URL, performance, setTimeout, clearTimeout, location: { hostname: 'localhost' },
    crypto: webcrypto, SEARCH_RELEASE: release,
    console: { log() {}, debug() {}, warn() {} },
    async fetch(url, { signal } = {}) {
        signal?.throwIfAborted();
        assert.ok(url.startsWith('data/'), `Unexpected remote request: ${url}`);
        requests.push(url);
        let bytes = await loadFile(url.slice(5));
        if (!httpBase && url.endsWith('.br')) bytes = brotliDecompressSync(bytes);
        return new Response(bytes);
    }
});
vm.runInContext(`${await readFile(path.join(clientDirectory, 'search.js'), 'utf8')}
    globalThis.check = { ensureCentroids, decodeSearchCluster, rankCentroids, l2Normalize,
        retrieveSimilarImages, searchDataUrl, fetchClusterMetaBinary, createSearchDiagnostics };`, context);
const api = context.check;
const index = await api.ensureCentroids();
assert.equal(index.count, manifest.head_count);
assert.equal(index.vectorLength, 384);
assert.equal(index.metric, 'l2');

vm.runInContext(`${await readFile(path.join(clientDirectory, 'search-quality.js'), 'utf8')}
    globalThis.qualityLevels = getSearchQualityLevels;`, context);
const presets = context.qualityLevels(manifest.head_count);
const smallestProbeCount = Math.min(...presets.map(level => level.clusters));
for (const quantization of ['fp16', 'lvq6', 'lvq8']) {
    assert.equal(manifest.formats[quantization].gzip_bytes.length, index.count);
}
const cases = [];
for (const clusterId of [0, Math.floor(index.count / 2), index.count - 1]) {
    const raw = gunzipSync(await loadFile(`clusters/cluster_${clusterId}.bin.gz`));
    const decoded = api.decodeSearchCluster(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    // Choose a source whose assigned posting lies in the smallest preset's route.
    const row = [...new Set([0, Math.floor(decoded.count / 2), decoded.count - 1])].find(row => {
        const embedding = api.l2Normalize(Array.from(decoded.vectors.subarray(row * 384, (row + 1) * 384)));
        return api.rankCentroids(embedding, index, smallestProbeCount).some(centroid => centroid.id === clusterId);
    });
    if (row === undefined) continue;
    const code = decoded.codes[row], page = decoded.pages[row];
    const embedding = api.l2Normalize(Array.from(decoded.vectors.subarray(row * 384, (row + 1) * 384)));
    const outcomes = [];
    for (const { label, quantization, clusters } of presets) {
        const firstRequest = requests.length;
        const diagnostics = api.createSearchDiagnostics({ quantization, clusters });
        const result = await api.retrieveSimilarImages(embedding, {
            signal: new AbortController().signal, threshold: .9, clusters, quantization, diagnostics
        });
        const match = result.results.find(item => item.code === code && item.page === page);
        assert.ok(match, `${label} did not recover source ${code}:${page}`);
        const directory = quantization === 'fp16' ? 'clusters' : `clusters_${quantization}`;
        assert.ok(requests.slice(firstRequest).filter(url => url.includes('/cluster_'))
            .every(url => url.startsWith(`data/${directory}/`)), `Mixed codecs in ${label}`);
        outcomes.push({ label, quantization, clusters, similarity: match.similarity,
            compared: result.totalCompared,
            fetchedPostingMB: diagnostics.resources.filter(resource => resource.kind === 'cluster')
                .reduce((bytes, resource) => bytes + (resource.compressedBytes || 0), 0) / 1e6 });
    }
    const metadata = await api.fetchClusterMetaBinary(api.searchDataUrl('meta', code));
    assert.ok(metadata.some(item => item.code === code), `Metadata missing for ${code}`);
    cases.push({ clusterId, row, code, page, outcomes });
}
assert.ok(cases.length, `No source queries qualified for the ${smallestProbeCount}-probe route`);
console.log(JSON.stringify({ queriesChecked: cases.length, presetsChecked: presets.length,
    centroidCount: index.count, centroidSha256: centroidHash,
    note: 'Unmodified source-vector smoke test through main-thread retrieval; not a recall or browser-latency benchmark.',
    cases }, null, 2));
