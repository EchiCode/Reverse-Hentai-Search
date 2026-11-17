import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { open, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const clientDirectory = fileURLToPath(new URL('..', import.meta.url));
const dataset = process.argv[2] || path.join(clientDirectory, 'data');
const httpBase = /^https?:\/\//.test(dataset) ? new URL(dataset.endsWith('/') ? dataset : `${dataset}/`) : null;
const dataDirectory = httpBase ? null : path.resolve(dataset);
const sourceDirectory = existsSync(path.join(clientDirectory, '../src'))
    ? path.join(clientDirectory, '../src') : path.join(clientDirectory, '..');
const vectorsDirectory = path.resolve(process.argv[3] || path.join(sourceDirectory, 'vectors'));

const NUMERICAL_TIE_TOLERANCE = 1e-7;

function sourceTieMatch(results, source, clusterId, cluster, sourceSimilarity) {
    if (results.length !== 50 || cluster.vectorLength !== source.vectorLength || !Number.isFinite(sourceSimilarity)) return;
    const identicalVector = row => source.vectors.every((value, dimension) =>
        Object.is(value, cluster.vectors[row * cluster.vectorLength + dimension]));
    if (!cluster.codes.some((code, row) => code === source.codes[0]
        && cluster.pages[row] === source.pages[0] && identicalVector(row))) return;
    const candidates = results.filter(item => item.clusterId === clusterId
        && cluster.codes[item.index] === item.code && cluster.pages[item.index] === item.page
        && Math.abs(item.similarity - sourceSimilarity) <= NUMERICAL_TIE_TOLERANCE);
    const match = candidates.find(item => identicalVector(item.index)) || candidates[0];
    if (!match) return;
    const duplicateContentTie = identicalVector(match.index);
    return { match, duplicateContentTie, numericalScoreTie: !duplicateContentTie,
        sourceSimilarity, scoreGap: Math.abs(match.similarity - sourceSimilarity) };
}

// Check the cap, unchanged-source requirement, and numerical tolerance before scanning real data.
{
    const source = { codes: [51], pages: [1], vectorLength: 2, vectors: [1, 0] };
    const cluster = { codes: Array.from({ length: 51 }, (_, row) => row + 1),
        pages: Array(51).fill(1), vectorLength: 2, vectors: Array.from({ length: 102 }, (_, i) => i % 2 ? 0 : 1) };
    const results = cluster.codes.slice(0, 50).map((code, index) => ({ code, page: 1, index, clusterId: 0, similarity: 1 }));
    assert.equal(sourceTieMatch(results, source, 0, cluster, 1).duplicateContentTie, true);
    assert.equal(sourceTieMatch(results.slice(0, 49), source, 0, cluster, 1), undefined);
    assert.equal(sourceTieMatch(results, { ...source, codes: [52] }, 0, cluster, 1), undefined);
    cluster.vectors[100] = 0.999;
    assert.equal(sourceTieMatch(results, source, 0, cluster, 1), undefined);
    cluster.vectors[100] = 1;
    cluster.vectors.fill(0.999, 0, 100);
    assert.equal(sourceTieMatch(results, source, 0, cluster, 1).numericalScoreTie, true);
    assert.equal(sourceTieMatch(results, source, 0, cluster, 1 - 0.99e-7).numericalScoreTie, true);
    assert.equal(sourceTieMatch(results, source, 0, cluster, 1 - 1.01e-7), undefined);
    assert.equal(sourceTieMatch(results, source, 0, cluster, 0.95), undefined);
}

const requests = [];
const context = vm.createContext({
    AbortController, DecompressionStream, Response, TextDecoder, TransformStream,
    URL, performance, setTimeout, clearTimeout, location: { hostname: 'localhost' },
    console: { log() {}, debug() {}, warn() {} },
    async fetch(url, { signal } = {}) {
        signal?.throwIfAborted();
        assert.ok(url.startsWith('data/'), `Unexpected remote request: ${url}`);
        const file = httpBase ? new URL(url.slice(5), httpBase).href : path.join(dataDirectory, url.slice(5));
        requests.push(file);
        return httpBase ? globalThis.fetch(file, { signal }) : new Response(await readFile(file));
    },
});
vm.runInContext(`${await readFile(path.join(clientDirectory, 'search.js'), 'utf8')}
    globalThis.check = { ensureCentroids, decodeSearchCluster, l2Normalize,
        searchByClusters, searchDataUrl, fetchClusterMetaBinary, loadSearchCluster, scoreSearchCluster };`, context);
const api = context.check;
const index = await api.ensureCentroids();
const manifestBytes = httpBase ? await globalThis.fetch(new URL('posting-codecs.json', httpBase)).then(response => {
    assert.ok(response.ok, 'Posting manifest is unavailable');
    return response.text();
}) : await readFile(path.join(dataDirectory, 'posting-codecs.json'), 'utf8');
assert.equal(index.count, JSON.parse(manifestBytes).head_count);
assert.equal(index.metric, 'l2');
assert.equal(index.vectorLength, 384);

const files = (await readdir(vectorsDirectory)).filter(file => file.endsWith('.bin'))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
assert.ok(files.length, 'No source vector files');
const fileIndices = [...new Set(Array.from({ length: 5 }, (_, i) => Math.floor(i * (files.length - 1) / 4)))];
const cases = [];
for (const fileIndex of fileIndices) {
    const file = path.join(vectorsDirectory, files[fileIndex]);
    const source = await open(file);
    try {
        const size = (await source.stat()).size, recordBytes = 774;
        assert.equal(size % recordBytes, 0, `Truncated source: ${file}`);
        if (!size) continue;
        const count = size / recordBytes;
        const rows = [...new Set([0, Math.floor(count / 2), count - 1])];
        for (const row of rows) {
            const bytes = Buffer.alloc(recordBytes);
            const { bytesRead } = await source.read(bytes, 0, recordBytes, row * recordBytes);
            assert.equal(bytesRead, recordBytes);
            const decoded = api.decodeSearchCluster(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
            const code = decoded.codes[0], page = decoded.pages[0];
            const embedding = Array.from(api.l2Normalize(Array.from(decoded.vectors)));
            const started = performance.now();
            const result = await api.searchByClusters(embedding, {
                signal: new AbortController().signal, threshold: 0.999, clusters: 10,
            });
            let match = result.results.find(item => item.code === code && item.page === page);
            let tie;
            if (!match && result.results.length === 50) {
                const query = new Float32Array(api.l2Normalize(embedding));
                const sourceSimilarity = api.scoreSearchCluster(query, decoded, 0.999)[0]?.similarity;
                for (const clusterId of new Set(result.results.map(item => item.clusterId))) {
                    const cluster = await api.loadSearchCluster(api.searchDataUrl('clusters', clusterId), new AbortController().signal);
                    tie = sourceTieMatch(result.results, decoded, clusterId, cluster, sourceSimilarity);
                    if (tie) { match = tie.match; break; }
                }
            }
            assert.ok(match, `Source image ${code}:${page} was not recovered with 10 probes`);
            assert.ok(match.similarity > 0.9999);
            const metadata = await api.fetchClusterMetaBinary(api.searchDataUrl('meta', code));
            assert.ok(metadata.some(item => item.code === code), `No metadata for ${code}`);
            cases.push({ source: files[fileIndex], row, code, page, clusterId: match.clusterId,
                duplicateContentTie: tie?.duplicateContentTie ?? false, numericalScoreTie: tie?.numericalScoreTie ?? false,
                sourceSimilarity: tie?.sourceSimilarity, scoreGap: tie?.scoreGap,
                matchedCode: match.code, matchedPage: match.page,
                similarity: match.similarity, compared: result.totalCompared,
                elapsedMs: Math.round(performance.now() - started) });
        }
    } finally {
        await source.close();
    }
}
assert.ok(cases.length, 'No nonempty source records tested');
console.log(JSON.stringify({ centroidCount: index.count, metric: index.metric,
    dimensions: index.vectorLength, probes: 10, sourcesChecked: cases.length,
    uniqueFilesLoaded: new Set(requests).size, cases }, null, 2));
