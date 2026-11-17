import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { gunzipSync } from 'node:zlib';

const [manifestUrl, originalPath] = process.argv.slice(2);
if (!manifestUrl || !originalPath) throw new Error('Usage: node check-centroid-parts.mjs MANIFEST_URL LOCAL_UNSPLIT_GZIP');
const context = vm.createContext({ fetch, URL, Response, AbortController, TextDecoder,
    performance, crypto: webcrypto, console });
vm.runInContext(await readFile(new URL('../search.js', import.meta.url), 'utf8')
    + '\nglobalThis.api = { fetchCentroidParts, decodeCentroidsLVQ, rankCentroids };', context);
const { api } = context;
const timing = {};
const parts = await api.fetchCentroidParts(manifestUrl, AbortSignal.timeout(120000), timing);
const downloaded = Buffer.from(parts.buffer);
const original = await readFile(originalPath);
const native = parts.native;
if (native) assert.equal(timing.sourceGzipSha256, createHash('sha256').update(original).digest('hex'),
    'Parts reference a different local unsplit source');
else assert(downloaded.equals(original), 'Reassembled gzip differs from the local unsplit file');
const arrayBuffer = buffer => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.length);
const localRaw = gunzipSync(original), remoteRaw = native ? downloaded : gunzipSync(downloaded);
assert(remoteRaw.equals(localRaw), 'Decompressed bytes differ');
const local = api.decodeCentroidsLVQ(arrayBuffer(localRaw));
const remote = api.decodeCentroidsLVQ(arrayBuffer(remoteRaw));
assert.equal(remote.count, local.count);
assert.equal(remote.metric, local.metric);
assert(Buffer.from(remote.vectors.buffer).equals(Buffer.from(local.vectors.buffer)), 'Decoded Float32 centroids differ');
if (local.bias) assert(Buffer.from(remote.bias.buffer).equals(Buffer.from(local.bias.buffer)), 'L2 biases differ');
const query = Float32Array.from({ length: 384 }, (_, i) => Math.sin(i + 1) * .05);
assert.equal(JSON.stringify(api.rankCentroids(query, remote, 128)), JSON.stringify(api.rankCentroids(query, local, 128)),
    'Top-128 IDs or scores differ');
console.log(JSON.stringify({ manifestUrl, originalPath, gzipBytes: original.length,
    centroids: remote.count, dimensions: remote.vectorLength, requests: timing.requestCount,
    compressedBytesIdentical: native ? null : true, sourceGzipSha256Matches: true, decompressedBytesIdentical: true,
    decodedFloat32Identical: true, top128IdsAndScoresIdentical: true }, null, 2));
