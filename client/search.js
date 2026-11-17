// Vector retrieval runs in search-worker.js when available; these functions also
// provide the fallback for browsers that cannot start a worker.
const searchScriptUrl = typeof document !== 'undefined' ? document.currentScript?.src : null;
let searchSequence = 0;

function getSearchRelease() {
    const release = globalThis.SEARCH_RELEASE;
    const local = typeof location !== 'undefined'
        && ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
    return release && (!release.localOnly || local) ? release : null;
}

function validatePostingQuantization(quantization) {
    if (!['fp16', 'lvq6', 'lvq8'].includes(quantization)) throw new Error('Unsupported posting quantization');
    return quantization;
}

function searchDataUrl(kind, id, quantization = 'fp16') {
    const release = getSearchRelease();
    if (release) {
        if (kind === 'centroids') return release.centroids;
        if (kind === 'formats') return release.formats;
        const index = kind === 'meta' ? Math.floor(id / 250) : id;
        const batch = Math.floor(index / (kind === 'meta' ? 250 : 1000));
        const routes = kind === 'meta' ? release.meta : release.clusters[validatePostingQuantization(quantization)];
        const template = Array.isArray(routes) ? routes[batch] : routes;
        return template.replaceAll('{id}', index).replaceAll('{batch}', batch);
    }
    const local = typeof location !== 'undefined'
        && ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
    if (kind === 'centroids') return local ? 'data/centroids.lvq6.bin.gz' : 'centroids.lvq6.bin.gz';
    if (kind === 'formats') return 'data/posting-codecs.json';
    const index = kind === 'meta' ? Math.floor(id / 250) : id;
    const file = `${kind === 'meta' ? 'meta' : 'cluster'}_${index}.bin.gz`;
    if (kind === 'clusters' && validatePostingQuantization(quantization) !== 'fp16') {
        return `data/clusters_${quantization}/${file}`;
    }
    if (local) return `data/${kind}/${file}`;
    const batch = Math.floor(index / (kind === 'meta' ? 250 : 1000));
    return `https://batch-${batch}.reverse-image-${kind}-vers2.pages.dev/${file}`;
}

function createSearchDiagnostics(settings = {}, id = ++searchSequence) {
    return { id, settings, startedAt: performance.now(), stages: {}, resources: [], clusters: [] };
}

function recordSearchStage(diagnostics, name, started, info = {}) {
    const durationMs = performance.now() - started;
    return recordSearchDuration(diagnostics, name, durationMs, info);
}

function recordSearchDuration(diagnostics, name, durationMs, info = {}) {
    if (diagnostics) {
        diagnostics.stages[name] = (diagnostics.stages[name] || 0) + durationMs;
        console.log(`[Search #${diagnostics.id}] ${name} ${durationMs.toFixed(2)}ms`, { durationMs, ...info });
    }
    return durationMs;
}

function mergeSearchDiagnostics(target, source) {
    if (!target || !source) return;
    for (const [name, duration] of Object.entries(source.stages)) {
        target.stages[name] = (target.stages[name] || 0) + duration;
    }
    target.resources.push(...source.resources);
    target.clusters.push(...source.clusters);
}

function finishSearchDiagnostics(diagnostics, outcome, info = {}) {
    const sum = key => diagnostics.resources.reduce((total, resource) => total + (resource[key] || 0), 0);
    const totalMs = performance.now() - diagnostics.startedAt;
    if (!globalThis.lastSearchTiming || diagnostics.id >= globalThis.lastSearchTiming.searchId) {
        globalThis.lastSearchTiming = { searchId: diagnostics.id, outcome, totalMs,
            stageMs: { ...diagnostics.stages }, inferenceRuntime: { ...diagnostics.inferenceRuntime } };
    }
    console.log(`[Search #${diagnostics.id}] ${outcome}`, {
        totalMs,
        ...info,
        settings: diagnostics.settings,
        inferenceRuntime: diagnostics.inferenceRuntime,
        stageMs: { ...diagnostics.stages },
        requests: diagnostics.resources.reduce((total, resource) => total
            + (resource.cacheHit ? 0 : resource.requestCount || 1), 0),
        cacheHits: diagnostics.resources.filter(resource => resource.cacheHit).length,
        compressedBodyBytes: sum('compressedBytes'),
        decodedBytes: sum('decodedBytes'),
        networkLoadMsSum: sum('loadMs'),
        gzipWallMsSum: sum('gzipWallMs'),
        gzipTailMsSum: sum('gzipTailMs'),
        binaryDecodeMsSum: sum('decodeMs'),
        scoringMsSum: diagnostics.clusters.reduce((total, cluster) => total + cluster.scoringMs, 0),
        timingNote: 'Concurrent requests overlap. modelPreprocessing, modelExecution and modelPipelineOther are components of modelInference. Streaming gzip wall time includes waiting for network; tail is time after the last compressed byte. These sums are not additive wall time. Body bytes may come from the browser HTTP cache.',
        resources: diagnostics.resources,
        clusters: diagnostics.clusters
    });
}

function l2Normalize(vec) {
    let squaredNorm = 0;
    for (let i = 0; i < vec.length; i++) squaredNorm += vec[i] * vec[i];
    const norm = Math.sqrt(squaredNorm);
    return vec.map(value => norm === 0 ? 0 : value / norm);
}

// A min-heap keeps only the best k candidates. Equal scores retain input order,
// matching a stable descending sort without sorting every qualifying vector.
class TopK {
    constructor(limit) { this.limit = limit; this.heap = []; this.sequence = 0; }
    canImprove(score) {
        return this.limit > 0 && Number.isFinite(score)
            && (this.heap.length < this.limit || score > this.heap[0].score);
    }
    worse(a, b) { return a.score < b.score || (a.score === b.score && a.order > b.order); }
    add(score, value) {
        if (!this.canImprove(score)) return;
        const item = { score, value, order: this.sequence++ };
        const heap = this.heap;
        if (heap.length < this.limit) {
            let i = heap.length;
            heap.push(item);
            while (i > 0) {
                const parent = (i - 1) >>> 1;
                if (!this.worse(item, heap[parent])) break;
                heap[i] = heap[parent];
                i = parent;
            }
            heap[i] = item;
        } else {
            let i = 0;
            while (2 * i + 1 < heap.length) {
                let child = 2 * i + 1;
                if (child + 1 < heap.length && this.worse(heap[child + 1], heap[child])) child++;
                if (!this.worse(heap[child], item)) break;
                heap[i] = heap[child];
                i = child;
            }
            heap[i] = item;
        }
    }
    values() {
        return this.heap.slice().sort((a, b) => b.score - a.score || a.order - b.order).map(item => item.value);
    }
}

function vectorDot(query, values, offset = 0) {
    let score = 0, i = 0;
    for (; i + 3 < query.length; i += 4) {
        score += query[i] * values[offset + i] + query[i + 1] * values[offset + i + 1]
            + query[i + 2] * values[offset + i + 2] + query[i + 3] * values[offset + i + 3];
    }
    for (; i < query.length; i++) score += query[i] * values[offset + i];
    return score;
}

let float16Lookup = null;
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
function getFloat16Lookup() {
    if (!float16Lookup) {
        float16Lookup = new Float32Array(65536);
        for (let bits = 0; bits < float16Lookup.length; bits++) {
            const sign = bits & 0x8000 ? -1 : 1;
            const exponent = (bits >>> 10) & 31;
            const fraction = bits & 1023;
            float16Lookup[bits] = sign * (exponent === 0 ? fraction * 2 ** -24
                : exponent === 31 ? (fraction ? NaN : Infinity)
                : (1 + fraction / 1024) * 2 ** (exponent - 15));
        }
    }
    return float16Lookup;
}

function decodeFloat16(buffer, byteOffset = 0, length = (buffer.byteLength - byteOffset) / 2) {
    const lookup = getFloat16Lookup();
    const values = new Float32Array(length);
    if (littleEndian && byteOffset % 2 === 0) {
        const bits = new Uint16Array(buffer, byteOffset, length);
        for (let i = 0; i < length; i++) values[i] = lookup[bits[i]];
    } else {
        const view = new DataView(buffer, byteOffset, length * 2);
        for (let i = 0; i < length; i++) values[i] = lookup[view.getUint16(i * 2, true)];
    }
    return values;
}

let gzipFallbackPromise = null;
function loadGzipFallback() {
    if (!gzipFallbackPromise) {
        gzipFallbackPromise = new Promise((resolve, reject) => {
            const url = 'https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js';
            if (typeof importScripts === 'function') {
                importScripts(url);
                resolve(globalThis.pako);
                return;
            }
            const script = document.createElement('script');
            script.src = url;
            script.onload = () => resolve(globalThis.pako);
            script.onerror = () => { script.remove(); reject(new Error('Could not load gzip decoder')); };
            document.head.appendChild(script);
        }).catch(error => { gzipFallbackPromise = null; throw error; });
    }
    return gzipFallbackPromise;
}

async function fetchCentroidParts(url, signal, timing = {}) {
    const started = performance.now();
    const response = await fetch(url, { signal });
    timing.headersMs = performance.now() - started;
    if (!response.ok) throw new Error(`Failed to fetch ${url} (HTTP ${response.status})`);
    const manifest = await response.json();
    const validHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
    const native = manifest?.schema_version === 2 && manifest.compression === 'http';
    const totalBytes = native ? manifest.raw_bytes : manifest?.gzip_bytes;
    const expectedHash = native ? manifest.raw_sha256 : manifest?.sha256;
    const partBytes = part => native ? part.raw_bytes : part.bytes;
    const validSize = bytes => Number.isSafeInteger(bytes) && bytes > 0 && bytes <= 25 * 1024 * 1024;
    if ((!native && manifest?.schema_version !== 1)
        || !Array.isArray(manifest.parts) || ![2, 4, 8].includes(manifest.parts.length)
        || !Number.isSafeInteger(totalBytes) || totalBytes < 1 || !validHash(expectedHash)
        || !manifest.parts.every(part => part && typeof part.file === 'string'
            && /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(part.file)
            && validSize(partBytes(part))
            && validHash(part.sha256))
        || new Set(manifest.parts.map(part => part.file)).size !== manifest.parts.length
        || manifest.parts.reduce((total, part) => total + partBytes(part), 0) !== totalBytes
        || (native && (!validHash(manifest.source_gzip_sha256)
            || !Number.isSafeInteger(manifest.compressed_bytes) || manifest.compressed_bytes < 1
            || !manifest.parts.every(part => validSize(part.compressed_bytes))
            || manifest.parts.reduce((total, part) => total + part.compressed_bytes, 0) !== manifest.compressed_bytes))) {
        throw new Error('Invalid centroid parts manifest');
    }
    const base = new URL(response.url || url, typeof location !== 'undefined' ? location.href : searchScriptUrl || undefined);
    const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
        value => value.toString(16).padStart(2, '0')).join('');
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    signal?.throwIfAborted();
    signal?.addEventListener('abort', abort, { once: true });
    timing.requestCount = 1 + manifest.parts.length;
    try {
        const parts = await Promise.all(manifest.parts.map(async part => {
            const partUrl = new URL(part.file, base).href;
            const fetched = await fetch(partUrl, { signal: controller.signal });
            if (!fetched.ok) throw new Error(`Failed to fetch ${partUrl} (HTTP ${fetched.status})`);
            const bytes = await fetched.arrayBuffer();
            if (bytes.byteLength !== partBytes(part) || await hash(bytes) !== part.sha256) {
                throw new Error(`Centroid part integrity check failed: ${part.file}`);
            }
            return bytes;
        }));
        timing.partsLoadMs = performance.now() - started;
        const combined = new Uint8Array(totalBytes);
        let offset = 0;
        for (const part of parts) {
            combined.set(new Uint8Array(part), offset);
            offset += part.byteLength;
        }
        if (await hash(combined) !== expectedHash) throw new Error('Centroid index integrity check failed');
        controller.signal.throwIfAborted();
        if (native) {
            timing.compressedBytes = manifest.compressed_bytes;
            timing.sourceGzipSha256 = manifest.source_gzip_sha256;
        }
        return { buffer: combined.buffer, native };
    } catch (error) {
        controller.abort(error);
        throw error;
    } finally {
        signal?.removeEventListener('abort', abort);
    }
}

async function fetchGzipBinary(url, signal, timing = {}) {
    const started = performance.now();
    const multipart = /\.json(?:[?#]|$)/.test(url);
    const parts = multipart ? await fetchCentroidParts(url, signal, timing) : null;
    if (parts?.native) {
        timing.loadMs = performance.now() - started;
        timing.status = 200;
        timing.gzipMode = 'http-native';
        timing.gzipWallMs = timing.gzipTailMs = 0;
        timing.decodedBytes = parts.buffer.byteLength;
        signal?.throwIfAborted();
        return parts.buffer;
    }
    const response = parts ? new Response(parts.buffer) : await fetch(url, { signal });
    timing.headersMs ??= performance.now() - started;
    timing.status = response.status;
    if (!response.ok) throw new Error(`Failed to fetch ${url} (HTTP ${response.status})`);
    if (/\.br(?:[?#]|$)/.test(url)) {
        const buffer = await response.arrayBuffer();
        timing.loadMs = performance.now() - started;
        timing.gzipMode = 'http-native';
        timing.gzipWallMs = timing.gzipTailMs = 0;
        timing.compressedBytes = Number(response.headers.get('content-length')) || 0;
        timing.decodedBytes = buffer.byteLength;
        signal?.throwIfAborted();
        return buffer;
    }
    if (typeof DecompressionStream !== 'undefined' && typeof TransformStream !== 'undefined' && response.body) {
        let downloaded;
        timing.compressedBytes = 0;
        const metered = response.body.pipeThrough(new TransformStream({
            transform(chunk, controller) {
                timing.compressedBytes += chunk.byteLength;
                controller.enqueue(chunk);
            },
            flush() {
                downloaded = performance.now();
                timing.loadMs = downloaded - started;
            }
        }));
        const gzipStarted = performance.now();
        timing.gzipMode = 'streaming';
        const buffer = await new Response(metered.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
        const finished = performance.now();
        timing.gzipWallMs = finished - gzipStarted;
        timing.gzipTailMs = downloaded === undefined ? 0 : finished - downloaded;
        timing.decodedBytes = buffer.byteLength;
        signal?.throwIfAborted();
        return buffer;
    }
    timing.gzipMode = 'buffered-fallback';
    const [compressed, decoder] = await Promise.all([
        response.arrayBuffer().then(buffer => {
            timing.loadMs = performance.now() - started;
            timing.compressedBytes = buffer.byteLength;
            return buffer;
        }),
        loadGzipFallback()
    ]);
    signal?.throwIfAborted();
    const gzipStarted = performance.now();
    const bytes = decoder.ungzip(new Uint8Array(compressed));
    timing.gzipWallMs = timing.gzipTailMs = performance.now() - gzipStarted;
    timing.decodedBytes = bytes.byteLength;
    return bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes.buffer : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function loadBinaryResource(url, signal, diagnostics, kind, parse) {
    const timing = { kind, url, cacheHit: false };
    diagnostics?.resources.push(timing);
    const started = performance.now();
    if (kind === 'centroids') console.log('[Index] Loading centroids', { url });
    try {
        const buffer = await fetchGzipBinary(url, signal, timing);
        signal?.throwIfAborted();
        const decodeStarted = performance.now();
        const result = await parse(buffer);
        timing.decodeMs = performance.now() - decodeStarted;
        return result;
    } catch (error) {
        timing.error = error.message;
        throw error;
    } finally {
        timing.totalMs = performance.now() - started;
        if (kind === 'centroids' && timing.decodeMs !== undefined) {
            const fetchMs = timing.loadMs || 0;
            console.log(`[Index] Centroids loaded: ${timing.totalMs.toFixed(2)} ms total `
                + `(${fetchMs.toFixed(2)} ms fetch, ${timing.decodeMs.toFixed(2)} ms decode/init)`,
                { url, totalMs: timing.totalMs, fetchMs, decodeAndInitMs: timing.decodeMs });
        }
        if (diagnostics) console.debug?.(`[Search #${diagnostics.id}] ${kind} resource`, { ...timing });
    }
}

function validateVectorBuffer(buffer, entrySize, vectorLength) {
    if (!Number.isInteger(vectorLength) || vectorLength < 1 || buffer.byteLength % entrySize !== 0) {
        throw new Error('Invalid vector file size');
    }
}

function decodeCentroidsLVQ(buffer, expectedVectorLength = 384, simdModule = null) {
    const headerBytes = 16;
    if (buffer.byteLength < headerBytes) throw new Error('Truncated LVQ centroid index');
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
    const bits = magic === 'LVQ6IDX\0' ? 6 : magic === 'LVQ8IDX\0' ? 8 : 0;
    if (!bits) throw new Error('Invalid LVQ centroid index');
    const version = view.getUint16(8, true);
    const vectorLength = view.getUint16(10, true);
    const count = view.getUint32(12, true);
    if ((version !== 1 && version !== 2) || vectorLength < 1 || vectorLength !== expectedVectorLength || count < 1) {
        throw new Error('Unsupported LVQ centroid index');
    }
    const meanOffset = headerBytes;
    const lowerOffset = meanOffset + 2 * vectorLength;
    const deltaOffset = lowerOffset + 2 * count;
    const codesOffset = deltaOffset + 2 * count;
    const rowBytes = Math.ceil(vectorLength * bits / 8);
    const expectedBytes = codesOffset + count * rowBytes;
    if (buffer.byteLength !== expectedBytes) throw new Error('Invalid LVQ centroid index size');

    const lookup = getFloat16Lookup();
    const codes = new Uint8Array(buffer, codesOffset, count * rowBytes);
    const simd = typeof createCentroidSimd === 'function' ? createCentroidSimd(simdModule, count, vectorLength) : null;
    const vectors = simd ? simd.vectors : new Float32Array(count * vectorLength);
    const blockSize = simd?.blockSize || 0;
    const halves = littleEndian ? new Uint16Array(buffer, 0, codesOffset / 2) : null;
    const metric = version === 2 ? 'l2' : 'ip';
    const bias = metric === 'l2' ? new Float64Array(count) : null;
    const mean = metric === 'l2' ? decodeFloat16(buffer, meanOffset, vectorLength) : null;
    if (mean && !mean.every(Number.isFinite)) throw new Error('Invalid LVQ centroid mean');
    for (let id = 0, output = 0; id < count; id++) {
        const lane = blockSize ? id % blockSize : 0;
        const blockOffset = blockSize ? (id - lane) * vectorLength : 0;
        const lower = lookup[halves ? halves[lowerOffset / 2 + id] : view.getUint16(lowerOffset + 2 * id, true)];
        const delta = lookup[halves ? halves[deltaOffset / 2 + id] : view.getUint16(deltaOffset + 2 * id, true)];
        if (!Number.isFinite(lower) || !Number.isFinite(delta) || delta <= 0) {
            throw new Error('Invalid LVQ centroid range');
        }
        let cursor = id * rowBytes, packed = 0, available = 0, squaredNorm = 0;
        for (let dimension = 0; dimension < vectorLength; dimension++, output++) {
            let code;
            if (bits === 8) code = codes[output];
            else {
                if (available < 6) {
                    packed |= codes[cursor++] << available;
                    available += 8;
                }
                code = packed & 63;
                packed >>>= 6;
                available -= 6;
            }
            // L2 needs the original mean and the norm of the rounded Float32 centroid.
            // Legacy IP can omit the common mean because it cannot change rank.
            const destination = blockSize ? blockOffset + dimension * blockSize + lane : output;
            vectors[destination] = lower + code * delta + (mean ? mean[dimension] : 0);
            if (bias) squaredNorm += vectors[destination] * vectors[destination];
        }
        if (packed) throw new Error('Invalid LVQ centroid row padding');
        if (bias) bias[id] = -.5 * squaredNorm;
    }
    return { vectors, count, vectorLength, metric, bias, simd, blockSize };
}

function rankCentroids(query, index, probes) {
    if (query.length !== index.vectorLength) throw new Error('Query and centroid dimensions differ');
    if (!index.simd && !index.blockSize) return rankScalarCentroids(query, index, probes);
    const best = new TopK(Math.min(index.count, Math.max(1, Math.floor(probes))));
    if (index.simd) {
        const scores = index.simd.scores(query, index.metric);
        for (let id = 0; id < index.count; id++) {
            const score = scores[id] + (index.bias ? index.bias[id] : 0);
            if (best.canImprove(score)) best.add(score, { id, score });
        }
        return best.values();
    }
    const values = index.metric === 'l2' ? new Float32Array(query) : query;
    if (index.blockSize) {
        for (let id = 0; id < index.count; id++) {
            const base = (id - id % index.blockSize) * index.vectorLength + id % index.blockSize;
            let score = 0;
            for (let i = 0; i < index.vectorLength; i++) score += values[i] * index.vectors[base + i * index.blockSize];
            if (index.bias) score += index.bias[id];
            if (best.canImprove(score)) best.add(score, { id, score });
        }
        return best.values();
    }
    return best.values();
}

function rankScalarCentroids(query, index, probes) {
    const values = index.metric === 'l2' ? new Float32Array(query) : query;
    const best = new TopK(Math.min(index.count, Math.max(1, Math.floor(probes))));
    for (let id = 0, offset = 0; id < index.count; id++, offset += index.vectorLength) {
        let score = 0;
        for (let i = 0; i < index.vectorLength; i++) score += values[i] * index.vectors[offset + i];
        if (index.bias) score += index.bias[id];
        if (best.canImprove(score)) best.add(score, { id, score });
    }
    return best.values();
}

async function fetchCentroidsLVQ(url, signal, vectorLength = 384, diagnostics) {
    const module = typeof loadCentroidSimd === 'function' ? loadCentroidSimd() : Promise.resolve(null);
    return loadBinaryResource(url, signal, diagnostics, 'centroids', async buffer => {
        const release = getSearchRelease();
        if (release) {
            const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)),
                value => value.toString(16).padStart(2, '0')).join('');
            if (hash !== release.centroidRawSha256) throw new Error('Centroid index differs from the configured release');
        }
        const index = decodeCentroidsLVQ(buffer, vectorLength, await module);
        if (release && (index.count !== release.headCount || index.metric !== 'l2')) {
            throw new Error('Centroid shape differs from the configured release');
        }
        if (!index.simd && typeof createCentroidWorkerPool === 'function') {
            try {
                index.workerPool = await createCentroidWorkerPool(index);
                if (index.workerPool.transferred) {
                    // Keep the packed source for recovery instead of a second decoded matrix.
                    index.recoveryBuffer = buffer;
                    index.vectors = index.bias = null;
                }
            } catch (error) {
                console.warn('Centroid workers unavailable; using scalar routing.', error);
            }
        }
        return index;
    });
}

async function routeCentroids(query, index, probes, signal) {
    signal?.throwIfAborted();
    if (query.length !== index.vectorLength) throw new Error('Query and centroid dimensions differ');
    if (index.workerPool) {
        try {
            return await index.workerPool.rank(query, probes, signal);
        } catch (error) {
            signal?.throwIfAborted();
            index.workerPool?.close();
            index.workerPool = null;
            if (!index.vectors) {
                Object.assign(index, decodeCentroidsLVQ(index.recoveryBuffer, index.vectorLength));
                index.recoveryBuffer = null;
            }
            console.warn('Centroid workers failed; using scalar routing.', error);
        }
    }
    return rankCentroids(query, index, probes);
}

// Pack each cluster into typed arrays and retain one inverse norm per vector.
// Applying that scalar after the dot product avoids a second 384-value decode
// pass while keeping cached scoring fast.
function decodeSearchCluster(buffer, vectorLength = 384, quantization = 'fp16') {
    validatePostingQuantization(quantization);
    if (quantization !== 'fp16') return decodeQuantizedSearchCluster(buffer, vectorLength, quantization);
    const entrySize = 6 + 2 * vectorLength;
    validateVectorBuffer(buffer, entrySize, vectorLength);
    const count = buffer.byteLength / entrySize;
    const codes = new Int32Array(count), pages = new Int16Array(count);
    const vectors = new Float32Array(count * vectorLength);
    const inverseNorms = new Float32Array(count);
    const view = new DataView(buffer);
    const bits = littleEndian ? new Uint16Array(buffer) : null;
    const lookup = getFloat16Lookup();
    for (let index = 0; index < count; index++) {
        const offset = index * entrySize, base = index * vectorLength;
        codes[index] = view.getInt32(offset, true);
        pages[index] = view.getInt16(offset + 4, true);
        let squaredNorm = 0;
        for (let i = 0; i < vectorLength; i++) {
            const value = lookup[bits ? bits[(offset + 6) / 2 + i] : view.getUint16(offset + 6 + i * 2, true)];
            vectors[base + i] = value;
            squaredNorm += value * value;
        }
        const norm = Math.sqrt(squaredNorm);
        // Invalid/zero vectors must never match, including negative thresholds.
        inverseNorms[index] = Number.isFinite(norm) && norm !== 0 ? 1 / norm : NaN;
    }
    return { codes, pages, vectors, inverseNorms, count, vectorLength,
        byteLength: codes.byteLength + pages.byteLength + vectors.byteLength + inverseNorms.byteLength };
}

function decodeQuantizedSearchCluster(buffer, vectorLength, quantization) {
    if (buffer.byteLength < 16) throw new Error('Truncated LVQ posting header');
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
    const dimensions = view.getUint16(8, true), bits = view.getUint8(10);
    const version = view.getUint8(11), count = view.getUint32(12, true);
    if (magic !== 'LVQPOST\0' || !Number.isInteger(vectorLength) || vectorLength < 1
        || dimensions !== vectorLength || version !== 1 || bits !== Number(quantization.slice(3))) {
        throw new Error('Unsupported LVQ posting header');
    }
    const rowBytes = Math.ceil(dimensions * bits / 8), entrySize = 10 + rowBytes;
    const prefix = 16 + 2 * dimensions;
    if (buffer.byteLength !== prefix + count * entrySize) throw new Error('Invalid LVQ posting size');
    const lookup = getFloat16Lookup(), mean = new Float64Array(dimensions);
    for (let dimension = 0; dimension < dimensions; dimension++) {
        mean[dimension] = lookup[view.getUint16(16 + 2 * dimension, true)];
        if (!Number.isFinite(mean[dimension])) throw new Error('Invalid LVQ posting mean');
    }
    const codes = new Int32Array(count), pages = new Int16Array(count);
    const vectors = new Float32Array(count * dimensions), inverseNorms = new Float32Array(count);
    for (let row = 0; row < count; row++) {
        const offset = prefix + row * entrySize, base = row * dimensions;
        codes[row] = view.getInt32(offset, true);
        pages[row] = view.getInt16(offset + 4, true);
        const lower = lookup[view.getUint16(offset + 6, true)];
        const delta = lookup[view.getUint16(offset + 8, true)];
        if (!Number.isFinite(lower) || !Number.isFinite(delta) || delta <= 0) {
            throw new Error('Invalid LVQ posting range');
        }
        let cursor = offset + 10, packed = 0, available = 0, squaredNorm = 0;
        for (let dimension = 0; dimension < dimensions; dimension++) {
            let code;
            if (bits === 8) code = view.getUint8(cursor++);
            else {
                if (available < 6) {
                    packed |= view.getUint8(cursor++) << available;
                    available += 8;
                }
                code = packed & 63;
                packed >>>= 6;
                available -= 6;
            }
            // Round the complete reconstruction once, exactly as the benchmark decoder does.
            vectors[base + dimension] = lower + code * delta + mean[dimension];
            const value = vectors[base + dimension];
            if (!Number.isFinite(value)) throw new Error('Invalid LVQ posting reconstruction');
            squaredNorm += value * value;
        }
        if (packed) throw new Error('Invalid LVQ posting row padding');
        const norm = Math.sqrt(squaredNorm);
        inverseNorms[row] = Number.isFinite(norm) && norm !== 0 ? 1 / norm : NaN;
    }
    return { codes, pages, vectors, inverseNorms, count, vectorLength,
        byteLength: codes.byteLength + pages.byteLength + vectors.byteLength + inverseNorms.byteLength };
}

class ByteLruCache {
    constructor(maxBytes) { this.maxBytes = maxBytes; this.bytes = 0; this.entries = new Map(); }
    get(key) {
        const entry = this.entries.get(key);
        if (!entry) return undefined;
        this.entries.delete(key);
        this.entries.set(key, entry);
        return entry.value;
    }
    set(key, value, bytes) {
        const previous = this.entries.get(key);
        if (previous) { this.bytes -= previous.bytes; this.entries.delete(key); }
        if (bytes > this.maxBytes) return;
        while (this.bytes + bytes > this.maxBytes && this.entries.size) {
            const oldest = this.entries.keys().next().value;
            this.bytes -= this.entries.get(oldest).bytes;
            this.entries.delete(oldest);
        }
        this.entries.set(key, { value, bytes });
        this.bytes += bytes;
    }
}

const clusterCache = new ByteLruCache(64 * 1024 * 1024);
const metadataCache = new ByteLruCache(8 * 1024 * 1024);
function recordCacheHit(diagnostics, kind, url) {
    diagnostics?.resources.push({ kind, url, cacheHit: true, totalMs: 0 });
}

async function loadSearchCluster(url, signal, diagnostics, quantization = 'fp16', vectorLength = 384) {
    signal.throwIfAborted();
    const cached = clusterCache.get(url);
    if (cached) { recordCacheHit(diagnostics, 'cluster', url); return cached; }
    const cluster = await loadBinaryResource(url, signal, diagnostics, 'cluster', buffer =>
        decodeSearchCluster(buffer, vectorLength, quantization));
    signal.throwIfAborted();
    clusterCache.set(url, cluster, cluster.byteLength);
    return cluster;
}

async function fetchClusterMetaBinary(url, signal, diagnostics) {
    signal?.throwIfAborted();
    const cached = metadataCache.get(url);
    if (cached) { recordCacheHit(diagnostics, 'metadata', url); return cached; }
    let bytes = 0;
    const result = await loadBinaryResource(url, signal, diagnostics, 'metadata', buffer => {
        // UTF-16 strings and object overhead are estimated conservatively.
        const records = parseClusterMetaBinary(buffer);
        bytes = buffer.byteLength * 2 + records.length * 256;
        return records;
    });
    signal?.throwIfAborted();
    metadataCache.set(url, result, bytes);
    return result;
}

// TLV metadata: reuse one decoder for all strings in the file.
function parseClusterMetaBinary(buffer) {
    const view = new DataView(buffer), decoder = new TextDecoder();
    let offset = 0;
    const result = [];
    function string(length) {
        const value = decoder.decode(new Uint8Array(buffer, offset, length));
        offset += length;
        return value;
    }
    function shortString() {
        const length = view.getUint16(offset, true); offset += 2;
        return string(length);
    }
    function strings() {
        const count = view.getUint8(offset++), values = [];
        for (let i = 0; i < count; i++) values.push(shortString());
        return values;
    }
    while (offset < buffer.byteLength) {
        const code = view.getInt32(offset, true); offset += 4;
        const name = shortString(), top_tags = strings(), top_chars = strings();
        const length = view.getUint32(offset, true); offset += 4;
        const thumbnail = string(length);
        result.push({ code, name, top_tags, top_chars, thumbnail });
    }
    return result;
}

let centroids = null;
let centroidLoad = null;
const MAX_CONCURRENT_CLUSTER_REQUESTS = 6;

function waitForCentroidLoad(load, signal) {
    signal?.throwIfAborted();
    if (!signal) return load;
    return new Promise((resolve, reject) => {
        let settled = false;
        const finish = (callback, value) => {
            if (settled) return;
            settled = true;
            signal.removeEventListener('abort', abort);
            callback(value);
        };
        const abort = () => finish(reject, signal.reason);
        signal.addEventListener('abort', abort, { once: true });
        load.then(value => finish(resolve, value), error => finish(reject, error));
    });
}

async function ensureCentroids(signal, diagnostics) {
    signal?.throwIfAborted();
    const url = searchDataUrl('centroids');
    if (centroids) { recordCacheHit(diagnostics, 'centroids', url); return centroids; }
    // The index is page-scoped rather than search-scoped. Image selection can
    // start it early, and later searches share the same transfer. Cancelling a
    // consumer stops its wait without aborting or poisoning the shared preload.
    if (!centroidLoad) {
        let load;
        load = fetchCentroidsLVQ(url, undefined, 384, diagnostics).then(
            loaded => {
                centroids = loaded;
                if (centroidLoad === load) centroidLoad = null;
                return loaded;
            },
            error => {
                if (centroidLoad === load) centroidLoad = null;
                throw error;
            }
        );
        centroidLoad = load;
    }
    return waitForCentroidLoad(centroidLoad, signal);
}

function scoreSearchCluster(query, cluster, threshold) {
    if (query.length !== cluster.vectorLength) throw new Error('Query and cluster dimensions differ');
    const best = new TopK(50), unique = new Map();
    for (let index = 0; index < cluster.count; index++) {
        const similarity = vectorDot(query, cluster.vectors, index * cluster.vectorLength) * cluster.inverseNorms[index];
        if (similarity > threshold && Number.isFinite(similarity)) {
            const code = cluster.codes[index], page = cluster.pages[index];
            const key = `${code}:${page}`, previous = unique.get(key);
            if (!previous || similarity > previous.similarity) {
                // A better duplicate takes its own position in equal-score input order.
                unique.delete(key);
                unique.set(key, { index, similarity, code, page });
            }
        }
    }
    for (const match of unique.values()) best.add(match.similarity, match);
    return best.values();
}

function mergeClusterResults(clusterResults) {
    const best = new TopK(50), unique = new Map();
    // Always merge by centroid rank, including for partial snapshots. Network
    // completion order must not change the stable ordering of equal scores.
    for (const matches of clusterResults) {
        if (matches) for (const match of matches) {
            const key = `${match.code}:${match.page}`, previous = unique.get(key);
            if (!previous || match.similarity > previous.similarity) unique.set(key, match);
        }
    }
    for (const match of unique.values()) best.add(match.similarity, match);
    return best.values();
}

async function searchByClusters(queryEmbedding, { signal, threshold, clusters, quantization = 'fp16', diagnostics, onProgress }) {
    validatePostingQuantization(quantization);
    const started = performance.now();
    const index = await ensureCentroids(signal, diagnostics);
    const rankingStarted = performance.now();
    const topCentroids = await routeCentroids(queryEmbedding, index, clusters, signal);
    signal.throwIfAborted();
    recordSearchStage(diagnostics, 'centroidRanking', rankingStarted, { candidates: index.count,
        selected: topCentroids.length, backend: index.simd ? 'simd' : index.workerPool?.backend || 'scalar' });
    const query = new Float32Array(l2Normalize(queryEmbedding));
    let totalCompared = 0, completedClusters = 0, lastYield = performance.now();
    const clusterResults = new Array(topCentroids.length);
    const searchCluster = async (centroid, taskSignal, clusterIndex) => {
        // Cached searches must yield too, so a worker can receive cancellation
        // messages and the main-thread fallback can handle input and paint.
        if (performance.now() - lastYield > 12) {
            await new Promise(resolve => setTimeout(resolve, 0));
            lastYield = performance.now();
        }
        taskSignal.throwIfAborted();
        const cluster = await loadSearchCluster(searchDataUrl('clusters', centroid.id, quantization),
            taskSignal, diagnostics, quantization, index.vectorLength);
        taskSignal.throwIfAborted();
        const scoringStarted = performance.now();
        const matches = scoreSearchCluster(query, cluster, threshold);
        diagnostics?.clusters.push({ id: centroid.id, vectors: cluster.count, matches: matches.length,
            scoringMs: performance.now() - scoringStarted });
        totalCompared += cluster.count;
        clusterResults[clusterIndex] = matches.map(result => ({ ...result, clusterId: centroid.id }));
        completedClusters++;
        // One update per completed concurrency batch keeps deep searches useful
        // without flooding the worker boundary or the main thread.
        if (onProgress && completedClusters < topCentroids.length
            && completedClusters % MAX_CONCURRENT_CLUSTER_REQUESTS === 0) {
            onProgress({ results: mergeClusterResults(clusterResults), totalCompared,
                completedClusters, totalClusters: topCentroids.length });
        }
        return clusterResults[clusterIndex];
    };
    await mapWithConcurrency(topCentroids, MAX_CONCURRENT_CLUSTER_REQUESTS, searchCluster, signal);
    signal.throwIfAborted();
    const mergeStarted = performance.now();
    const results = mergeClusterResults(clusterResults);
    recordSearchStage(diagnostics, 'resultMerge', mergeStarted);
    recordSearchStage(diagnostics, 'retrieval', started, { totalCompared, matches: results.length,
        clusters: topCentroids.length, quantization, concurrency: MAX_CONCURRENT_CLUSTER_REQUESTS, clusterCacheBytes: clusterCache.bytes });
    return { results, totalCompared };
}

async function mapWithConcurrency(items, limit, task, signal) {
    signal.throwIfAborted();
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
    const results = new Array(items.length);
    let next = 0, failure;
    async function worker() {
        while (!failure && next < items.length) {
            const index = next++;
            try {
                controller.signal.throwIfAborted();
                results[index] = await task(items[index], controller.signal, index);
            } catch (error) {
                failure = failure || error;
                controller.abort(error);
            }
        }
    }
    // Drain active tasks before returning an error so no old work is left
    // mutating caches/diagnostics after its search has finished.
    try { await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker)); }
    finally { signal.removeEventListener('abort', abort); }
    if (failure) throw failure;
    return results;
}

let retrievalWorker = null, workerUnavailable = false, workerRequestSequence = 0;
const workerRequests = new Map();
function getRetrievalWorker() {
    if (workerUnavailable || typeof Worker === 'undefined' || typeof document === 'undefined') return null;
    if (retrievalWorker) return retrievalWorker;
    try {
        retrievalWorker = new Worker(new URL('search-worker.js?v=20261117-echicode-data', searchScriptUrl || document.baseURI));
        retrievalWorker.onmessage = ({ data }) => {
            const pending = workerRequests.get(data.id);
            if (!pending) return;
            if (data.progress) {
                pending.onProgress?.(data.progress);
                return;
            }
            workerRequests.delete(data.id);
            pending.cleanup();
            mergeSearchDiagnostics(pending.diagnostics, data.diagnostics);
            if (data.error) {
                const error = new Error(data.error.message); error.name = data.error.name;
                pending.reject(error);
            } else pending.resolve(data.result);
        };
        retrievalWorker.onerror = event => {
            event.preventDefault();
            disableRetrievalWorker(new Error(event.message || 'Retrieval worker failed'));
        };
        retrievalWorker.onmessageerror = () => disableRetrievalWorker(new Error('Invalid retrieval worker response'));
    } catch (error) { disableRetrievalWorker(error); }
    return retrievalWorker;
}

function disableRetrievalWorker(cause) {
    workerUnavailable = true;
    retrievalWorker?.terminate();
    retrievalWorker = null;
    console.warn('Retrieval worker unavailable; using the main-thread fallback.', cause);
    for (const pending of workerRequests.values()) {
        pending.cleanup();
        const error = new Error('Retrieval worker unavailable');
        error.name = 'SearchWorkerError';
        pending.reject(error);
    }
    workerRequests.clear();
}

async function runRetrievalTask(type, payload, signal, diagnostics, fallback, onProgress) {
    signal.throwIfAborted();
    const worker = getRetrievalWorker();
    if (!worker) return fallback();
    try {
        return await new Promise((resolve, reject) => {
            const id = ++workerRequestSequence;
            const abort = () => {
                workerRequests.delete(id);
                signal.removeEventListener('abort', abort);
                worker.postMessage({ type: 'cancel', id });
                reject(signal.reason);
            };
            const cleanup = () => signal.removeEventListener('abort', abort);
            workerRequests.set(id, { resolve, reject, cleanup, diagnostics, onProgress });
            signal.addEventListener('abort', abort, { once: true });
            try { worker.postMessage({ id, type, payload, searchId: diagnostics?.id }); }
            catch (error) { disableRetrievalWorker(error); }
        });
    } catch (error) {
        signal.throwIfAborted();
        if (error.name !== 'SearchWorkerError') throw error;
        return fallback();
    }
}

function prepareSearchIndex(signal, diagnostics) {
    return runRetrievalTask('index', {}, signal, diagnostics, () => ensureCentroids(signal, diagnostics));
}
function retrieveSimilarImages(embedding, options) {
    const { signal, diagnostics, threshold, clusters, quantization = 'fp16', onProgress } = options;
    validatePostingQuantization(quantization);
    return runRetrievalTask('search', { embedding, threshold, clusters, quantization }, signal, diagnostics,
        () => searchByClusters(embedding, options), onProgress);
}
