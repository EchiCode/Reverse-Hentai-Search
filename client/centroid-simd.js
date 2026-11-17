const centroidSimdAssetBase = typeof document !== 'undefined'
    ? document.currentScript?.src || document.baseURI : globalThis.location?.href;
let centroidSimdModulePromise;

function loadCentroidSimd() {
    if (!centroidSimdModulePromise) {
        centroidSimdModulePromise = (async () => {
            if (typeof WebAssembly === 'undefined' || !centroidSimdAssetBase) return null;
            const url = new URL('centroid-simd.wasm?v=20261116-simd-centroids', centroidSimdAssetBase);
            const response = await fetch(url);
            if (!response.ok) throw new Error(`SIMD kernel unavailable (HTTP ${response.status})`);
            return WebAssembly.compile(await response.arrayBuffer());
        })().catch(error => {
            console.warn('Centroid SIMD unavailable; using worker routing.', error);
            return null;
        });
    }
    return centroidSimdModulePromise;
}

function createCentroidSimd(module, count, vectorLength = 384) {
    if (!module || vectorLength !== 384) return null;
    if (!Number.isSafeInteger(count) || count < 1) throw new Error('Invalid centroid count');
    try {
        const { memory, matrix384, block_size } = new WebAssembly.Instance(module).exports;
        const blockSize = block_size?.value;
        if (![8, 16, 24, 32].includes(blockSize)) throw new Error('Unsupported centroid SIMD layout');
        const paddedCount = Math.ceil(count / blockSize) * blockSize;
        const matrixBytes = paddedCount * 384 * 4;
        const queryOffset = matrixBytes, outputOffset = queryOffset + 384 * 8;
        const totalBytes = outputOffset + paddedCount * 8;
        if (totalBytes > 0xffffffff) throw new Error('Centroid matrix exceeds WebAssembly address space');
        const pages = Math.ceil(totalBytes / 65536);
        if (pages > memory.buffer.byteLength / 65536) memory.grow(pages - memory.buffer.byteLength / 65536);
        const vectors = new Float32Array(memory.buffer, 0, paddedCount * 384);
        const values = new Float64Array(memory.buffer, queryOffset, 384);
        const output = new Float64Array(memory.buffer, outputOffset, paddedCount);
        return {
            backend: 'simd', vectors, blockSize,
            count, paddedCount, memoryBytes: memory.buffer.byteLength,
            scores(query, metric) {
                if (query.length !== 384) throw new Error('Query and centroid dimensions differ');
                if (metric === 'l2') {
                    for (let i = 0; i < 384; i++) values[i] = Math.fround(query[i]);
                } else values.set(query);
                matrix384(paddedCount, 0, queryOffset, outputOffset);
                return output;
            }
        };
    } catch (error) {
        console.warn('Centroid SIMD allocation unavailable; using worker routing.', error);
        return null;
    }
}
