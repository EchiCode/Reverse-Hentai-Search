(() => {
    const scriptUrl = (typeof document !== 'undefined' && document.currentScript?.src) || globalThis.location?.href;

    globalThis.createCentroidWorkerPool = async function createCentroidWorkerPool(index) {
        const { count, vectorLength, metric } = index;
        if (typeof Worker === 'undefined') throw new Error('Centroid workers are unavailable');
        if (!Number.isInteger(count) || count < 1 || !Number.isInteger(vectorLength) || vectorLength < 1
            || !(index.vectors instanceof Float32Array) || index.vectors.length !== count * vectorLength
            || (index.bias && (!(index.bias instanceof Float64Array) || index.bias.length !== count))) {
            throw new Error('Invalid centroid index');
        }

        let shared = null;
        if (typeof SharedArrayBuffer !== 'undefined' && globalThis.crossOriginIsolated !== false) {
            try {
                const share = values => {
                    if (!values || values.buffer instanceof SharedArrayBuffer) return values;
                    const result = new values.constructor(new SharedArrayBuffer(values.byteLength));
                    result.set(values);
                    return result;
                };
                shared = { vectors: share(index.vectors), bias: share(index.bias) };
            } catch {
                // Browsers that disallow shared memory can transfer one disjoint shard to each worker.
            }
        }
        const parts = Array.from({ length: 4 }, (_, part) => {
            const start = Math.floor(part * count / 4), end = Math.floor((part + 1) * count / 4);
            return {
                offset: start,
                index: {
                    count: end - start, vectorLength, metric,
                    vectors: shared ? shared.vectors.subarray(start * vectorLength, end * vectorLength)
                        : index.vectors.slice(start * vectorLength, end * vectorLength),
                    bias: shared ? shared.bias?.subarray(start, end) : index.bias?.slice(start, end)
                }
            };
        });

        const workers = [], pending = new Map();
        let sequence = 0, failure = null, readyCount = 0;
        let resolveReady, rejectReady;
        const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });

        function close(error = new Error('Centroid worker pool closed')) {
            if (failure) return;
            failure = error;
            for (const worker of workers) worker.terminate();
            rejectReady(error);
            for (const task of pending.values()) {
                task.cleanup();
                task.reject(error);
            }
            pending.clear();
        }

        try {
            for (let part = 0; part < 4; part++) {
                const worker = new Worker(new URL('centroid-rank-worker.js?v=20261116-simd-centroids', scriptUrl));
                workers.push(worker);
                let initialized = false;
                worker.onerror = event => {
                    event.preventDefault?.();
                    close(new Error(event.message || 'Centroid worker failed'));
                };
                worker.onmessageerror = () => close(new Error('Invalid centroid worker response'));
                worker.onmessage = ({ data }) => {
                    if (failure) return;
                    if (data.error) {
                        const error = new Error(data.error.message || 'Centroid worker failed');
                        error.name = data.error.name || 'Error';
                        close(error);
                        return;
                    }
                    if (data.type === 'ready') {
                        if (!initialized) { initialized = true; if (++readyCount === 4) resolveReady(); }
                        return;
                    }
                    const task = pending.get(data.id);
                    if (!task || task.parts[part]) return;
                    if (!Array.isArray(data.results)) { close(new Error('Invalid centroid worker results')); return; }
                    task.parts[part] = data.results;
                    if (--task.remaining !== 0) return;
                    pending.delete(data.id);
                    task.cleanup();
                    task.resolve(task.parts.flat().sort((a, b) => b.score - a.score || a.id - b.id).slice(0, task.limit));
                };
            }
            workers.forEach((worker, part) => {
                const shard = parts[part];
                const transfer = shared ? [] : [shard.index.vectors.buffer,
                    ...(shard.index.bias ? [shard.index.bias.buffer] : [])];
                worker.postMessage({ type: 'init', ...shard }, transfer);
            });
        } catch (error) {
            close(error);
        }
        await ready;
        if (shared) { index.vectors = shared.vectors; index.bias = shared.bias; }

        return {
            backend: 'workers-4',
            transferred: !shared,
            close,
            async rank(query, probes, signal) {
                signal?.throwIfAborted();
                if (failure) throw failure;
                if (query.length !== vectorLength) throw new Error('Query and centroid dimensions differ');
                const values = metric === 'l2' ? new Float32Array(query) : query;
                const limit = Math.min(count, Math.max(1, Math.floor(probes)));
                return new Promise((resolve, reject) => {
                    const id = ++sequence;
                    const abort = () => {
                        const task = pending.get(id);
                        if (!task) return;
                        pending.delete(id);
                        task.cleanup();
                        reject(signal.reason);
                    };
                    pending.set(id, {
                        resolve, reject, limit, parts: new Array(4), remaining: 4,
                        cleanup: () => signal?.removeEventListener('abort', abort)
                    });
                    signal?.addEventListener('abort', abort, { once: true });
                    try { workers.forEach(worker => worker.postMessage({ type: 'rank', id, query: values, probes: limit })); }
                    catch (error) { close(error); }
                });
            }
        };
    };
})();
