// The worker owns decoded vectors and their cache. Only the query and top 50
// results cross the thread boundary, never the full index or cluster buffers.
importScripts('search-release.js?v=20261117-public-96k', 'centroid-simd.js?v=20261116-simd-centroids',
    'centroid-worker-pool.js?v=20261116-simd-centroids', 'search.js?v=20261117-public-96k');
const activeSearchTasks = new Map();
self.onmessage = async ({ data }) => {
    if (data.type === 'cancel') {
        activeSearchTasks.get(data.id)?.abort();
        return;
    }
    const controller = new AbortController();
    activeSearchTasks.set(data.id, controller);
    const diagnostics = data.searchId === undefined ? undefined : createSearchDiagnostics({}, data.searchId);
    try {
        let result;
        if (data.type === 'index') {
            const started = performance.now();
            const index = await ensureCentroids(controller.signal, diagnostics);
            recordSearchStage(diagnostics, 'indexReady', started, { centroids: index.count });
            result = { count: index.count };
        } else if (data.type === 'search') {
            result = await searchByClusters(data.payload.embedding, {
                ...data.payload, signal: controller.signal, diagnostics,
                onProgress: progress => {
                    if (!controller.signal.aborted) self.postMessage({ id: data.id, progress });
                }
            });
        } else throw new Error('Unknown retrieval task');
        controller.signal.throwIfAborted();
        self.postMessage({ id: data.id, result, diagnostics });
    } catch (error) {
        self.postMessage({ id: data.id, error: { name: error.name, message: error.message }, diagnostics });
    } finally {
        activeSearchTasks.delete(data.id);
    }
};
