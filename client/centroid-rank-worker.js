importScripts('search.js?v=20261116-simd-centroids');

let centroidShard = null, centroidOffset = 0;
self.onmessage = ({ data }) => {
    try {
        if (data.type === 'init') {
            centroidShard = data.index;
            centroidOffset = data.offset;
            self.postMessage({ type: 'ready' });
            return;
        }
        if (data.type !== 'rank' || !centroidShard) throw new Error('Centroid worker is not initialized');
        const results = rankCentroids(data.query, centroidShard, data.probes);
        for (const result of results) result.id += centroidOffset;
        self.postMessage({ id: data.id, results });
    } catch (error) {
        self.postMessage({ id: data.id, error: { name: error.name, message: error.message } });
    }
};
