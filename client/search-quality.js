function getSearchQualityLevels(headCount) {
    if (headCount === 96000 || headCount === 128000) return [
        { label: 'Budget', quantization: 'lvq6', clusters: 16 },
        { label: 'Balanced', quantization: 'lvq8', clusters: 32 },
        { label: 'High', quantization: 'lvq8', clusters: 64 },
        { label: 'Very high', quantization: 'fp16', clusters: 64 }
    ];
    if (headCount === 8000) return [
        { label: 'Budget', quantization: 'lvq6', clusters: 8 },
        { label: 'Balanced', quantization: 'lvq8', clusters: 16 },
        { label: 'High', quantization: 'fp16', clusters: 16 },
        { label: 'Very high', quantization: 'fp16', clusters: 32 }
    ];
    throw new Error('No quality profile for this dataset');
}

// 100k-query, noise-0.025 calibration for LVQ6 centroids. Selected clusters
// are larger than the uniform file average; rescale to the current build's mean.
const SEARCH_DOWNLOAD_CALIBRATION = {
    96000: {
        fp16: { fileMeanBytes: 21788572869 / 96000,
            queryMB: [1.07612730673, 2.14428996097, 4.26630345308, 8.47128208295, 16.79323709838] },
        lvq6: { fileMeanBytes: 8715939825 / 96000,
            queryMB: [0.43033341388, 0.85768112446, 1.70669650397, 3.38950536402, 6.72044374759] },
        lvq8: { fileMeanBytes: 11145672466 / 96000,
            queryMB: [0.55030873644, 1.09681652316, 2.18266706906, 4.33478804737, 8.59464911899] }
    },
    128000: {
        fp16: { fileMeanBytes: 169163.429046875,
            queryMB: [0.86023239601, 1.7096795893, 3.38990253895, 6.71101440643, 13.25096489715] },
        lvq6: { fileMeanBytes: 67947.9771640625,
            queryMB: [0.34489679699, 0.68561251469, 1.35940840291, 2.69153915014, 5.31566427606] },
        lvq8: { fileMeanBytes: 86802.9888671875,
            queryMB: [0.44077831548, 0.87621939933, 1.73749388943, 3.44025386479, 6.79419450527] }
    }
};

function estimateSearchDownloadMB(headCount, quantization, probes, fileMeanBytes) {
    if (!SEARCH_DOWNLOAD_CALIBRATION[headCount]) return fileMeanBytes * probes / 1e6;
    const calibration = SEARCH_DOWNLOAD_CALIBRATION[headCount][quantization];
    const grid = [4, 8, 16, 32, 64], values = calibration.queryMB;
    let megabytes;
    if (probes <= grid[0]) megabytes = values[0] * probes / grid[0];
    else {
        let upper = grid.findIndex(value => value >= probes);
        if (upper === -1) upper = grid.length - 1;
        const lower = upper - 1;
        megabytes = values[lower] + (values[upper] - values[lower])
            * (probes - grid[lower]) / (grid[upper] - grid[lower]);
    }
    return megabytes * fileMeanBytes / calibration.fileMeanBytes;
}
