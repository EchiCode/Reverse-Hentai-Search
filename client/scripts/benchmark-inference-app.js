// Evaluate on the full app; reuse its model and embedding function without retrieving postings.
(async () => {
    const inputs = globalThis.INFERENCE_BENCH_IMAGES;
    if (!Array.isArray(inputs) || inputs.length !== 9
        || inputs.some(input => !input?.image || !input?.url)) {
        throw new Error('Expected nine gallery images');
    }
    const pause = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
    const blobs = [];
    for (const input of inputs) {
        const response = await fetch(input.url);
        if (!response.ok) throw new Error(`Gallery image HTTP ${response.status}: ${input.image}`);
        blobs.push(await response.blob());
    }
    const waitStarted = performance.now();
    while (!(typeof dinoModel !== 'undefined' && dinoModel
        && typeof searchSettingsReady !== 'undefined' && searchSettingsReady)) {
        if (performance.now() - waitStarted > 60000) throw new Error('App model/search options did not become ready');
        await pause(100);
    }
    const readyWaitMs = performance.now() - waitStarted;
    const previousImage = imageDataUrl, previousCache = cachedImageEmbedding;
    async function measure(index) {
        const url = URL.createObjectURL(blobs[index]);
        imageDataUrl = url;
        const diagnostics = createSearchDiagnostics({ benchmarkImage: inputs[index].image });
        const started = performance.now();
        try {
            const embedding = await getSearchEmbedding(url, new AbortController().signal, diagnostics);
            const totalMs = performance.now() - started;
            const vector = Array.from(embedding);
            if (vector.length !== 384 || !vector.every(Number.isFinite)
                || Math.abs(vector.reduce((sum, value) => sum + value * value, 0) - 1) > 1e-6) {
                throw new Error(`Invalid app CLS vector: ${inputs[index].image}`);
            }
            for (const stage of ['modelInference', 'modelPreprocessing', 'modelExecution', 'modelPipelineOther']) {
                if (!Number.isFinite(diagnostics.stages[stage])) throw new Error(`Missing production timing: ${stage}`);
            }
            if ('embeddingCacheHit' in diagnostics.stages || !diagnostics.inferenceRuntime) {
                throw new Error('App query used a cached embedding or omitted runtime information');
            }
            return { image: inputs[index].image, totalMs, stages: diagnostics.stages,
                runtime: diagnostics.inferenceRuntime, vector };
        } finally {
            URL.revokeObjectURL(url);
        }
    }
    try {
        const indexPendingAtFirst = Boolean(searchIndexPreload);
        const first = [await measure(0)];
        const indexWaitStarted = performance.now();
        if (searchIndexPreload) await searchIndexPreload;
        const indexWaitMs = performance.now() - indexWaitStarted;
        const steady = [];
        for (let i = 0; i < 18; i++) {
            await pause(1000);
            steady.push(await measure(i % inputs.length));
        }
        return { mode: 'full-app', readyWaitMs, indexPendingAtFirst, indexWaitMs,
            userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
            crossOriginIsolated: globalThis.crossOriginIsolated, runtime: dinoInferenceRuntime,
            images: inputs.map(({ image, url }) => ({ image, url })), first, steady,
            note: 'Actual app model and getSearchEmbedding; fresh object URL per call. No second model or posting retrieval.' };
    } finally {
        imageDataUrl = previousImage;
        cachedImageEmbedding = previousCache;
    }
})()
