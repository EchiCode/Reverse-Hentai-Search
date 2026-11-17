// Run as RANK_SCRIPT through the CPU-only CDP harness; each thread setting needs a fresh browser.
(async () => {
    const libraryStarted = performance.now();
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
    const libraryMs = performance.now() - libraryStarted;
    const useClient = globalThis.INFERENCE_BENCH_USE_CLIENT === true;
    if (useClient) {
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = '/image-inference.js';
            script.onload = resolve;
            script.onerror = () => reject(new Error('Could not load client inference configuration'));
            document.head.appendChild(script);
        });
    }
    const modelOptions = useClient ? configureDinoInference(env) : { dtype: 'q8', device: 'wasm' };
    const requestedThreads = globalThis.INFERENCE_BENCH_THREADS;
    const prewarmCount = globalThis.INFERENCE_BENCH_PREWARM ?? 0;
    const repeats = globalThis.INFERENCE_BENCH_REPEATS ?? 7;
    const warmupCount = globalThis.INFERENCE_BENCH_WARMUPS ?? 2;
    if (![0, 2].includes(prewarmCount) || !Number.isInteger(repeats) || repeats < 1
        || !Number.isInteger(warmupCount) || warmupCount < 0) {
        throw new Error('Invalid prewarm, repeat or warmup count');
    }
    const wasm = env.backends.onnx.wasm;
    if (globalThis.INFERENCE_BENCH_WASM_PATHS) wasm.wasmPaths = globalThis.INFERENCE_BENCH_WASM_PATHS;
    if (requestedThreads !== undefined) {
        if (!Number.isInteger(requestedThreads) || requestedThreads < 1) throw new Error('Invalid thread count');
        wasm.numThreads = requestedThreads;
    }
    const started = performance.now();
    const extractor = await pipeline('image-feature-extraction', 'Xenova/dino-vits16', {
        ...modelOptions,
        ...(globalThis.INFERENCE_BENCH_SESSION_OPTIONS ? { session_options: globalThis.INFERENCE_BENCH_SESSION_OPTIONS } : {})
    });
    const modelLoadMs = performance.now() - started;
    if (useClient) profileDinoModel(extractor);
    let activeTiming;
    const timed = (callable, field) => new Proxy(callable, {
        async apply(target, receiver, args) {
            const started = performance.now();
            try {
                return await Reflect.apply(target, receiver, args);
            } finally {
                if (activeTiming) activeTiming[field] += performance.now() - started;
            }
        }
    });
    extractor.processor = timed(extractor.processor, 'processorMs');
    extractor.model = timed(extractor.model, 'modelMs');

    const canvas = () => {
        const value = document.createElement('canvas');
        value.width = value.height = 256;
        return value;
    };
    const canvases = [], sourceImages = [], labels = [], inputDetails = [];
    const galleryInputs = globalThis.INFERENCE_BENCH_IMAGES;
    if (galleryInputs !== undefined) {
        if (!Array.isArray(galleryInputs) || !galleryInputs.length) throw new Error('Missing gallery images');
        for (const source of galleryInputs) {
            if (!source || typeof source.image !== 'string' || !source.image
                || typeof source.url !== 'string' || !source.url || labels.includes(source.image)) {
                throw new Error('Invalid gallery image');
            }
            const image = new Image();
            const loadStarted = performance.now();
            await new Promise((resolve, reject) => {
                image.onload = resolve;
                image.onerror = () => reject(new Error(`Could not load gallery image: ${source.image}`));
                image.src = source.url;
            });
            const decodeStarted = performance.now();
            if (image.decode) await image.decode();
            const imageDecodeMs = performance.now() - decodeStarted;
            const imageLoadAndDecodeMs = performance.now() - loadStarted;
            const resizeStarted = performance.now();
            const value = canvas();
            const context = value.getContext('2d');
            context.imageSmoothingEnabled = true;
            context.imageSmoothingQuality = 'high';
            context.drawImage(image, 0, 0, 256, 256);
            const imageResizeMs = performance.now() - resizeStarted;
            canvases.push(value);
            sourceImages.push(image);
            labels.push(source.image);
            inputDetails.push({ image: source.image, url: source.url,
                originalWidth: image.naturalWidth, originalHeight: image.naturalHeight,
                imageLoadAndDecodeMs, imageDecodeMs, imageResizeMs });
            image.onload = image.onerror = null;
        }
    } else {
        const gradient = canvas(), noise = canvas(), logo = canvas();
        for (const [value, noisy] of [[gradient, false], [noise, true]]) {
            const context = value.getContext('2d');
            const image = context.createImageData(256, 256);
            let seed = 123456789;
            for (let y = 0; y < 256; y++) {
                for (let x = 0; x < 256; x++) {
                    const offset = (y * 256 + x) * 4;
                    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                    image.data[offset] = noisy ? seed >>> 24 : x;
                    image.data[offset + 1] = noisy ? (seed >>> 16) & 255 : y;
                    image.data[offset + 2] = noisy ? (seed >>> 8) & 255 : (x + y) >>> 1;
                    image.data[offset + 3] = 255;
                }
            }
            context.putImageData(image, 0, 0);
        }
        const context = logo.getContext('2d');
        context.fillStyle = '#162333';
        context.fillRect(0, 0, 256, 256);
        context.fillStyle = '#63b9ff';
        context.beginPath();
        context.arc(128, 128, 95, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = '#fff';
        context.fillRect(62, 91, 132, 22);
        context.fillRect(62, 143, 90, 22);
        canvases.push(gradient, noise, logo);
        labels.push('gradient', 'noise', 'logo');
    }
    const normalizeCLS = result => {
        const vector = Array.from(result.data?.slice(0, 384) ?? []);
        if (result.dims?.at(-1) !== 384 || vector.length !== 384 || !vector.every(Number.isFinite)) {
            throw new Error('Invalid 384-dimensional CLS output');
        }
        const norm = Math.hypot(...vector);
        if (!norm) throw new Error('Zero CLS output');
        return vector.map(value => value / norm);
    };
    const references = [];
    async function measure(index) {
        const timing = { image: labels[index], processorMs: 0, modelMs: 0 };
        let input = canvases[index];
        if (globalThis.INFERENCE_BENCH_FRESH_CANVAS) {
            const resizeStarted = performance.now();
            input = canvas();
            const context = input.getContext('2d', globalThis.INFERENCE_BENCH_CPU_CANVAS
                ? { willReadFrequently: true } : undefined);
            context.imageSmoothingEnabled = true;
            context.imageSmoothingQuality = 'high';
            context.drawImage(sourceImages[index] || canvases[index], 0, 0, 256, 256);
            timing.freshCanvasMs = performance.now() - resizeStarted;
        }
        activeTiming = timing;
        const started = performance.now();
        let result;
        try {
            result = await extractor(input, { pool: false });
        } finally {
            timing.pipelineMs = performance.now() - started;
            activeTiming = undefined;
        }
        timing.otherMs = timing.pipelineMs - timing.processorMs - timing.modelMs;
        if (useClient) {
            const profiled = getDinoInferenceTiming(result);
            if (!profiled) throw new Error('Client inference timing did not match the pipeline result');
            timing.profiledProcessorMs = profiled.processorMs;
            timing.profiledModelMs = profiled.modelMs;
        }
        const vector = normalizeCLS(result);
        if (!references[index]) references[index] = vector;
        const cosine = vector.reduce((sum, value, i) => sum + value * references[index][i], 0);
        if (cosine < .999999) throw new Error(`Repeated CLS output changed: ${cosine}`);
        timing.referenceCosine = cosine;
        return timing;
    }
    const stats = values => {
        const sorted = values.toSorted((a, b) => a - b);
        return { median: sorted[Math.floor(sorted.length / 2)],
            p95: sorted[Math.ceil(sorted.length * .95) - 1], samples: values };
    };
    try {
        const prewarm = [];
        const prewarmStarted = performance.now();
        if (useClient) await warmDinoModel(extractor);
        else for (let i = 0; i < prewarmCount; i++) prewarm.push(await measure((i + 1) % canvases.length));
        const prewarmMs = useClient || prewarmCount ? performance.now() - prewarmStarted : 0;
        const firstInference = await measure(0);
        const warmups = [];
        for (let i = 0; i < warmupCount; i++) warmups.push(await measure((i + 1) % canvases.length));
        const measurements = [];
        const idleMs = globalThis.INFERENCE_BENCH_IDLE_MS || 0;
        if (!Number.isFinite(idleMs) || idleMs < 0 || idleMs > 5000) throw new Error('Invalid idle interval');
        for (let i = 0; i < repeats; i++) {
            if (idleMs) await new Promise(resolve => setTimeout(resolve, idleMs));
            measurements.push(await measure(i % canvases.length));
        }
        return {
            model: 'Xenova/dino-vits16', transformers: '4.2.0', dtype: 'q8', device: 'wasm', useClient,
            requestedThreads: requestedThreads ?? null, configuredThreads: wasm.numThreads ?? null,
            wasmPaths: wasm.wasmPaths,
            sessionOptions: globalThis.INFERENCE_BENCH_SESSION_OPTIONS ?? null,
            crossOriginIsolated: globalThis.crossOriginIsolated,
            sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
            hardwareConcurrency: navigator.hardwareConcurrency,
            userAgent: navigator.userAgent, idleMs,
            freshCanvas: !!globalThis.INFERENCE_BENCH_FRESH_CANVAS,
            cpuCanvas: !!globalThis.INFERENCE_BENCH_CPU_CANVAS,
            libraryMs, modelLoadMs, prewarm, prewarmMs, readyMs: modelLoadMs + prewarmMs,
            firstInference, warmups,
            warm: Object.fromEntries(['pipelineMs', 'processorMs', 'modelMs', 'otherMs']
                .map(field => [field, stats(measurements.map(value => value[field]))])),
            measurements, normalizedCLS: labels.map((image, index) => ({ image, vector: references[index] })),
            inputKind: galleryInputs === undefined ? 'synthetic' : 'gallery', inputDetails,
            note: `${labels.length} ${galleryInputs === undefined ? 'synthetic canvases' : 'gallery images resized to 256px'}, exact client pipeline with pool:false. Compare CLS outputs across variants; not a retrieval-accuracy benchmark.`
        };
    } finally {
        await extractor.dispose();
    }
})()
