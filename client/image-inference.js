function configureDinoInference(env) {
    const wasm = env.backends.onnx.wasm;
    if (!(wasm.numThreads > 0)) {
        const cores = globalThis.navigator?.hardwareConcurrency || 2;
        wasm.numThreads = globalThis.crossOriginIsolated
            ? Math.max(1, Math.min(6, Math.ceil(cores / 2))) : 1;
    }
    // Keep this CPU pair matched to the runtime bundled with Transformers.js 4.2.0.
    const runtime = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/';
    wasm.wasmPaths = {
        mjs: runtime + 'ort-wasm-simd-threaded.mjs',
        wasm: runtime + 'ort-wasm-simd-threaded.wasm'
    };
    return { dtype: 'q8', device: 'wasm' };
}

async function warmDinoModel(model) {
    try {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 256;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.fillStyle = '#808080';
        context.fillRect(0, 0, 256, 256);
        await model(canvas, { pool: false });
        await model(canvas, { pool: false });
        return model;
    } catch (error) {
        try { await model.dispose(); } catch {}
        throw error;
    }
}

const dinoProcessorTimings = new WeakMap();
const dinoResultTimings = new WeakMap();

function profileDinoModel(extractor) {
    extractor.processor = new Proxy(extractor.processor, {
        async apply(target, receiver, args) {
            const started = performance.now();
            const result = await Reflect.apply(target, receiver, args);
            if (result?.pixel_values && typeof result.pixel_values === 'object') {
                dinoProcessorTimings.set(result.pixel_values, { processorMs: performance.now() - started });
            }
            return result;
        }
    });
    extractor.model = new Proxy(extractor.model, {
        async apply(target, receiver, args) {
            const timing = dinoProcessorTimings.get(args[0]?.pixel_values);
            const started = performance.now();
            const result = await Reflect.apply(target, receiver, args);
            const data = result?.last_hidden_state?.data;
            if (timing && ArrayBuffer.isView(data)) {
                dinoResultTimings.set(data, { ...timing, modelMs: performance.now() - started });
            }
            return result;
        }
    });
    return extractor;
}

function getDinoInferenceTiming(result) {
    return dinoResultTimings.get(result?.data);
}
