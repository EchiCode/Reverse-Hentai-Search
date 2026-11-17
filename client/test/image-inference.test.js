import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../image-inference.js', import.meta.url), 'utf8');
function helpers(platform = {}) {
    const drawing = {};
    const context = vm.createContext({ ...platform, document: {
        createElement(tag) {
            assert.equal(tag, 'canvas');
            return { getContext(kind, options) {
                assert.equal(kind, '2d');
                assert.equal(options?.willReadFrequently, true);
                return { set fillStyle(value) { drawing.color = value; },
                    fillRect(...args) { drawing.rectangle = args; } };
            } };
        }
    } });
    vm.runInContext(source + '\nglobalThis.api = { configureDinoInference, warmDinoModel, profileDinoModel, getDinoInferenceTiming };', context);
    return { ...context.api, drawing };
}

test('DINO uses the matching CPU runtime pair and preserves the thread setting', () => {
    const { configureDinoInference } = helpers({ navigator: { hardwareConcurrency: 12 }, crossOriginIsolated: true });
    for (const numThreads of [1, 4]) {
        const wasm = { numThreads };
        const options = configureDinoInference({ backends: { onnx: { wasm } } });
        const base = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/';
        assert.equal(wasm.wasmPaths.mjs, base + 'ort-wasm-simd-threaded.mjs');
        assert.equal(wasm.wasmPaths.wasm, base + 'ort-wasm-simd-threaded.wasm');
        assert.equal(wasm.numThreads, numThreads);
        assert.equal(Object.hasOwn(wasm, 'numThreads'), true);
        assert.equal(options.dtype, 'q8');
        assert.equal(options.device, 'wasm');
    }
});

test('automatic CPU threads use half the logical CPUs up to six with an isolated context', () => {
    for (const [cores, isolated, expected] of [[12, true, 6], [8, true, 4], [4, true, 2],
        [1, true, 1], [undefined, true, 1], [12, false, 1], [undefined, undefined, 1]]) {
        const { configureDinoInference } = helpers({
            navigator: { hardwareConcurrency: cores }, crossOriginIsolated: isolated
        });
        for (const initial of [undefined, 0]) {
            const wasm = initial === undefined ? {} : { numThreads: initial };
            configureDinoInference({ backends: { onnx: { wasm } } });
            assert.equal(wasm.numThreads, expected, `${cores} logical CPUs, isolated=${isolated}`);
        }
    }
    const wasm = {};
    helpers({ crossOriginIsolated: true }).configureDinoInference({ backends: { onnx: { wasm } } });
    assert.equal(wasm.numThreads, 1);
});

test('readiness waits for both serial warmups on the same neutral canvas', async () => {
    const { warmDinoModel, drawing } = helpers();
    const resolve = [], calls = [];
    let markSecondStarted;
    const secondStarted = new Promise(done => { markSecondStarted = done; });
    const model = (canvas, options) => {
        calls.push({ canvas, options });
        if (calls.length === 2) markSecondStarted();
        return new Promise(done => resolve.push(done));
    };
    let ready = false;
    const warming = warmDinoModel(model).then(result => { ready = true; return result; });
    assert.equal(calls.length, 1);
    assert.equal(ready, false);
    resolve[0]();
    await secondStarted;
    assert.equal(calls.length, 2);
    assert.equal(ready, false);
    assert.equal(calls[0].canvas, calls[1].canvas);
    for (const { canvas, options } of calls) {
        assert.equal(canvas.width, 256);
        assert.equal(canvas.height, 256);
        assert.equal(options.pool, false);
    }
    assert.equal(drawing.color, '#808080');
    assert.deepEqual(drawing.rectangle, [0, 0, 256, 256]);
    resolve[1]();
    assert.equal(await warming, model);
    assert.equal(ready, true);
});

test('a failed warmup disposes the model and preserves its original error', async () => {
    const { warmDinoModel } = helpers();
    for (const failAt of [1, 2]) {
        let calls = 0, disposals = 0;
        const failure = new Error('warmup failed');
        const model = async () => { if (++calls === failAt) throw failure; };
        model.dispose = async () => { disposals++; throw new Error('dispose failed'); };
        await assert.rejects(warmDinoModel(model), error => error === failure);
        assert.equal(calls, failAt);
        assert.equal(disposals, 1);
    }
});

test('profiling keeps concurrent calls separate and preserves inputs, outputs and disposal', async () => {
    let clock = 0;
    const { profileDinoModel, getDinoInferenceTiming } = helpers({ performance: { now: () => clock } });
    const gate = () => {
        let resolve;
        const promise = new Promise(done => { resolve = done; });
        return { promise, resolve };
    };
    const processorGates = [gate(), gate()], modelGates = [gate(), gate()];
    const images = [{ id: 0 }, { id: 1 }], inputs = [{ pixel_values: {} }, { pixel_values: {} }];
    const outputs = [0, 1].map(value => ({ last_hidden_state: { data: new Float32Array([value]) } }));
    const receiver = {}, options = {}, session = {};
    const extractor = {
        processor(image, actualOptions) {
            assert.equal(this, receiver);
            assert.equal(actualOptions, options);
            return processorGates[images.indexOf(image)].promise;
        },
        model(input, actualOptions) {
            assert.equal(this, receiver);
            assert.equal(actualOptions, options);
            return modelGates[inputs.indexOf(input)].promise;
        }
    };
    extractor.model.session = session;
    extractor.model.dispose = function () { assert.equal(this.session, session); return 'disposed'; };
    assert.equal(profileDinoModel(extractor), extractor);
    const firstProcessed = extractor.processor.call(receiver, images[0], options);
    clock = 1;
    const secondProcessed = extractor.processor.call(receiver, images[1], options);
    clock = 3;
    processorGates[1].resolve(inputs[1]);
    assert.equal(await secondProcessed, inputs[1]);
    clock = 5;
    processorGates[0].resolve(inputs[0]);
    assert.equal(await firstProcessed, inputs[0]);
    clock = 10;
    const firstOutput = extractor.model.call(receiver, inputs[0], options);
    clock = 11;
    const secondOutput = extractor.model.call(receiver, inputs[1], options);
    clock = 15;
    modelGates[1].resolve(outputs[1]);
    assert.equal(await secondOutput, outputs[1]);
    clock = 20;
    modelGates[0].resolve(outputs[0]);
    assert.equal(await firstOutput, outputs[0]);
    const first = getDinoInferenceTiming(outputs[0].last_hidden_state);
    const second = getDinoInferenceTiming(outputs[1].last_hidden_state);
    assert.equal(first.processorMs, 5);
    assert.equal(first.modelMs, 10);
    assert.equal(second.processorMs, 2);
    assert.equal(second.modelMs, 4);
    assert.notEqual(first, second);
    assert.equal(extractor.model.dispose(), 'disposed');
});

test('profiling leaves unmatched pipeline results untimed', async () => {
    const { profileDinoModel, getDinoInferenceTiming } = helpers({ performance: { now: () => 0 } });
    const output = { last_hidden_state: { data: new Float32Array([1]) } };
    const extractor = profileDinoModel({ processor: async () => ({}), model: async () => output });
    const result = await extractor.model(await extractor.processor());
    assert.equal(result, output);
    assert.equal(getDinoInferenceTiming(result.last_hidden_state), undefined);
    assert.equal(getDinoInferenceTiming(undefined), undefined);
    assert.equal(getDinoInferenceTiming({ data: new Float32Array([1]) }), undefined);
});

test('profiling propagates processor and model errors unchanged', async () => {
    const { profileDinoModel } = helpers({ performance: { now: () => 0 } });
    for (const phase of ['processor', 'model']) {
        const failure = new Error(phase + ' failed');
        const extractor = profileDinoModel({
            processor() { if (phase === 'processor') throw failure; return { pixel_values: {} }; },
            model() { throw failure; }
        });
        await assert.rejects(async () => {
            const inputs = await extractor.processor();
            await extractor.model(inputs);
        }, error => error === failure);
    }
});
