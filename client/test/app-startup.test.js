import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

test('page startup starts the model while centroid loading is pending and shares the preload', async () => {
    let onReady;
    const context = vm.createContext({ AbortController,
        console: { warn() {}, log() {} },
        document: { addEventListener(event, callback) {
            if (event === 'DOMContentLoaded') onReady = callback;
        } } });
    const source = ['search.js', 'search-quality.js', 'app.js']
        .map(file => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')).join('\n');
    vm.runInContext(source + `
        const started = [];
        let finishIndex;
        prepareSearchIndex = () => {
            started.push('centroids');
            return new Promise(resolve => { finishIndex = resolve; });
        };
        loadDinoModel = () => { started.push('model'); return new Promise(() => {}); };
        setupEventListeners = () => {};
        initializeUI = () => {};
        globalThis.startup = { started, preloadSearchIndex, finish: () => finishIndex({ count: 96000 }) };
    `, context);
    onReady();
    assert.ok(context.startup.started.includes('centroids'));
    assert.ok(context.startup.started.includes('model'));
    const pending = context.startup.preloadSearchIndex();
    assert.equal(context.startup.started.filter(name => name === 'centroids').length, 1);
    context.startup.finish();
    assert.equal((await pending).count, 96000);
    assert.equal(context.startup.preloadSearchIndex(), pending);
});
