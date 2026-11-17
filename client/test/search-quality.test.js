import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = ['search.js', 'search-quality.js', 'app.js'].map(file => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')).join('\n');
const centroidHash = 'd1ff168ac7a5f0475b0ebcd1c665a55881485bace9e4398332bb1010c6276d42';
function manifest() {
    return { schema_version: 1, head_count: 8000, dimensions: 384, centroid_sha256: centroidHash,
        formats: Object.fromEntries(['fp16', 'lvq6', 'lvq8'].map((format, index) => [format,
            { path: format === 'fp16' ? 'clusters' : `clusters_${format}`,
                gzip_bytes: Array.from({ length: 8000 }, (_, row) => (index + 1) * (row % 2 ? 200000 : 100000)) }])) };
}

function largeManifest(headCount) {
    const totals = headCount === 96000
        ? { fp16: 21788572869, lvq6: 8715939825, lvq8: 11145672466 }
        : { fp16: 21652918918, lvq6: 8697341077, lvq8: 11110782575 };
    return { schema_version: 1, head_count: headCount, dimensions: 384, centroid_sha256: 'b'.repeat(64),
        formats: Object.fromEntries(Object.entries(totals).map(([format, total]) => [format,
            { path: format === 'fp16' ? 'clusters' : `clusters_${format}`,
                gzip_bytes: Array.from({ length: headCount }, (_, row) =>
                    Math.floor(total / headCount) + (row < total % headCount ? 1 : 0)) }])) };
}
const manifest128k = () => largeManifest(128000);

function client(response = { ok: true, json: async () => manifest() }, hostname = 'localhost') {
    const elements = new Map();
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            value: '', disabled: false, checked: false, textContent: '', attributes: {}, listeners: {},
            options: ['lvq6', 'lvq8', 'fp16'].map(value => ({ value, disabled: value !== 'fp16' })),
            addEventListener(type, listener) { this.listeners[type] = listener; },
            setAttribute(name, value) { this.attributes[name] = value; },
            emit(type, properties = {}) { Object.assign(this, properties); this.listeners[type]?.({ target: this }); }
        });
        return elements.get(id);
    }
    const searches = [];
    const context = vm.createContext({
        AbortController, setTimeout, clearTimeout, console: { warn() {}, log() {} }, location: { hostname },
        fetch: async url => { assert.equal(url, 'data/posting-codecs.json'); return response; },
        document: { addEventListener() {}, getElementById: element }, searches
    });
    vm.runInContext(`${source}
        handleSearchResult = (image, controller, settings) => searches.push(settings);
        globalThis.qualityTests = { setupClusterSlider, setupExtraPrecisionToggle, setupSearchQualityControl, setupThresholdControl,
            loadPostingFormats, performSearch,
            getSettings: () => ({ quantization: currentSearchQuantization, clusters: currentSearchClusters }),
            useDataset: dataset => Object.assign(SEARCH_DATASET, dataset),
            readyImage: () => { imageDataUrl = 'image:test'; dinoModel = {}; }
        };`, context);
    const api = context.qualityTests;
    api.setupClusterSlider();
    api.setupExtraPrecisionToggle();
    api.setupSearchQualityControl();
    return { api, element, searches };
}

test('quality levels use current average posting sizes and forward codec/probe pairs to retrieval', async () => {
    const { api, element, searches } = client();
    assert.equal(element('qualitySlider').disabled, true);
    await api.loadPostingFormats();
    assert.equal(element('qualityValue').textContent, 'Balanced');
    const cases = [
        ['lvq6', 8, '2.4'], ['lvq8', 16, '7.2'], ['fp16', 16, '2.4'], ['fp16', 32, '4.8']
    ];
    for (const [level, [quantization, clusters, size]] of cases.entries()) {
        element('qualitySlider').emit('input', { value: String(level) });
        assert.deepEqual({ ...api.getSettings() }, { quantization, clusters });
        assert.equal(element('qualityEstimate').textContent, `Estimated ${size} MB per search (average cluster sizes)`);
        assert.equal(element('clusterInput').disabled, true);
        assert.equal(element('quantizationSelect').disabled, true);
        assert.equal(element('manualSearchCheckbox').checked, false);
    }
    api.readyImage();
    api.performSearch();
    assert.equal(searches[0].quantization, 'fp16');
    assert.equal(searches[0].clusters, 32);
});

test('manual settings override presets and moving quality slider restores automatic choices', async () => {
    const { api, element } = client();
    await api.loadPostingFormats();
    element('manualSearchCheckbox').emit('change', { checked: true });
    assert.equal(element('quantizationSelect').disabled, false);
    assert.equal(element('clusterInput').disabled, false);
    element('quantizationSelect').emit('change', { value: 'lvq6' });
    element('clusterInput').emit('input', { value: '12' });
    assert.deepEqual({ ...api.getSettings() }, { quantization: 'lvq6', clusters: 12 });
    assert.equal(element('qualityValue').textContent, 'Custom');
    assert.match(element('qualitySlider').attributes['aria-valuetext'], /^Custom,/);
    assert.equal(element('qualityEstimate').textContent, 'Estimated 3.6 MB per search (average cluster sizes)');
    element('extraPrecisionCheckbox').emit('change', { checked: true });
    element('clusterInput').emit('input', { value: '9999' });
    assert.equal(api.getSettings().clusters, 1000);
    element('qualitySlider').emit('input', { value: '1' });
    assert.deepEqual({ ...api.getSettings() }, { quantization: 'lvq8', clusters: 16 });
    assert.equal(element('qualityValue').textContent, 'Balanced');
    assert.equal(element('clusterInput').disabled, true);
    element('qualitySlider').emit('input', { value: '1.5' });
    assert.deepEqual({ ...api.getSettings() }, { quantization: 'lvq8', clusters: 16 });
});

test('search uses a threshold change immediately', async () => {
    const { api, element, searches } = client();
    api.setupThresholdControl();
    await api.loadPostingFormats();
    api.readyImage();
    element('thresholdInput').emit('input', { value: '95' });
    api.performSearch();
    assert.equal(searches[0].threshold, 0.95);
    assert.equal(element('thresholdSlider').value, 95);
    assert.equal(element('thresholdValue').textContent, '95%');
});

test('hosted pages cannot mix the local 8k quality presets with the legacy hosted corpus', async () => {
    const { api, element } = client(undefined, 'example.test');
    await api.loadPostingFormats();
    assert.equal(element('qualitySlider').disabled, true);
    assert.equal(element('quantizationSelect').disabled, true);
    assert.deepEqual({ ...api.getSettings() }, { quantization: 'fp16', clusters: 10 });
});

test('missing, incomplete or different datasets retain manual FP16 without advertising budget estimates', async () => {
    const invalid = [
        { ok: false, status: 404 },
        { ok: true, json: async () => ({ ...manifest(), centroid_sha256: 'different' }) },
        { ok: true, json: async () => ({ ...manifest(), formats: { fp16: manifest().formats.fp16 } }) },
        { ok: true, json: async () => ({ ...manifest(), formats: {
            ...manifest().formats, lvq6: { path: '../wrong', gzip_bytes: [0] }
        } }) },
        { ok: true, json: async () => ({ ...manifest(), formats: {
            ...manifest().formats, lvq6: { ...manifest().formats.lvq6, gzip_bytes: Array(8000).fill(-1) }
        } }) }
    ];
    for (const response of invalid) {
        const { api, element } = client(response);
        await api.loadPostingFormats();
        assert.equal(element('qualitySlider').disabled, true);
        assert.equal(element('manualSearchCheckbox').checked, true);
        assert.equal(element('advancedOptions').open, true);
        assert.equal(element('clusterInput').disabled, false);
        assert.equal(element('quantizationSelect').disabled, true);
        assert.deepEqual({ ...api.getSettings() }, { quantization: 'fp16', clusters: 10 });
        assert.match(element('qualityEstimate').textContent, /unavailable/);
    }
});

test('128k release uses measured, widely spaced presets and query-average download estimates', async () => {
    const data = manifest128k();
    const { api, element, searches } = client({ ok: true, json: async () => data });
    api.useDataset({ headCount: data.head_count, centroidSha256: data.centroid_sha256 });
    await api.loadPostingFormats();
    assert.equal(element('qualityValue').textContent, 'Balanced');
    assert.deepEqual({ ...api.getSettings() }, { quantization: 'lvq8', clusters: 32 });
    for (const [level, [quantization, clusters, size]] of [
        ['lvq6', 16, '1.36'], ['lvq8', 32, '3.44'], ['lvq8', 64, '6.79'], ['fp16', 64, '13.25']
    ].entries()) {
        element('qualitySlider').emit('input', { value: String(level) });
        assert.deepEqual({ ...api.getSettings() }, { quantization, clusters });
        assert.equal(element('qualityEstimate').textContent, `Estimated ${size} MB per search (query average)`);
    }
    api.readyImage();
    api.performSearch();
    assert.equal(searches[0].quantization, 'fp16');
    assert.equal(searches[0].clusters, 64);
    element('qualitySlider').emit('input', { value: '1' });
    element('manualSearchCheckbox').emit('change', { checked: true });
    assert.equal(element('qualityEstimate').textContent, 'Estimated 3.44 MB per search (query average)');
    element('clusterInput').emit('input', { value: '33' });
    assert.equal(element('qualityEstimate').textContent, 'Estimated 3.55 MB per search (query average)');
});

test('an unpinned 128k manifest cannot activate on the existing 8k release', async () => {
    const { api, element } = client({ ok: true, json: async () => manifest128k() });
    await api.loadPostingFormats();
    assert.equal(element('qualitySlider').disabled, true);
    assert.equal(element('manualSearchCheckbox').checked, true);
    assert.deepEqual({ ...api.getSettings() }, { quantization: 'fp16', clusters: 10 });
});

test('128k estimates track actual exported file sizes rather than fixed displayed targets', async () => {
    const data = manifest128k();
    data.formats.lvq8.gzip_bytes = data.formats.lvq8.gzip_bytes.map(bytes => bytes * 2);
    const { api, element } = client({ ok: true, json: async () => data });
    api.useDataset({ headCount: data.head_count, centroidSha256: data.centroid_sha256 });
    await api.loadPostingFormats();
    assert.equal(element('qualityEstimate').textContent, 'Estimated 6.88 MB per search (query average)');
});

test('96k release uses its measured presets and download calibration', async () => {
    const data = largeManifest(96000);
    const { api, element, searches } = client({ ok: true, json: async () => data });
    api.useDataset({ headCount: data.head_count, centroidSha256: data.centroid_sha256 });
    await api.loadPostingFormats();
    assert.equal(element('qualityValue').textContent, 'Balanced');
    for (const [level, [quantization, clusters, size]] of [
        ['lvq6', 16, '1.71'], ['lvq8', 32, '4.33'], ['lvq8', 64, '8.59'], ['fp16', 64, '16.79']
    ].entries()) {
        element('qualitySlider').emit('input', { value: String(level) });
        assert.deepEqual({ ...api.getSettings() }, { quantization, clusters });
        assert.equal(element('qualityEstimate').textContent, `Estimated ${size} MB per search (query average)`);
    }
    api.readyImage();
    api.performSearch();
    assert.equal(searches[0].quantization, 'fp16');
    assert.equal(searches[0].clusters, 64);
    element('qualitySlider').emit('input', { value: '1' });
    element('manualSearchCheckbox').emit('change', { checked: true });
    assert.equal(element('qualityEstimate').textContent, 'Estimated 4.33 MB per search (query average)');
    element('clusterInput').emit('input', { value: '48' });
    assert.equal(element('qualityEstimate').textContent, 'Estimated 6.46 MB per search (query average)');
});

test('96k estimates rescale to the final build and an unpinned release cannot activate', async () => {
    const data = largeManifest(96000);
    const unpinned = client({ ok: true, json: async () => data });
    await unpinned.api.loadPostingFormats();
    assert.equal(unpinned.element('qualitySlider').disabled, true);
    assert.deepEqual({ ...unpinned.api.getSettings() }, { quantization: 'fp16', clusters: 10 });

    data.formats.lvq8.gzip_bytes = data.formats.lvq8.gzip_bytes.map(bytes => bytes * 2);
    const { api, element } = client({ ok: true, json: async () => data });
    api.useDataset({ headCount: data.head_count, centroidSha256: data.centroid_sha256 });
    await api.loadPostingFormats();
    assert.equal(element('qualityEstimate').textContent, 'Estimated 8.67 MB per search (query average)');
});
