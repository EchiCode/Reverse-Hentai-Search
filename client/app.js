let dinoModel = null;
let dinoInferenceRuntime = null;
let imageDataUrl = null;
let selectedObjectUrl = null;
let cachedImageEmbedding = null;
let searchIndexPreload = null;
let searchInProgress = false;
let currentSearchAbortController = null;
let currentThreshold = 80;
let currentSearchClusters = 10;
let currentSearchQuantization = 'fp16';
let currentQualityLevel = 1;
let manualSearchEnabled = false;
let quantizedPostingsReady = false;
let meanPostingBytes = null;
let searchSettingsReady = false;
let extraPrecisionEnabled = false;
let imageSelectionVersion = 0;

const MAX_IMAGE_BYTES = 100 * 1024 * 1024;
const DEFAULT_MAX_CLUSTERS = 100;
const EXTRA_PRECISION_MAX_CLUSTERS = 1000;
// Switch both fields only after the complete replacement dataset is verified.
const SEARCH_DATASET = getSearchRelease() || {
    headCount: 8000,
    centroidSha256: 'd1ff168ac7a5f0475b0ebcd1c665a55881485bace9e4398332bb1010c6276d42'
};
const SEARCH_QUALITY_LEVELS = getSearchQualityLevels(SEARCH_DATASET.headCount);

function getMaxClusters() {
    return extraPrecisionEnabled ? EXTRA_PRECISION_MAX_CLUSTERS : DEFAULT_MAX_CLUSTERS;
}

function clamp(val) {
    return Math.max(1, Math.min(getMaxClusters(), parseInt(val) || 1));
}

function syncClusterControl(value = currentSearchClusters) {
    const max = getMaxClusters();
    const clusterSlider = document.getElementById('clusterSlider');
    const clusterInput = document.getElementById('clusterInput');
    const clusterValue = document.getElementById('clusterValue');
    clusterSlider.max = max;
    clusterInput.max = max;
    currentSearchClusters = clamp(value);
    clusterSlider.value = currentSearchClusters;
    clusterInput.value = currentSearchClusters;
    clusterValue.textContent = currentSearchClusters;
    syncQualitySummary();
}

document.addEventListener('DOMContentLoaded', () => {
    preloadSearchIndex();
    loadDinoModel();
    setupEventListeners();
    initializeUI();
});

function initializeUI() {
    updateResultsInfo(0);
    toggleClearButton(false);
    setupThresholdControl();
    setupClusterSlider();
    setupExtraPrecisionToggle();
    setupSearchQualityControl();
    loadPostingFormats();
}

function syncQualitySummary() {
    const level = SEARCH_QUALITY_LEVELS[currentQualityLevel];
    const label = manualSearchEnabled ? 'Custom' : level.label;
    document.getElementById('qualityValue').textContent = label;
    const megabytes = quantizedPostingsReady ? estimateSearchDownloadMB(SEARCH_DATASET.headCount,
        currentSearchQuantization, currentSearchClusters, meanPostingBytes[currentSearchQuantization]) : 0;
    const calibrated = !!SEARCH_DOWNLOAD_CALIBRATION[SEARCH_DATASET.headCount];
    const size = megabytes.toFixed(calibrated ? 2 : 1);
    const estimateBasis = calibrated ? 'query average' : 'average cluster sizes';
    document.getElementById('qualityEstimate').textContent = quantizedPostingsReady
        ? `Estimated ${size} MB per search (${estimateBasis})`
        : searchSettingsReady ? 'Quality presets are unavailable for this dataset. Use manual settings in Advanced.' : 'Loading search options…';
    document.getElementById('qualitySlider').setAttribute('aria-valuetext',
        `${label}${quantizedPostingsReady ? `, estimated ${size} megabytes per search` : ''}`);
}

function syncSearchMode() {
    document.getElementById('manualSearchCheckbox').checked = manualSearchEnabled;
    document.getElementById('qualitySlider').disabled = !quantizedPostingsReady;
    document.getElementById('quantizationSelect').value = currentSearchQuantization;
    document.getElementById('quantizationSelect').disabled = !manualSearchEnabled || !quantizedPostingsReady;
    for (const id of ['clusterSlider', 'clusterInput', 'extraPrecisionCheckbox']) {
        document.getElementById(id).disabled = !manualSearchEnabled;
    }
    syncClusterControl();
}

function applySearchQuality(value) {
    const parsed = Number(value);
    if (!quantizedPostingsReady || !Number.isInteger(parsed) || !SEARCH_QUALITY_LEVELS[parsed]) return;
    currentQualityLevel = parsed;
    const level = SEARCH_QUALITY_LEVELS[parsed];
    manualSearchEnabled = false;
    currentSearchQuantization = level.quantization;
    currentSearchClusters = level.clusters;
    document.getElementById('qualitySlider').value = parsed;
    syncSearchMode();
}

function setupSearchQualityControl() {
    document.getElementById('qualitySlider').addEventListener('input', event => applySearchQuality(event.target.value));
    document.getElementById('manualSearchCheckbox').addEventListener('change', event => {
        manualSearchEnabled = event.target.checked || !quantizedPostingsReady;
        if (manualSearchEnabled) syncSearchMode();
        else applySearchQuality(currentQualityLevel);
    });
    document.getElementById('quantizationSelect').addEventListener('change', event => {
        if (!manualSearchEnabled || !quantizedPostingsReady || !meanPostingBytes[event.target.value]) return;
        currentSearchQuantization = event.target.value;
        syncQualitySummary();
    });
    syncSearchMode();
}

async function loadPostingFormats() {
    try {
        // Hosted pages still use the legacy centroid/posting build until the complete dataset is deployed.
        if (!getSearchRelease() && !['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
            throw new Error('The quality presets require the configured local dataset');
        }
        const response = await fetch(searchDataUrl('formats'), { cache: 'no-cache' });
        if (!response.ok) throw new Error(`Search formats unavailable (HTTP ${response.status})`);
        const manifest = await response.json();
        if (manifest.schema_version !== 1 || manifest.head_count !== SEARCH_DATASET.headCount || manifest.dimensions !== 384
            || manifest.centroid_sha256 !== SEARCH_DATASET.centroidSha256) throw new Error('Search dataset differs from the configured release');
        const averages = {};
        for (const format of ['fp16', 'lvq6', 'lvq8']) {
            const entry = manifest.formats?.[format];
            if (entry?.path !== (format === 'fp16' ? 'clusters' : `clusters_${format}`)
                || !Array.isArray(entry.gzip_bytes) || entry.gzip_bytes.length !== manifest.head_count
                || !entry.gzip_bytes.every(size => Number.isSafeInteger(size) && size > 0)) {
                throw new Error('Incomplete search formats');
            }
            averages[format] = entry.gzip_bytes.reduce((total, bytes) => total + bytes, 0) / manifest.head_count;
        }
        SEARCH_QUALITY_LEVELS.splice(0, SEARCH_QUALITY_LEVELS.length, ...getSearchQualityLevels(manifest.head_count));
        meanPostingBytes = averages;
        quantizedPostingsReady = true;
        for (const option of document.getElementById('quantizationSelect').options) option.disabled = false;
        applySearchQuality(currentQualityLevel);
    } catch (error) {
        console.warn('Quality presets unavailable; full-precision manual search remains available.', error);
        manualSearchEnabled = true;
        currentSearchQuantization = 'fp16';
        document.getElementById('manualSearchCheckbox').disabled = true;
        document.getElementById('advancedOptions').open = true;
    } finally {
        searchSettingsReady = true;
        syncSearchMode();
        updateSearchButtonAvailability();
    }
}

function setupThresholdControl() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdInput = document.getElementById('thresholdInput');
    const thresholdValue = document.getElementById('thresholdValue');
    
    function updateThreshold(value) {
        currentThreshold = Math.max(10, Math.min(100, parseInt(value) || 80));
        thresholdSlider.value = currentThreshold;
        thresholdInput.value = currentThreshold;
        thresholdValue.textContent = `${currentThreshold}%`;
    }
    
    const update = event => updateThreshold(event.target.value);
    thresholdSlider.addEventListener('input', update);
    thresholdInput.addEventListener('input', update);
    updateThreshold(80);
}

function setupClusterSlider() {
    const clusterSlider = document.getElementById('clusterSlider');
    const clusterInput = document.getElementById('clusterInput');
    syncClusterControl();
    const update = event => syncClusterControl(event.target.value);
    clusterSlider.addEventListener('input', update);
    clusterInput.addEventListener('input', update);
}

function setupExtraPrecisionToggle() {
    const checkbox = document.getElementById('extraPrecisionCheckbox');
    if (!checkbox) return;
    checkbox.addEventListener('change', (e) => {
        extraPrecisionEnabled = e.target.checked;
        syncClusterControl();
    });
}

async function loadDinoModel() {
    const loadStarted = performance.now();
    const maxRetries = 3;
    let retryCount = 0;
    
    while (retryCount < maxRetries) {
        try {
            const attemptStarted = performance.now();
            updateStatus('Loading DINO model...', 'loading');
            console.log(`Loading DINO model (attempt ${retryCount + 1}/${maxRetries})`);
            
            const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
            console.log('[Model] Transformers library loaded', { durationMs: performance.now() - attemptStarted });
            
            const modelStarted = performance.now();
            const model = await pipeline('image-feature-extraction', 'Xenova/dino-vits16', configureDinoInference(env));
            const modelLoadAndInitMs = performance.now() - modelStarted;
            profileDinoModel(model);
            dinoInferenceRuntime = { threads: env.backends.onnx.wasm.numThreads,
                crossOriginIsolated: globalThis.crossOriginIsolated,
                hardwareConcurrency: globalThis.navigator?.hardwareConcurrency,
                userAgent: globalThis.navigator?.userAgent, scriptVersion: '20261117-parallel-startup' };
            console.log('[Model] CPU inference configured', dinoInferenceRuntime);
            const warmupStarted = performance.now();
            dinoModel = await warmDinoModel(model);
            console.log('[Model] DINO ready', { modelLoadAndInitMs, modelWarmupMs: performance.now() - warmupStarted,
                totalMs: performance.now() - loadStarted, attempt: retryCount + 1 });
            
            updateStatus('DINO model loaded! Ready to search.', 'success');
            updateSearchButtonAvailability();
            return;
        }
        catch (error) {
            retryCount++;
            console.error(`Error loading DINO model (attempt ${retryCount}):`, error);
            
            if (retryCount >= maxRetries) {
                updateStatus(`Failed to load DINO model after ${maxRetries} attempts. Please refresh the page.`, 'error');
            } else {
                updateStatus(`Loading failed, retrying... (${retryCount}/${maxRetries})`, 'error');
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
}

function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const searchBtn = document.getElementById('searchBtn');
    const clearBtn = document.getElementById('clearBtn');

    uploadArea.addEventListener('click', (event) => {
        if (event.target !== fileInput) fileInput.click();
    });

    uploadArea.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            if (file.type.startsWith('image/')) {
                handleFile(file);
            } else {
                updateStatus('Please select a valid image file', 'error');
            }
        }
    });

    searchBtn.addEventListener('click', performSearch);
    clearBtn.addEventListener('click', clearAll);
    
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !searchBtn.disabled) {
            e.preventDefault();
            performSearch();
        }
    });
}

function handleFile(file) {
    if (file.size > MAX_IMAGE_BYTES) {
        updateStatus('File too large. Please select an image of 100 MB or less.', 'error');
        return;
    }

    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        updateStatus('Please select a valid image file (JPG, PNG, GIF, WebP)', 'error');
        return;
    }

    preloadSearchIndex();
    imageDataUrl = null;
    cachedImageEmbedding = null;
    invalidateSearch();
    if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
    const selectionVersion = ++imageSelectionVersion;
    const objectUrl = URL.createObjectURL(file);
    selectedObjectUrl = objectUrl;
    updateSearchButtonAvailability();
    toggleClearButton(true);
    updateStatus('Loading image...', 'loading');
    const preview = document.createElement('img');
    const imageLoadStarted = performance.now();
    preview.alt = 'Selected image';
    preview.onload = () => {
        if (selectionVersion !== imageSelectionVersion) return;
        console.log('[Image] Selected image loaded', { loadMs: performance.now() - imageLoadStarted,
            compressedBytes: file.size, width: preview.naturalWidth, height: preview.naturalHeight });
        imageDataUrl = objectUrl;
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.replaceChildren(preview, document.getElementById('fileInput'));
        uploadArea.classList.add('has-image');
        updateSearchButtonAvailability();
        updateStatus('Image ready. Select Find Source to search.', 'success', 'Image ready.');
    };
    preview.onerror = () => {
        if (selectionVersion !== imageSelectionVersion) return;
        URL.revokeObjectURL(objectUrl);
        selectedObjectUrl = null;
        updateStatus('Unable to decode this image. Please try another file.', 'error');
    };
    preview.src = objectUrl;
}

function preloadSearchIndex() {
    if (searchIndexPreload) return searchIndexPreload;
    const controller = new AbortController();
    const pending = prepareSearchIndex(controller.signal);
    searchIndexPreload = pending;
    pending.catch(error => {
        if (searchIndexPreload === pending) searchIndexPreload = null;
        console.warn('[Index] Image-selection preload failed; search will retry.', error);
    });
    return pending;
}

function clearAll() {
    cachedImageEmbedding = null;
    if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
    selectedObjectUrl = null;
    imageSelectionVersion++;
    invalidateSearch();
    imageDataUrl = null;
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    fileInput.value = '';
    uploadArea.innerHTML = `
        <span class="upload-icon" aria-hidden="true">📁</span>
        <p class="upload-text">Click to select image</p>
        <p class="upload-hint">JPG, PNG, GIF or WebP · Up to 100&nbsp;MB</p>
    `;
    uploadArea.appendChild(fileInput);
    uploadArea.classList.remove('has-image');
    document.getElementById('searchBtn').disabled = true;
    toggleClearButton(false);
    
    const resultsDiv = document.getElementById('results');
    delete resultsDiv.dataset.progressiveSearch;
    resultsDiv.innerHTML = `
        <div class="placeholder">
            <span class="placeholder-icon" aria-hidden="true">📄</span>
            <div class="placeholder-title">No search performed yet</div>
            <div class="placeholder-subtitle">Upload an image and click &quot;Find Source&quot; to search</div>
        </div>
    `;
    updateResultsInfo(0);
    updateStatus('', '');
}

function toggleClearButton(visible) {
    document.getElementById('clearBtn').style.display = visible ? 'flex' : 'none';
}

function updateSearchButtonAvailability() {
    document.getElementById('searchBtn').disabled = !imageDataUrl || !dinoModel || !searchSettingsReady;
}

function invalidateSearch() {
    currentSearchAbortController?.abort();
    currentSearchAbortController = null;
    resetSearchButton();
}

function performSearch() {
    if (searchInProgress) {
        invalidateSearch();
        updateStatus('Search cancelled.', '');
        return;
    }
    if (!imageDataUrl || !dinoModel || !searchSettingsReady) {
        updateStatus(!searchSettingsReady ? 'Search options are still loading' : dinoModel ? 'Please select an image first' : 'The model is still loading', 'error');
        return;
    }
    searchInProgress = true;
    const controller = new AbortController();
    currentSearchAbortController = controller;
    document.getElementById('searchBtn').innerHTML = '<svg class="icon" aria-hidden="true"><use href="#icon-stop"/></svg><span class="btn-text">Cancel Search</span>';
    handleSearchResult(imageDataUrl, controller, {
        threshold: currentThreshold / 100,
        clusters: clamp(currentSearchClusters),
        quantization: currentSearchQuantization
    });
}

async function getSearchEmbedding(imageUrl, signal, diagnostics) {
    if (dinoInferenceRuntime) diagnostics.inferenceRuntime = { ...dinoInferenceRuntime };
    signal.throwIfAborted();
    if (cachedImageEmbedding?.url === imageUrl) {
        console.log(`[Search #${diagnostics.id}] Reusing image embedding`);
        diagnostics.stages.embeddingCacheHit = 0;
        return cachedImageEmbedding.vector;
    }
    const imageStarted = performance.now();
    const img = new Image();
    await new Promise((resolve, reject) => {
        const cleanup = () => {
            signal.removeEventListener('abort', abort);
            img.onload = img.onerror = null;
        };
        const abort = () => { cleanup(); img.src = ''; reject(signal.reason); };
        img.onload = () => { cleanup(); resolve(); };
        img.onerror = () => { cleanup(); reject(new Error('Failed to load image')); };
        signal.addEventListener('abort', abort, { once: true });
        img.src = imageUrl;
    });
    signal.throwIfAborted();
    const decodeStarted = performance.now();
    // onload alone can leave pixel decoding deferred until drawImage.
    if (img.decode) await img.decode();
    signal.throwIfAborted();
    recordSearchStage(diagnostics, 'imageDecode', decodeStarted, { explicitDecode: Boolean(img.decode) });
    recordSearchStage(diagnostics, 'imageLoadAndDecode', imageStarted,
        { width: img.naturalWidth, height: img.naturalHeight });

    const resizeStarted = performance.now();
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 256;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, 256, 256);
    recordSearchStage(diagnostics, 'imageResize', resizeStarted);

    const inferenceStarted = performance.now();
    // Transformers.js accepts canvases directly. Avoiding a PNG data URL skips
    // lossless encoding, base64 allocation, and decoding the same pixels again.
    const result = await dinoModel(canvas, { pool: false });
    const inferenceMs = recordSearchStage(diagnostics, 'modelInference', inferenceStarted);
    const inferenceTiming = getDinoInferenceTiming(result);
    if (inferenceTiming) {
        recordSearchDuration(diagnostics, 'modelPreprocessing', inferenceTiming.processorMs);
        recordSearchDuration(diagnostics, 'modelExecution', inferenceTiming.modelMs);
        recordSearchDuration(diagnostics, 'modelPipelineOther', Math.max(0,
            inferenceMs - inferenceTiming.processorMs - inferenceTiming.modelMs));
    }
    signal.throwIfAborted();

    const normalizationStarted = performance.now();
    // Match hash.py: select the CLS token before normalizing, so L2 routing sees a unit query.
    const embedding = Array.from(result.data?.slice(0, 384) ?? []);
    if (embedding.length !== 384 || result.dims?.at(-1) !== 384
        || !embedding.every(Number.isFinite) || !embedding.some(value => value !== 0)) {
        throw new Error('DINO model did not return a valid embedding');
    }
    const finalEmbedding = l2Normalize(embedding);
    recordSearchStage(diagnostics, 'queryNormalization', normalizationStarted, { dimensions: finalEmbedding.length });
    // Cancelled or replaced images must not populate the next search's cache.
    if (imageDataUrl === imageUrl) cachedImageEmbedding = { url: imageUrl, vector: finalEmbedding };
    return finalEmbedding;
}

async function handleSearchResult(imageDataUrl, controller, settings) {
    const { signal } = controller;
    const diagnostics = createSearchDiagnostics(settings);
    console.log(`[Search #${diagnostics.id}] Started`, settings);
    let outcome = 'cancelled', summary = {};
    const progressiveResults = createProgressiveResultsRenderer(controller, diagnostics.id);
    try {
        updateStatus('Processing image with DINO...', 'loading');
        const finalEmbedding = await getSearchEmbedding(imageDataUrl, signal, diagnostics);
        signal.throwIfAborted();
        updateStatus('Searching for sources...', 'loading');

        const searchStart = performance.now();
        const { results: similarities, totalCompared } = await retrieveSimilarImages(finalEmbedding,
            { ...settings, signal, diagnostics, onProgress: progressiveResults.update });
        recordSearchStage(diagnostics, 'searchIncludingIndexWait', searchStart,
            { totalCompared, matches: similarities.length });

        let mostLikely = null;
        let metadataUnavailable = false;
        if (similarities.length > 0) {
            const metadataStarted = performance.now();
            try {
                const topResult = similarities[0];
                const metadata = await fetchClusterMetaBinary(searchDataUrl('meta', topResult.code), signal, diagnostics);
                const match = metadata.find(m => m.code === topResult.code);
                if (match) mostLikely = { ...match, similarity: topResult.similarity, code: match.code, page: topResult.page };
            } catch (error) {
                signal.throwIfAborted();
                metadataUnavailable = true;
                console.warn(`[Search #${diagnostics.id}] Match details could not be loaded:`, error);
            } finally {
                recordSearchStage(diagnostics, 'metadata', metadataStarted, { unavailable: metadataUnavailable });
            }
        }
        signal.throwIfAborted();
        await progressiveResults.finish();
        signal.throwIfAborted();
        const renderStarted = performance.now();
        displayResults(similarities, mostLikely);
        recordSearchStage(diagnostics, 'renderResults', renderStarted);
        // Report the full search through result rendering, using the same
        // snapshot for the visible duration and the completed console summary.
        const totalMs = performance.now() - diagnostics.startedAt;
        updateStatus(`Search completed! Found ${similarities.length} sources in ${(totalMs/1000).toFixed(2)}s.${metadataUnavailable ? ' Match details are unavailable; source links are shown below.' : ''}`, 'success');
        outcome = 'completed';
        summary = { totalMs, totalCompared, matches: similarities.length, metadataUnavailable };
    } catch (error) {
        progressiveResults.cancel();
        if (!signal.aborted) {
            outcome = 'failed';
            summary = { error: error.message };
            console.error(`[Search #${diagnostics.id}] Search error:`, error);
            updateStatus(`Search error: ${error.message}`, 'error');
            controller.abort();
        }
    } finally {
        finishSearchDiagnostics(diagnostics, outcome, summary);
        if (currentSearchAbortController === controller) resetSearchButton();
    }
}

function createProgressiveResultsRenderer(controller, searchId) {
    const resultsDiv = document.getElementById('results');
    const resultsInfo = document.getElementById('resultsInfo');
    const resultsCount = document.getElementById('resultsCount');
    let frame = 0, latest = null, previousNodes = null, previousCount = '';
    let previousInfoDisplay = '', progressive = null, list = null, progressLabel = null, empty = null;
    let linkInteractionInProgress = false;

    function ownsResults() {
        return resultsDiv.dataset.progressiveSearch === String(searchId);
    }

    function schedulePaint() {
        if (latest && !frame && !linkInteractionInProgress) frame = requestAnimationFrame(paint);
    }

    function beginLinkInteraction(event) {
        if (event.target.closest?.('a')) linkInteractionInProgress = true;
    }

    function endLinkInteraction() {
        if (!linkInteractionInProgress) return;
        linkInteractionInProgress = false;
        // The click generated by pointerup/keyup runs before this animation
        // frame, so an active anchor is never detached mid-activation.
        schedulePaint();
    }

    function removeInteractionListeners() {
        resultsDiv.removeEventListener('pointerdown', beginLinkInteraction);
        resultsDiv.removeEventListener('keydown', beginLinkInteraction);
        window.removeEventListener('pointerup', endLinkInteraction);
        window.removeEventListener('pointercancel', endLinkInteraction);
        window.removeEventListener('keyup', endLinkInteraction);
    }

    resultsDiv.addEventListener('pointerdown', beginLinkInteraction);
    resultsDiv.addEventListener('keydown', beginLinkInteraction);
    window.addEventListener('pointerup', endLinkInteraction);
    window.addEventListener('pointercancel', endLinkInteraction);
    window.addEventListener('keyup', endLinkInteraction);

    function ensureProgressiveResults() {
        if (progressive) return;
        previousNodes = Array.from(resultsDiv.childNodes);
        previousCount = resultsCount.textContent;
        previousInfoDisplay = resultsInfo.style.display;
        progressive = document.createElement('div');
        progressive.className = 'progressive-results';
        progressLabel = document.createElement('p');
        progressLabel.className = 'progressive-results-label';
        progressLabel.append('Top matches so far ', document.createElement('span'));
        list = document.createElement('div');
        list.className = 'results-list progressive-results-list';
        empty = document.createElement('p');
        empty.className = 'progressive-results-empty';
        empty.textContent = 'No matches above the threshold yet.';
        progressive.append(progressLabel, list, empty);
        resultsDiv.dataset.progressiveSearch = String(searchId);
        resultsDiv.replaceChildren(progressive);
    }

    function paint() {
        frame = 0;
        if (!latest || linkInteractionInProgress || controller.signal.aborted
            || currentSearchAbortController !== controller) return;
        ensureProgressiveResults();
        const { results, completedClusters, totalClusters } = latest;
        progressLabel.lastElementChild.textContent = `${completedClusters} of ${totalClusters} clusters searched`;
        const visibleResults = results.slice(0, 10);
        const existing = new Map(Array.from(list.children,
            node => [node.dataset.resultKey, node]));
        for (let index = 0; index < visibleResults.length; index++) {
            const item = visibleResults[index];
            const key = `${item.code}:${item.page}`;
            let node = existing.get(key);
            if (!node) {
                node = createResultItem(item, index + 1);
                node.dataset.resultKey = key;
            } else {
                const rank = node.querySelector('.result-rank');
                rank.textContent = `#${index + 1}`;
                rank.setAttribute('aria-label', `Match ${index + 1}`);
                existing.delete(key);
            }
            const position = list.children[index];
            if (position !== node) list.insertBefore(node, position || null);
        }
        for (const node of existing.values()) node.remove();
        list.hidden = visibleResults.length === 0;
        empty.hidden = visibleResults.length !== 0;
        updateResultsInfo(results.length);
        updateStatus(`Searching for sources... ${completedClusters} of ${totalClusters} clusters searched.`, 'loading');
    }

    return {
        update(progress) {
            if (controller.signal.aborted || currentSearchAbortController !== controller) return;
            latest = progress;
            schedulePaint();
        },
        async finish() {
            if (frame) cancelAnimationFrame(frame);
            frame = 0;
            if (linkInteractionInProgress) {
                await new Promise(resolve => {
                    const done = () => {
                        controller.signal.removeEventListener('abort', done);
                        window.removeEventListener('pointerup', done);
                        window.removeEventListener('pointercancel', done);
                        window.removeEventListener('keyup', done);
                        resolve();
                    };
                    controller.signal.addEventListener('abort', done, { once: true });
                    window.addEventListener('pointerup', done, { once: true });
                    window.addEventListener('pointercancel', done, { once: true });
                    window.addEventListener('keyup', done, { once: true });
                });
            }
            if (frame) cancelAnimationFrame(frame);
            frame = 0;
            removeInteractionListeners();
            if (ownsResults()) delete resultsDiv.dataset.progressiveSearch;
        },
        cancel() {
            if (frame) cancelAnimationFrame(frame);
            frame = 0;
            removeInteractionListeners();
            if (ownsResults() && previousNodes) {
                resultsDiv.replaceChildren(...previousNodes);
                resultsCount.textContent = previousCount;
                resultsInfo.style.display = previousInfoDisplay;
                delete resultsDiv.dataset.progressiveSearch;
            }
        }
    };
}


function resetSearchButton() {
    searchInProgress = false;
    currentSearchAbortController = null;
    document.getElementById('searchBtn').innerHTML = `
        <svg class="icon" aria-hidden="true"><use href="#icon-search"/></svg><span class="btn-text">Find Source</span>
    `;
    updateSearchButtonAvailability();
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
}

function displayResults(similarities, mostLikely) {
    const resultsDiv = document.getElementById('results');
    delete resultsDiv.dataset.progressiveSearch;
    resultsDiv.innerHTML = '';
    if (similarities.length === 0) {
        const empty = document.createElement('p');
        empty.textContent = 'No matching sources found. Try lowering the similarity threshold or searching more clusters.';
        resultsDiv.appendChild(empty);
    }

    if (mostLikely) 
    {
        const likelyDiv = document.createElement('div');
likelyDiv.className = 'most-likely highlight-pop';

const thumbnailSrc = `data:image/webp;base64,${mostLikely.thumbnail}`;

likelyDiv.innerHTML = `
    <div class="most-likely-header">
        <span aria-hidden="true">👑</span> Most Likely Match
    </div>
    <div class="most-likely-flex">
        <div class="most-likely-thumb-col">
            <img src="${escapeHtml(thumbnailSrc)}" alt="Thumbnail" class="most-likely-thumbnail-large">
        </div>
        <div class="most-likely-details-col">
            <div class="most-likely-detail-row"><span class="ml-label">Name:</span> <span class="ml-value">${escapeHtml(mostLikely.name)}</span></div>
            <div class="most-likely-detail-row"><span class="ml-label">Code:</span> <span class="ml-value">${mostLikely.code}</span></div>
            <div class="most-likely-detail-row">
                <span class="ml-label">Link:</span> 
                <span class="ml-value">
                    <a href="https://nhentai.net/g/${mostLikely.code}/" target="_blank" rel="noopener noreferrer">
                        www.nhentai.net/g/${mostLikely.code}
                    </a>
                </span>
            </div>
            <div class="most-likely-detail-row"><span class="ml-label">Tags:</span> <span class="ml-value">${escapeHtml(mostLikely.top_tags.join(', '))}</span></div>
            <div class="most-likely-detail-row"><span class="ml-label">Characters:</span> <span class="ml-value">${escapeHtml(mostLikely.top_chars.join(', '))}</span></div>
            <div class="most-likely-detail-row"><span class="ml-label">Similarity:</span> <span class="ml-value">${(mostLikely.similarity * 100).toFixed(1)}%</span></div>
        </div>
    </div>
`;

resultsDiv.appendChild(likelyDiv);
    }

    if (similarities.length > 0) {
        const alternativeResults = mostLikely ? similarities.slice(1) : similarities;
        if (alternativeResults.length === 0) {
            updateResultsInfo(similarities.length);
            return;
        }

        const expandableContainer = document.createElement('div');
        expandableContainer.className = 'expandable-results';
        const toggleBtn = document.createElement('button');
        toggleBtn.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#icon-list"/></svg><span>Show ${alternativeResults.length} more matches</span>`;
        toggleBtn.className = 'expand-btn';
        toggleBtn.setAttribute('aria-expanded', 'false');
        let expanded = false;
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'results-list';
        resultsContainer.style.display = 'none';

        alternativeResults.forEach((item, index) => resultsContainer.appendChild(createResultItem(item, index + (mostLikely ? 2 : 1))));

        toggleBtn.onclick = () => {
            expanded = !expanded;
            toggleBtn.setAttribute('aria-expanded', String(expanded));
            resultsContainer.style.display = expanded ? 'block' : 'none';
            toggleBtn.innerHTML = expanded
                ? '<svg class="icon" aria-hidden="true"><use href="#icon-chevron"/></svg><span>Hide additional matches</span>'
                : `<svg class="icon" aria-hidden="true"><use href="#icon-list"/></svg><span>Show ${alternativeResults.length} more matches</span>`;
        };

        expandableContainer.appendChild(toggleBtn);
        expandableContainer.appendChild(resultsContainer);
        resultsDiv.appendChild(expandableContainer);
    }

    updateResultsInfo(similarities.length);
}

function createResultItem(item, rank) {
    const resultItem = document.createElement('article');
    const score = item.similarity * 100;
    const tone = score >= 90 ? 'excellent' : score >= 80 ? 'strong' : 'close';
    const url = `https://nhentai.net/g/${item.code}/${item.page}`;
    resultItem.className = `result-item result-item--${tone}`;
    resultItem.innerHTML = `
        <div class="result-rank" aria-label="Match ${rank}">#${rank}</div>
        <div class="result-details">
            <div class="result-code">${item.code}</div>
            <div class="result-meta"><span>Page ${item.page}</span><span class="similarity">${score.toFixed(1)}% similar</span></div>
        </div>
        <a href="${url}" target="_blank" rel="noopener noreferrer" class="result-link" aria-label="Open source ${item.code}, page ${item.page}"><span>Open</span><svg class="icon" aria-hidden="true"><use href="#icon-external"/></svg></a>
    `;
    return resultItem;
}

function updateResultsInfo(count) {
    const resultsInfo = document.getElementById('resultsInfo');
    const resultsCount = document.getElementById('resultsCount');
    
    if (count > 0) {
        resultsCount.textContent = count;
        resultsInfo.style.display = 'flex';
    } else {
        resultsInfo.style.display = 'none';
    }
}

function updateStatus(message, type, boldPrefix = '') {
    const statusDiv = document.getElementById('status');
    if (message) {
        statusDiv.innerText = message;
        if (boldPrefix && message.startsWith(boldPrefix)) {
            const emphasis = document.createElement('strong');
            emphasis.textContent = boldPrefix;
            statusDiv.replaceChildren(emphasis, message.slice(boldPrefix.length));
        }
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
    } else {
        statusDiv.style.display = 'none';
    }
}
