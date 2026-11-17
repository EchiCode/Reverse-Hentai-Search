import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../search.js', import.meta.url), 'utf8');
const context = vm.createContext({
    AbortController, DecompressionStream, Response, TextDecoder, TransformStream,
    URL, console, performance, setTimeout, clearTimeout
});
vm.runInContext(`${source}\nglobalThis.searchTests = { mergeClusterResults, mapWithConcurrency, scoreSearchCluster };`, context);
const { mergeClusterResults, mapWithConcurrency, scoreSearchCluster } = context.searchTests;

function match(code, similarity) {
    return { code, page: 1, similarity };
}

test('partial merges preserve final centroid order for equal scores', () => {
    const slots = new Array(3);
    slots[2] = [match('last', 0.9), match('last-low', 0.8)];
    assert.deepEqual(Array.from(mergeClusterResults(slots), item => item.code), ['last', 'last-low']);

    slots[0] = [match('first', 0.9)];
    assert.deepEqual(Array.from(mergeClusterResults(slots), item => item.code), ['first', 'last', 'last-low']);

    slots[1] = [match('middle', 0.9)];
    assert.deepEqual(Array.from(mergeClusterResults(slots), item => item.code),
        ['first', 'middle', 'last', 'last-low']);
});

test('partial and final merges keep the same stable top-50 cutoff', () => {
    const slots = Array.from({ length: 6 }, (_, cluster) =>
        Array.from({ length: 12 }, (_, item) => match(`${cluster}-${item}`, 0.75)));
    const expected = slots.flat().slice(0, 50).map(item => item.code);
    assert.deepEqual(Array.from(mergeClusterResults(slots), item => item.code), expected);

    const outOfOrder = new Array(6);
    for (const index of [5, 2, 4, 0, 3, 1]) outOfOrder[index] = slots[index];
    assert.deepEqual(Array.from(mergeClusterResults(outOfOrder), item => item.code), expected);
});

test('concurrent mapping reports each original input slot', async () => {
    const controller = new AbortController();
    const seen = [];
    const values = await mapWithConcurrency(['a', 'b', 'c', 'd'], 3, async (value, signal, index) => {
        signal.throwIfAborted();
        await new Promise(resolve => setTimeout(resolve, value === 'a' ? 5 : 0));
        seen.push(index);
        return `${index}:${value}`;
    }, controller.signal);
    assert.deepEqual(Array.from(values), ['0:a', '1:b', '2:c', '3:d']);
    assert.deepEqual([...seen].sort((a, b) => a - b), [0, 1, 2, 3]);
});


test('duplicate posting records merge to one result per image', () => {
    const first = { ...match(100, 0.9), clusterId: 0 };
    const duplicate = { ...match(100, 0.9), clusterId: 1 };
    const results = mergeClusterResults([[first], [duplicate, match(200, 0.8)], [duplicate]]);
    assert.deepEqual(Array.from(results, item => item.code), [100, 200]);
    assert.equal(results[0].clusterId, 0);
});

test('duplicate record merging keeps the highest score and the first equal score', () => {
    const results = mergeClusterResults([
        [{ ...match(100, 0.7), clusterId: 0 }],
        [{ ...match(100, 0.95), clusterId: 1 }],
        [{ ...match(100, 0.95), clusterId: 2 }],
    ]);
    assert.equal(results.length, 1);
    assert.equal(results[0].similarity, 0.95);
    assert.equal(results[0].clusterId, 1);
});

test('duplicate record merging keeps different pages from the same gallery', () => {
    const results = mergeClusterResults([
        [match(100, 0.9)],
        [{ ...match(100, 0.8), page: 2 }, match(100, 0.9)],
    ]);
    assert.deepEqual(Array.from(results, item => `${item.code}:${item.page}`), ['100:1', '100:2']);
});

test('duplicate records cannot crowd unique images out of the top fifty', () => {
    const common = Array.from({ length: 20 }, (_, item) => match(`common-${item}`, 0.99));
    const slots = Array.from({ length: 5 }, (_, cluster) => [
        ...common,
        ...Array.from({ length: 10 }, (_, item) => match(`${cluster}-${item}`, 0.9)),
    ]);
    const results = Array.from(mergeClusterResults(slots), item => item.code);
    assert.equal(results.length, 50);
    assert.equal(new Set(results).size, 50);
    assert.deepEqual(results, [
        ...common.map(item => item.code),
        ...Array.from({ length: 3 }, (_, cluster) =>
            Array.from({ length: 10 }, (_, item) => `${cluster}-${item}`)).flat(),
    ]);
});

test('duplicate records inside one cluster cannot crowd out distinct qualifying images', () => {
    const cluster = {
        count: 52, vectorLength: 1,
        codes: new Int32Array(52).fill(100),
        pages: new Int16Array(52).fill(1),
        vectors: new Float32Array(52).fill(1),
        inverseNorms: new Float32Array(52).fill(1),
    };
    cluster.pages[50] = 2;
    cluster.codes[51] = 200;
    cluster.vectors[51] = 0.9;
    const results = mergeClusterResults([scoreSearchCluster([1], cluster, 0.8)]);
    assert.deepEqual(Array.from(results, item => `${item.code}:${item.page}`), ['100:1', '100:2', '200:1']);
});

test('cluster scoring keeps the highest duplicate score and its first equal record', () => {
    const cluster = {
        count: 4, vectorLength: 1,
        codes: new Int32Array([100, 200, 100, 100]),
        pages: new Int16Array(4).fill(1),
        vectors: new Float32Array([0.8, 0.95, 0.95, 0.95]),
        inverseNorms: new Float32Array(4).fill(1),
    };
    const results = scoreSearchCluster([1], cluster, 0.7);
    assert.deepEqual(Array.from(results, item => item.index), [1, 2]);
});
