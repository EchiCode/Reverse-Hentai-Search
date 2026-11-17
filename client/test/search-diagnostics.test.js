import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

test('explicit component timings print by name and the latest snapshot stays stable', () => {
    let now = 0;
    const messages = [];
    const context = vm.createContext({ performance: { now: () => now },
        console: { log: (...args) => messages.push(args) } });
    vm.runInContext(readFileSync(new URL('../search.js', import.meta.url), 'utf8') +
        '\nglobalThis.api = { createSearchDiagnostics, recordSearchDuration, finishSearchDiagnostics };', context);
    const api = context.api;
    const diagnostics = api.createSearchDiagnostics({}, 2);
    diagnostics.inferenceRuntime = { threads: 6 };
    api.recordSearchDuration(diagnostics, 'modelExecution', 33.5);
    assert.equal(messages[0][0], '[Search #2] modelExecution 33.50ms');
    now = 200;
    api.finishSearchDiagnostics(diagnostics, 'completed');
    assert.equal(context.lastSearchTiming.totalMs, 200);
    assert.equal(context.lastSearchTiming.stageMs.modelExecution, 33.5);
    assert.equal(context.lastSearchTiming.inferenceRuntime.threads, 6);
    diagnostics.stages.modelExecution = 99;
    diagnostics.inferenceRuntime.threads = 1;
    assert.equal(context.lastSearchTiming.stageMs.modelExecution, 33.5);
    assert.equal(context.lastSearchTiming.inferenceRuntime.threads, 6);
    api.finishSearchDiagnostics(api.createSearchDiagnostics({}, 1), 'cancelled');
    assert.equal(context.lastSearchTiming.searchId, 2);
});
