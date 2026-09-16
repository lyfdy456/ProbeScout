import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  PROBE_UPDATE_ALGORITHM, compatibleProbeUpdates, resolveProbeSelection,
  probeUpdateCapability, createProbeUpdate, fetchProbeUpdates, fetchProbeUpdate,
  createTuningRun,
} from "../app/lib/tuningApi.ts";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const job = (id, overrides = {}) => ({
  id, taskId: "task/a", sessionId: "session/a", status: "succeeded",
  createdAt: "2026-09-05T14:00:00Z", snapshotId: `snapshot-${id}`,
  compatible: true, stale: false, annotationCount: 3, positiveCount: 2, negativeCount: 1,
  ...overrides,
});

test("Update Probes has its own exact capability without unlocking old combined buttons", () => {
  for (const value of [null, {}, { update_probes: { available: true } },
    { update_probes: { available: true, algorithmVersion: "old" } },
    { update_probes: { available: "true", algorithmVersion: PROBE_UPDATE_ALGORITHM } }]) {
    assert.equal(probeUpdateCapability(value).available, false);
  }
  assert.equal(probeUpdateCapability({ update_probes: { available: true, algorithmVersion: PROBE_UPDATE_ALGORITHM } }).available, true);
});

test("available snapshots are completed, compatible and scoped to the exact session/task", () => {
  const candidates = [job("old"), job("new", { createdAt: "2026-09-06T14:00:00Z" }),
    job("wrong-owner", { sessionId: "other" }), job("wrong-task", { taskId: "other" }),
    job("pending", { status: "running" }), job("failed", { status: "failed" }),
    job("unavailable", { compatible: false }), job("missing-artifact", { snapshotId: null })];
  assert.deepEqual(compatibleProbeUpdates(candidates, "session/a", "task/a").map((row) => row.id), ["new", "old"]);
});

test("selected snapshot is pinned, not replaced by a newer update; changed feedback is reusable", () => {
  const selected = job("chosen", { stale: true });
  const newer = job("newer", { createdAt: "2026-09-06T14:00:00Z" });
  assert.equal(resolveProbeSelection([newer, selected], "session/a", "task/a", "chosen"), selected);
  assert.equal(resolveProbeSelection([newer], "session/a", "task/a", "chosen"), null);
  assert.equal(resolveProbeSelection([newer], "session/a", "task/a", null), null);
});

test("standalone update and listing use private distinct endpoints; both Weight requests pin same job", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url, ...init });
    return { ok: true, json: async () => ({ probeUpdate: job("chosen"), probeUpdates: [job("chosen")], run: {} }) };
  };
  try {
    await createProbeUpdate("session/a");
    await fetchProbeUpdates("session/a");
    await fetchProbeUpdate("chosen");
    for (const mode of ["weight_staged", "weight_joint"]) await createTuningRun("session/a", {
      mode, baseMethod: "Ours-Full", probeSource: "updated", probeUpdateId: "chosen",
    });
    await createTuningRun("session/a", { mode: "weight_joint", baseMethod: "Ours-Full", probeSource: "original" });
    assert.equal(requests[0].url, "/api/tuning/sessions/session%2Fa/probe-updates");
    assert.equal(requests[0].method, "POST");
    assert.deepEqual(JSON.parse(requests[0].body), {});
    assert.equal(requests[1].cache, "no-store");
    assert.equal(requests[2].url, "/api/tuning/probe-updates/chosen");
    for (const request of requests) assert.equal(request.credentials, "same-origin");
    for (const request of requests.slice(3, 5)) {
      assert.equal(JSON.parse(request.body).probeUpdateId, "chosen");
      assert.equal(JSON.parse(request.body).probeSource, "updated");
    }
    assert.equal(JSON.parse(requests[5].body).probeUpdateId, undefined);
  } finally { globalThis.fetch = originalFetch; }
});

test("Update handler never launches weights, replaces a Tune card, or applies rankings", async () => {
  const hook = await source("app/lib/useTuningSession.ts");
  const update = hook.slice(hook.indexOf("const updateProbes = useCallback"), hook.indexOf("const applyRun = useCallback"));
  assert.match(update, /await waitForPendingAnnotations[\s\S]*?fetchRefinementCapabilities[\s\S]*?probeUpdateCapability[\s\S]*?createProbeUpdate/);
  assert.match(update, /probeUpdate\.sessionId !== sessionId \|\| probeUpdate\.taskId !== taskId/);
  assert.doesNotMatch(update, /createTuningRun|runTuning\(|setRun\(|setApplied|setSelectedProbeUpdateId|requestRefinement/);
  assert.match(update, /failureAttributionConfirmed/);
  assert.match(update, /validationReady/);
});

test("Weight handler rechecks the exact pinned snapshot and never refreshes Probes implicitly", async () => {
  const hook = await source("app/lib/useTuningSession.ts");
  const run = hook.slice(hook.indexOf("const runTuning = useCallback"), hook.indexOf("const updateProbes = useCallback"));
  assert.match(run, /const pinnedProbeUpdateId = selectedProbeUpdateId/);
  assert.match(run, /fetchProbeUpdate\(pinnedProbeUpdateId\)[\s\S]*?resolveProbeSelection\(\[probeUpdate\], sessionId, taskId, pinnedProbeUpdateId\)/);
  assert.match(run, /probeSource: pinnedProbeUpdateId \? "updated" : "original"/);
  assert.match(run, /probeUpdateId: pinnedProbeUpdateId/);
  assert.doesNotMatch(run, /createProbeUpdate|updateProbes\(|if \(probeUpdate.stale\)/);
});

test("polling and restored sources are session scoped; unavailable saved source stays visible", async () => {
  const hook = await source("app/lib/useTuningSession.ts");
  assert.match(hook, /pcp:probe-source:\$\{userId\}:\$\{sessionId\}:\$\{taskId\}/);
  assert.match(hook, /setSelectedProbeUpdateId\(savedId\)/);
  const polling = hook.slice(hook.indexOf("// Probe jobs are separate"), hook.indexOf("const selectProbeSource ="));
  assert.match(polling, /controller.signal.aborted \|\| !isActiveSessionOperation\(generation, sessionId\)/);
  assert.match(polling, /listRevision !== probeListRevisionRef.current/);
  assert.match(polling, /job.sessionId !== sessionId \|\| job.taskId !== taskId/);
  assert.doesNotMatch(polling, /setSelectedProbeUpdateId|setRun\(|setApplied/);
});

test("Refinement offers exactly three operations while advanced source controls retain stale status", async () => {
  const [actions, probeSource] = await Promise.all([
    source("app/components/TuningFunctionActions.tsx"),
    source("app/components/ProbeSourceControls.tsx"),
  ]);
  assert.equal((actions.match(/mode: "weight_(?:staged|joint)"/g) ?? []).length, 2);
  assert.equal((actions.match(/>\s*Update Probes\s*</g) ?? []).length, 1);
  assert.doesNotMatch(actions, /mode: "probe_(?:staged|joint)"/);
  assert.doesNotMatch(actions, /aria-label="Probe source"/);
  assert.match(probeSource, /aria-label="Probe source"/);
  assert.match(probeSource, /disabled=\{!availableUpdates.length\}/);
  assert.match(actions, /missingSnapshot/);
  assert.match(probeSource, /Feedback changed; update again if needed/);
  assert.match(probeSource, /Snapshot unavailable; choose a source/);
});
