import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  PROBE_JOINT_ALGORITHM,
  PROBE_STAGED_ALGORITHM,
  PROBE_JOINT_ALGORITHM_V2,
  PROBE_STAGED_ALGORITHM_V2,
  WEIGHT_JOINT_ALGORITHM_V4,
  WEIGHT_STAGED_ALGORITHM_V4,
  TUNING_LAUNCH_ALGORITHMS,
  fetchRefinementCapabilities,
  refinementCapability,
  tuningModeLabel,
  tuningRankingLabel,
} from "../app/lib/tuningApi.ts";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("training algorithm capabilities remain strict, including legacy run compatibility", () => {
  for (const [mode, algorithmVersion] of Object.entries(TUNING_LAUNCH_ALGORITHMS)) {
    for (const capabilities of [null, undefined, {}, { [mode]: { available: false } },
      { [mode]: { available: "true", algorithmVersion } },
      { [mode]: { available: true } }, { [mode]: { available: true, algorithmVersion: "wrong" } }]) {
      assert.equal(refinementCapability(capabilities, mode).available, false);
    }
    assert.deepEqual(refinementCapability({ [mode]: { available: true, algorithmVersion } }, mode), {
      available: true, algorithmVersion,
    });
  }
  assert.equal(refinementCapability({ probe_joint: {
    available: false, reason: "No isolated base bank", algorithmVersion: PROBE_JOINT_ALGORITHM,
  } }, "probe_joint").reason, "No isolated base bank");
  assert.equal(refinementCapability({ weight_joint: {
    available: true, algorithmVersion: TUNING_LAUNCH_ALGORITHMS.weight_joint,
  } }, "probe_joint").available, false);
});

test("native schedules describe frozen fusion and never rename an old probe run", () => {
  assert.equal(tuningModeLabel("weight_staged"), "Weight-only · Staged");
  assert.equal(tuningModeLabel("weight_joint"), "Weight-only · Joint");
  assert.equal(tuningModeLabel("probe_staged", PROBE_STAGED_ALGORITHM), "Native → Fusion · Staged");
  assert.equal(tuningModeLabel("probe_joint", PROBE_JOINT_ALGORITHM), "Native → Fusion · Joint");
  assert.equal(tuningRankingLabel("probe_joint", PROBE_JOINT_ALGORITHM), "Native → Fusion · Joint Tune");
  assert.equal(tuningModeLabel("probe_staged", PROBE_STAGED_ALGORITHM_V2), "Native → Fusion · Staged");
  assert.equal(tuningRankingLabel("probe_joint", PROBE_JOINT_ALGORITHM_V2), "Native → Fusion · Joint Tune");
  assert.equal(tuningModeLabel("probe_joint", "conjunction-holistic-probe-joint-v1"), "Legacy Probe-adaptive · Joint");
});

test("new training requires fixed-gate algorithms without relaunching historical versions", () => {
  assert.deepEqual(TUNING_LAUNCH_ALGORITHMS, {
    weight_staged: "conjunction-holistic-weight-staged-v5",
    weight_joint: "conjunction-holistic-weight-joint-v5",
    probe_staged: "conjunction-holistic-probe-staged-v3",
    probe_joint: "conjunction-holistic-probe-joint-v3",
  });
  for (const [mode, algorithmVersion] of [
    ["weight_staged", WEIGHT_STAGED_ALGORITHM_V4],
    ["weight_joint", WEIGHT_JOINT_ALGORITHM_V4],
    ["probe_staged", PROBE_STAGED_ALGORITHM_V2],
    ["probe_joint", PROBE_JOINT_ALGORITHM_V2],
  ]) {
    const capability = refinementCapability({ [mode]: { available: true, algorithmVersion } }, mode);
    assert.equal(capability.available, false);
    assert.match(capability.reason, /implementation mismatch/);
  }
});

test("fresh capabilities remain task-scoped, private and uncached", async () => {
  const previousFetch = globalThis.fetch;
  const controller = new AbortController();
  const capabilities = { probe_joint: { available: true, algorithmVersion: PROBE_JOINT_ALGORITHM } };
  try {
    globalThis.fetch = async (url, init) => {
      assert.equal(url, "/api/tuning/tasks/task%2Fa/refinement-capabilities");
      assert.equal(init.credentials, "same-origin");
      assert.equal(init.cache, "no-store");
      assert.equal(init.signal, controller.signal);
      return { ok: true, json: async () => ({ taskId: "task/a", refinementCapabilities: capabilities }) };
    };
    assert.deepEqual(await fetchRefinementCapabilities("task/a", controller.signal), capabilities);
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ taskId: "other", refinementCapabilities: capabilities }) });
    await assert.rejects(fetchRefinementCapabilities("task/a"), /different task/);
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ taskId: "task/a" }) });
    assert.equal(await fetchRefinementCapabilities("task/a"), null);
  } finally { globalThis.fetch = previousFetch; }
});

test("Run rechecks capabilities after writes and rejects unavailable or stale tasks before creation", async () => {
  const hook = await source("app/lib/useTuningSession.ts");
  const run = hook.slice(hook.indexOf("const runTuning = useCallback"), hook.indexOf("const applyRun = useCallback"));
  assert.match(hook, /setRefinementCapabilities\(payload\.refinementCapabilities \?\? null\)/);
  assert.match(hook, /setServiceState\("loading"\);\s*setRefinementCapabilities\(null\)/);
  assert.match(run, /await waitForPendingAnnotations[\s\S]*?await fetchRefinementCapabilities\(taskId\)[\s\S]*?if \(!isActiveSessionOperation\(generation, sessionId\)\) return;[\s\S]*?if \(!capability\.available\) throw new Error[\s\S]*?createTuningRun\(sessionId/);
  assert.match(run, /fetchRefinementCapabilities[\s\S]*?!feedbackContextRef\.current\.input\.validationReady/);
  assert.match(run, /confirmedAnnotationsRef\.current\.values\(\)[\s\S]*?failureAttributionConfirmed/);
  assert.doesNotMatch(run, /mode === "weight_staged".*mode === "weight_joint"/);
});

test("compact UI separates native update from both frozen weight schedules and retains legacy audit", async () => {
  const [actions, panel, dashboard, probeSource] = await Promise.all([
    source("app/components/TuningFunctionActions.tsx"), source("app/components/TuningPanel.tsx"), source("app/Dashboard.tsx"),
    source("app/components/ProbeSourceControls.tsx"),
  ]);
  assert.match(actions, /Update Probes/);
  assert.match(actions, /Staged Weight Refinement/);
  assert.match(actions, /Joint Weight Tune/);
  assert.match(actions, /state\.updateProbes\(\)/);
  assert.match(probeSource, /aria-label="Probe source"/);
  assert.doesNotMatch(actions, /mode: "probe_staged"|mode: "probe_joint"/);
  assert.match(actions, /!refinementCapability\(state\.refinementCapabilities, action\.mode\)\.available/);
  assert.doesNotMatch(actions, /enabled: true|enabled: false|更新 Probes｜联合训练/);
  assert.match(panel, /native-updated-frozen-probe-snapshot/);
  assert.match(panel, /probeState\.snapshotId/);
  assert.match(panel, /probeState\.sharedAcrossFusionSchedules/);
  assert.match(panel, /probeState\.frozenDuringFusion/);
  assert.match(panel, /Native Probe update → Frozen probes → Fusion/);
  assert.match(dashboard, /tuningRankingLabel\(tuning\.run\.mode, tuning\.run\.algorithmVersion\)/);
});
