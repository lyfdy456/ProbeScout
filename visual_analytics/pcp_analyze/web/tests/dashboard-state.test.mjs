import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  enabledMethodsAfterVirtualMethodRevert,
  isStaticValidationRankingCurrent,
  scopeActivitySummary,
} from "../app/lib/dashboardState.ts";

test("reverting a virtual method restores the real fallback when no axes remain", () => {
  const restored = enabledMethodsAfterVirtualMethodRevert(
    new Set(["Ours-Weighted"]),
    "Ours-Weighted",
    "Ours-Full",
  );
  assert.deepEqual([...restored], ["Ours-Full"]);

  const retained = enabledMethodsAfterVirtualMethodRevert(
    new Set(["MLP", "Ours-Weighted"]),
    "Ours-Weighted",
    "Ours-Full",
  );
  assert.deepEqual([...retained], ["MLP"]);
});

test("static Validation current state ignores an unrelated applied Weighted result", () => {
  assert.equal(isStaticValidationRankingCurrent("Ours-Full", "Ours-Full", false), true);
  assert.equal(isStaticValidationRankingCurrent("MLP", "Ours-Full", false), false);
  assert.equal(isStaticValidationRankingCurrent("Ours-Full", "Ours-Full", true), false);
});

test("scope header uses fixed Validation rows and post-filter counts elsewhere", () => {
  assert.deepEqual(scopeActivitySummary("validation", 9_554, 17), {
    count: 9_554,
    label: "comparison rows",
  });
  assert.deepEqual(scopeActivitySummary("development", 9_554, 17), {
    count: 17,
    label: "active",
  });
  assert.deepEqual(scopeActivitySummary("test", 9_554, 23), {
    count: 23,
    label: "active",
  });
});

test("Dashboard wires invariant helpers and clears Query preview across task lifecycle", async () => {
  const dashboard = await readFile(
    new URL("../app/Dashboard.tsx", import.meta.url),
    "utf8",
  );

  assert.doesNotMatch(dashboard, /enabledMethodsAfterVirtualMethodRevert|WEIGHTED_FUSION_METHOD_ID/);
  assert.match(
    dashboard,
    /isStaticValidationRankingCurrent\(\s*method,\s*rankMethod,\s*Boolean\(activeHierarchicalFusionResult\),?\s*\)/,
  );
  assert.match(dashboard, /scopeActivitySummary\([\s\S]*?dataset\.manifest\.evaluation\.validation\.rowCount,[\s\S]*?resultCandidateCount/);
  assert.match(dashboard, /headerActivity\.count\.toLocaleString\(\)[\s\S]*?headerActivity\.label/);
  assert.doesNotMatch(dashboard, /current: rankMethod === method[\s\S]{0,100}!activeWeightedFusionResult/);
  for (const accessibleName of [
    "PCP value",
    "PCP line mode",
    "PCP cluster scheme",
    "Projection",
    "Embedding cluster scheme",
    "Retrieval target",
    "Ranked by",
  ]) {
    assert.match(dashboard, new RegExp(`aria-label="${accessibleName}"`));
  }

  const loadSuccess = dashboard.slice(
    dashboard.indexOf("setDataset(loaded)"),
    dashboard.indexOf("setProjection(", dashboard.indexOf("setDataset(loaded)")),
  );
  assert.match(loadSuccess, /setQueryPreviewIndex\(null\)/);

  for (const handlerName of ["changeDataset", "changeTask", "reloadCurrentTask"]) {
    const start = dashboard.indexOf(`const ${handlerName}`);
    const end = dashboard.indexOf("\n  };", start);
    assert.notEqual(start, -1, `${handlerName} must exist`);
    assert.match(dashboard.slice(start, end), /setQueryPreviewIndex\(null\)/);
  }
});

test("Frozen Test keeps hierarchical weights read-only while showing an applied result", async () => {
  const dashboard = await readFile(
    new URL("../app/Dashboard.tsx", import.meta.url),
    "utf8",
  );

  assert.match(dashboard, /const hierarchicalWeightsReadOnly = testMode/);
  assert.match(
    dashboard,
    /const displayedHierarchicalConfig = hierarchicalWeightsReadOnly[\s\S]*?activeHierarchicalConfig[\s\S]*?: hierarchicalDraft/,
  );
  assert.match(
    dashboard,
    /weightsReadOnly=\{hierarchicalWeightsReadOnly\}/,
  );
  assert.match(
    dashboard,
    /Frozen Test shows the applied hierarchy; edit weights in Development\./,
  );
});
