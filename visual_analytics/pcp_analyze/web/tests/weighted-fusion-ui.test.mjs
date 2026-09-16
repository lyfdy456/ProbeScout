import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function source(relativePath) {
  return readFile(new URL(`../${relativePath}`, import.meta.url), "utf8");
}

test("Weighted Fusion keeps the audited eight-learner and binary API contracts", async () => {
  const apiSource = await source("app/lib/weightedFusion.ts");
  for (const learner of [
    "MLP",
    "K-Fold",
    "Triplet Loss",
    "Attention Pooling",
    "Attribute-conditioned Attention",
    "nnPU",
    "DC-PU",
    "Ours-PURA",
  ]) {
    assert.match(apiSource, new RegExp(`"${learner.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"`));
  }
  assert.match(apiSource, /WEIGHTED_FUSION_API_PATH = "\/api\/tuning\/weighted-fusion"/);
  assert.match(apiSource, /WEIGHTED_FUSION_LAYOUT = \["raw", "calibrated", "rank"\]/);
  assert.match(apiSource, /body: JSON\.stringify\(\{ taskId, weights: normalizedWeights \}\)/);
  assert.match(apiSource, /buffer\.byteLength !== expectedBytes/);
  assert.match(apiSource, /generation !== generationRef\.current/);
});

test("weight edits are draft-only until Apply and zero remains an exclusion weight", async () => {
  const panelSource = await source("app/components/WeightedFusionPanel.tsx");
  assert.match(panelSource, /const WEIGHT_MIN = 0/);
  assert.match(panelSource, /0 excludes a learner/);
  assert.match(panelSource, /draftWeights: \{ \.\.\.scoped\.draftWeights, \[learner\]: value \}/);
  assert.match(panelSource, /void fusion\.apply\(normalizedDraft\)/);
  assert.match(panelSource, /Apply to Top/);
  assert.match(panelSource, /Reset equal/);
  assert.match(panelSource, /onRevert\(\)/);
  assert.match(panelSource, /appliedResult\?: WeightedFusionResult \| null/);
  assert.match(panelSource, /createEditorState\(scopeKey, appliedResult\)/);
  assert.match(panelSource, /appliedResult\?\.weights \?\? DEFAULT_WEIGHTED_FUSION_WEIGHTS/);
});

test("Dashboard keeps manual Weighted Fusion out of the active workbench", async () => {
  const dashboardSource = await source("app/Dashboard.tsx");
  assert.doesNotMatch(dashboardSource, /WeightedFusionPanel/);
  assert.doesNotMatch(dashboardSource, /WEIGHTED_FUSION_METHOD_ID/);
  assert.doesNotMatch(dashboardSource, /activeWeightedFusionResult|weightedFusionResult/);
  assert.doesNotMatch(dashboardSource, /Ours-Weighted/);
  assert.match(dashboardSource, /\.\.\.\(activeHierarchicalFusionResult \? \[HIERARCHICAL_FUSION_METHOD_ID\] : \[\]\)/);
  assert.match(dashboardSource, /buildHierarchicalPcpAxes/);
  assert.match(
    dashboardSource,
    /const runtimeSource = axis\.kind === "full" \|\| axis\.kind === "attribute"[\s\S]*?\? finalSource/,
  );
  assert.match(
    dashboardSource,
    /const runtimeSource[\s\S]*?: null;[\s\S]*?runtimeSource[\s\S]*?: source\[scoreOffset\(/,
    "baseline and learner axes must keep using their exported static columns",
  );
  assert.match(
    dashboardSource,
    /const oursFullIndex = methodIndex\.get\("Ours-Full"\)[\s\S]*?axis\.methodId[\s\S]*?: oursFullIndex/,
    "before a fusion is applied, total and attribute aggregates must keep their Ours-Full fallback",
  );
  assert.match(dashboardSource, /enabled: Boolean\(dataset\) && !hierarchicalFusionSelected/);
  assert.doesNotMatch(
    dashboardSource,
    /Each attribute has a 13-method rank simplex[\s\S]*?outer attribute weights/,
  );
});

test("the compatibility editor remains available without a Dashboard mount", async () => {
  const dashboardSource = await source("app/Dashboard.tsx");
  const panelSource = await source("app/components/WeightedFusionPanel.tsx");
  const revertBlock = panelSource.match(/const revert = \(\) => \{[\s\S]*?\n  \};/)?.[0];

  assert.ok(revertBlock, "Weighted Fusion Revert handler is missing");
  assert.doesNotMatch(dashboardSource, /weightedFusionFrozenTestAudit|handleWeightedFusionRevert/);
  assert.doesNotMatch(dashboardSource, /<WeightedFusionPanel/);
  assert.match(dashboardSource, /<TuningPanel[\s\S]*?state=\{tuning\}/);
  assert.match(revertBlock, /fusion\.clear\(\)/);
  assert.match(revertBlock, /appliedWeights: null/);
  assert.match(revertBlock, /onRevert\(\)/);
  assert.match(panelSource, /disabled=\{disabled \|\| busy \|\| !appliedWeights\}[\s\S]*?onClick=\{revert\}/);

  assert.match(panelSource, /testAudit\?: WeightedFusionTestAudit \| null/);
  assert.match(panelSource, /areWeightedFusionWeightsEquivalent\(fusion\.result\.weights, appliedWeights\)/);
  assert.match(panelSource, /!busy && !awaitingAppliedResult && appliedWeights && testAudit/);
  assert.match(panelSource, /FULL FROZEN TEST/);
  assert.match(panelSource, /Ours-Full → Weighted/);
  assert.match(panelSource, /<span>TP@\{k\}<\/span>/);
  assert.match(panelSource, /<strong>\{before\} → \{after\}<\/strong>/);
});
