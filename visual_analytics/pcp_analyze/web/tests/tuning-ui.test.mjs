import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  HIERARCHICAL_PCP_LEARNERS,
  HIERARCHICAL_PCP_METHODS,
  hierarchicalPcpConfigFromTuningWeights,
} from "../app/lib/hierarchicalPcp.ts";

async function source(relativePath) {
  return readFile(new URL(`../${relativePath}`, import.meta.url), "utf8");
}

test("gallery feedback exposes the complete five-level label contract", async () => {
  const [labelsSource, gallerySource] = await Promise.all([
    source("app/lib/feedbackLabels.ts"),
    source("app/components/TopGallery.tsx"),
  ]);

  const labels = [
    ["strong-positive", "2"],
    ["positive", "1"],
    ["uncertain", "0"],
    ["negative", "-1"],
    ["strong-negative", "-2"],
  ];
  for (const [label, numeric] of labels) {
    assert.match(
      labelsSource,
      new RegExp(`value:\\s*"${label}"[\\s\\S]*?numeric:\\s*${numeric}(?:,|\\s)`),
      `${label} must retain its audited numeric training value`,
    );
  }
  assert.match(labelsSource, /"unmarked"/);
  assert.match(gallerySource, /FEEDBACK_LABELS\.map\(\(option\) =>/);
  assert.match(
    gallerySource,
    /value === option\.value \? "unmarked" : option\.value/,
    "clicking the active label must remain a reversible clear action",
  );
});

test("only Development Gallery rows can be annotated", async () => {
  const [dashboardSource, gallerySource] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TopGallery.tsx"),
  ]);

  assert.match(
    dashboardSource,
    /const canAnnotate = useCallback\([\s\S]*?developmentMode[\s\S]*?dataset\.developmentMask\[item\.rowIndex\] === 1[\s\S]*?\);/,
  );
  assert.match(
    dashboardSource,
    /const updatePreference = useCallback\([\s\S]*?if \(!canAnnotate\(item\)\) return false;/,
    "the state mutation must repeat the row-level guard instead of trusting disabled controls",
  );
  assert.match(
    dashboardSource,
    /<TopGallery[\s\S]*?canAnnotate=\{canAnnotate\}[\s\S]*?onPreferenceChange=\{updatePreference\}/,
  );
  assert.doesNotMatch(
    dashboardSource,
    /onPreferenceChange=\{resultScope === "test"/,
    "the row-level Development guard must own feedback locking",
  );
  assert.match(
    gallerySource,
    /canAnnotate=\{canAnnotate\?\.\(item\) \?\? Boolean\(onPreferenceChange\)\}/,
  );
  assert.match(gallerySource, /Feedback unavailable/i);
});

test("Dashboard applies personal tuned outputs to Top, selection ranges, and dynamic rank-profile clusters", async () => {
  const dashboardSource = await source("app/Dashboard.tsx");

  assert.match(dashboardSource, /import \{ useTuningSession \} from "\.\/lib\/useTuningSession"/);
  assert.match(
    dashboardSource,
    /const tuning = useTuningSession\(\{[\s\S]*?taskId: selectedTaskId,[\s\S]*?targetId: retrievalTarget,[\s\S]*?baseMethod: rankMethod,[\s\S]*?rowCount:/,
  );
  assert.match(
    dashboardSource,
    /enabled: Boolean\(dataset\) && !hierarchicalFusionSelected/,
    "an applied personal run must survive a data-purpose change for final Test audit",
  );
  assert.match(
    dashboardSource,
    /const activeTunedOutputMatches = Boolean\([\s\S]*?tuning\.appliedRanks[\s\S]*?tuning\.appliedScores[\s\S]*?tuning\.appliedRunId === tuning\.run\.id[\s\S]*?tuning\.session\?\.taskId === selectedTaskId[\s\S]*?tuning\.session\.targetId === retrievalTarget[\s\S]*?tuning\.session\.baseMethod === rankMethod/,
  );
  assert.match(
    dashboardSource,
    /const getRank = useCallback\([\s\S]*?if \(activeRuntimeFusionResult\)[\s\S]*?activeRuntimeFusionResult\.ranks\[[\s\S]*?if \(activeTunedRanks\) return activeTunedRanks\[item\.rowIndex\];[\s\S]*?: dataset\.ranks\[/,
    "Top must prefer a complete Fusion cube, retain target-only Residual support, then fall back to the immutable baseline",
  );
  assert.match(dashboardSource, /<TopGallery[\s\S]*?getRank=\{getRank\}/);
  assert.match(
    dashboardSource,
    /const selectionRuleAttributes[\s\S]*?activeRuntimeFusionResult\.ranks[\s\S]*?activeRuntimeFusionResult\.calibratedScores[\s\S]*?activeTunedRanks[\s\S]*?dataset\.ranks[\s\S]*?activeTunedCalibratedScores[\s\S]*?dataset\.calibratedScores[\s\S]*?summarizeSelectionRules/,
    "selection ranges must pair rank and calibrated score from the complete tuned cube or active target artifact",
  );
  assert.match(dashboardSource, /buildSmartFilterMask\([\s\S]*?ranks: dataset\.ranks/);
  assert.match(
    dashboardSource,
    /import \{[\s\S]*?useRuntimeRankProfileClusters[\s\S]*?\} from "\.\/lib\/runtimeRankProfileClusters"/,
    "rank-profile clustering must have a dedicated runtime lifecycle instead of mutating exported data",
  );
  assert.match(
    dashboardSource,
    /const runtimeRankClusters = useRuntimeRankProfileClusters\(\{[\s\S]*?taskId: selectedTaskId,[\s\S]*?scheme: rankClusterScheme/,
    "the selected rank-cluster scheme must address the matching runtime result",
  );
  assert.match(
    dashboardSource,
    /activeRuntimeRankClusters\?\.labels[\s\S]*?dataset\.clusterLabels\.get\(rankClusterScheme\)/,
    "runtime labels must override, and safely fall back to, the exported static labels",
  );
  const summaryStart = dashboardSource.indexOf("const rankClusterSummaries");
  const summaryEnd = dashboardSource.indexOf("const displayedHierarchicalConfig", summaryStart);
  const summaryBlock = dashboardSource.slice(summaryStart, summaryEnd);
  assert.match(summaryBlock, /activeRuntimeRankClusters/);
  assert.match(
    summaryBlock,
    /dataset\.metadata\.clusterSummary\.filter/,
    "runtime summary identities must be derived from active labels, with exported summaries as baseline fallback",
  );
  const structuralClusterPipeline = dashboardSource.slice(
    dashboardSource.indexOf("const rankStructuralMask"),
    dashboardSource.indexOf("const pcpAxes"),
  );
  assert.match(
    structuralClusterPipeline,
    /rankClusterLabels/,
    "cluster filtering must consume the active dynamic-or-static label vector",
  );
  assert.match(
    dashboardSource,
    /<ParallelCoordinates[\s\S]*?labels=\{rankClustersReady \? rankClusterLabels : undefined\}/,
    "sample colors must use the active runtime labels only after their matching result is ready",
  );
  assert.match(
    dashboardSource,
    /<ClusterSummaryParallelCoordinates[\s\S]*?labels=\{rankClusterLabels\}[\s\S]*?activeClusters=\{rankClusterSummaries\.map/,
    "sample colors, centroid profiles, filtering, and drill-down must share one active rank-cluster identity",
  );
  const diagnosticPipeline = dashboardSource.slice(
    dashboardSource.indexOf("const smartFilterResult"),
    dashboardSource.indexOf("const resultMask"),
  );
  assert.match(diagnosticPipeline, /liveValues: diagnosticSource\.values \?\? undefined/);
  assert.match(diagnosticPipeline, /if \(diagnosticSource\.error\) return null/);
  const diagnosticSourcePipeline = dashboardSource.slice(
    dashboardSource.indexOf("const diagnosticActive"),
    dashboardSource.indexOf("const smartFilterResult"),
  );
  assert.match(diagnosticSourcePipeline, /buildSnapshotDiagnosticValues\([\s\S]*?activeRefinementVisualization, retrievalTarget, smartFilterLearners/);
  assert.match(diagnosticSourcePipeline, /values\.fusionRanks = activeTunedRanks/);
  assert.match(diagnosticSourcePipeline, /values\.comparisonRanks = activeTunedRanks/);
  assert.doesNotMatch(diagnosticSourcePipeline, /tuning\.run\?\.after|groundTruth|testMask/);
  assert.match(diagnosticPipeline, /rawScores: dataset\.rawScores/);
  assert.match(diagnosticPipeline, /ranks: dataset\.ranks/);
  const projectionPipeline = dashboardSource.slice(
    dashboardSource.indexOf("const projectionPoints"),
    dashboardSource.indexOf("const galleryItems"),
  );
  assert.doesNotMatch(
    projectionPipeline,
    /activeTuned|appliedRanks|appliedScores/,
    "published projection coordinates must remain independent of personal tuning",
  );
  assert.match(
    dashboardSource,
    /const attributeStrengthSourceLabel[\s\S]*?refinementSourceLabel[\s\S]*?activeFusionTuneResult && activeTunedRanking[\s\S]*?all attributes recomputed[\s\S]*?other attributes use \$\{rankMethodLabel\}/,
    "attribute strength must identify the complete snapshot or active-target score source",
  );
  assert.match(dashboardSource, /attributeStrengthSourceLabel=\{attributeStrengthSourceLabel\}/);
  assert.match(dashboardSource, /<SelectionRuleSummaryPanel[\s\S]*?result=\{selectionRuleSummary\}/);
  assert.doesNotMatch(dashboardSource, /<RuleBuilder/);
});

test("Weight Tune Apply switches PCP curves and rank clusters to the same immutable run snapshot as Top", async () => {
  const dashboardSource = await source("app/Dashboard.tsx");

  const activationStart = dashboardSource.indexOf("const activeTunedOutputMatches");
  const preferencesStart = dashboardSource.indexOf("const preferences", activationStart);
  assert.ok(activationStart >= 0 && preferencesStart > activationStart);
  const activationBlock = dashboardSource.slice(activationStart, preferencesStart);
  assert.match(
    activationBlock,
    /const activePersonalRefinementVisualization = activeTunedOutputMatches[\s\S]*?isWeightRefinementRun\(tuning\.run\)[\s\S]*?tuning\.appliedVisualization\?\.runId === tuning\.run\.id[\s\S]*?componentCount === refinementComponentIds\.length/,
    "a stale, wrong-run, wrong-row, or wrong-component snapshot must never reach PCP",
  );
  assert.match(
    activationBlock,
    /const refinementRankClusters = useRefinementClusters\(\{[\s\S]*?run: activePersonalRefinementVisualization \? tuning\.run : null,[\s\S]*?snapshot: activePersonalRefinementVisualization,[\s\S]*?scheme: rankClusterScheme/,
  );
  assert.match(
    activationBlock,
    /refinementRankClusters\.data\.runId === tuning\.run\.id[\s\S]*?refinementRankClusters\.data\.scheme === rankClusterScheme[\s\S]*?sourceFingerprint[\s\S]*?activePersonalRefinementVisualization\.sourceFingerprint/,
    "rank labels must be from this exact run, score artifact, and cluster scheme",
  );

  const labelsStart = activationBlock.indexOf("const rankClusterLabels");
  const labelsEnd = activationBlock.indexOf("const rankClusterSourceKey", labelsStart);
  const labelsBlock = activationBlock.slice(labelsStart, labelsEnd);
  const refinementAt = labelsBlock.indexOf("if (activeRefinementVisualization)");
  const legacyAt = labelsBlock.indexOf("if (activeRuntimeFusionResult)");
  const staticAt = labelsBlock.indexOf("dataset.clusterLabels.get");
  assert.ok(
    refinementAt >= 0 && legacyAt > refinementAt && staticAt > legacyAt,
    "Weight Tune labels must take precedence over legacy runtime and static Ours-Full clusters",
  );
  assert.match(activationBlock, /activeRefinementVisualization[\s\S]*?refinementRankClusters\.sourceKey/);

  const axesStart = dashboardSource.indexOf("const pcpAxes");
  const valuesStart = dashboardSource.indexOf("const pcpValues", axesStart);
  const zoomStart = dashboardSource.indexOf("const pcpZoomActive", valuesStart);
  assert.ok(axesStart >= 0 && valuesStart > axesStart && zoomStart > valuesStart);
  const axesBlock = dashboardSource.slice(axesStart, valuesStart);
  const valuesBlock = dashboardSource.slice(valuesStart, zoomStart);
  assert.match(
    axesBlock,
    /if \(activeRefinementVisualization\)[\s\S]*?buildRefinementPcpAxes\(\{/,
    "the applied score-level formula needs its own hierarchy instead of the old 13-way rank fusion",
  );
  assert.match(valuesBlock, /pcpValueKind === "rank"[\s\S]*?activeRefinementVisualization\.ranks/);
  assert.match(valuesBlock, /activeRefinementVisualization\.scores/);
  assert.match(
    valuesBlock,
    /refinementComponentIndex\.get\(pcpAxes\[axisIndex\]\.id\)[\s\S]*?rowIndex \* activeRefinementVisualization\.componentCount \+ componentIndex/,
    "every visible PCP curve must address a column of the exact applied snapshot",
  );

  const applyStart = dashboardSource.indexOf("const applyTunedRanking");
  const revertStart = dashboardSource.indexOf("const revertTunedRanking", applyStart);
  const applyBlock = dashboardSource.slice(applyStart, revertStart);
  assert.match(
    applyBlock,
    /await tuning\.applyRun\(\{[\s\S]*?refinementClusterScheme: isWeightRefinementRun\(run\)[\s\S]*?\? rankClusterScheme[\s\S]*?: undefined/,
    "the visible Apply action must preflight clusters for the currently selected scheme",
  );

  const schemeStart = dashboardSource.indexOf("const changeRankClusterScheme");
  const schemeEnd = dashboardSource.indexOf("const changeVisualClusterScheme", schemeStart);
  const schemeBlock = dashboardSource.slice(schemeStart, schemeEnd);
  const refinementRequestAt = schemeBlock.indexOf("await requestRefinementClusters");
  const schemeCommitAt = schemeBlock.indexOf("setRankClusterScheme(scheme)", refinementRequestAt);
  assert.ok(
    refinementRequestAt >= 0 && schemeCommitAt > refinementRequestAt,
    "switching K/scheme must load matching tuned labels before committing the selector",
  );
  assert.match(schemeBlock, /snapshot: activePersonalRefinementVisualization/);

  const topStart = dashboardSource.indexOf("const getRank = useCallback");
  const topEnd = dashboardSource.indexOf("const getScoreReadout", topStart);
  const topBlock = dashboardSource.slice(topStart, topEnd);
  assert.match(topBlock, /if \(activeTunedRanks\) return activeTunedRanks\[item\.rowIndex\]/);
  assert.match(
    dashboardSource,
    /<ClusterSummaryParallelCoordinates[\s\S]*?values=\{pcpValues\}[\s\S]*?labels=\{rankClusterLabels\}/,
    "cluster centroids and sample PCP must share the tuned values and tuned labels",
  );
  assert.match(
    dashboardSource,
    /<HierarchicalPcpRail[\s\S]*?refinementWeights=\{activeRefinementWeights\}/,
    "the visible weight rail must identify beta, gamma, eta, and lambda from the applied run",
  );
});

test("dynamic rank-profile cluster identity is independent of PCP presentation state", async () => {
  const dashboardSource = await source("app/Dashboard.tsx");
  const hookStart = dashboardSource.indexOf("const runtimeRankClusters = useRuntimeRankProfileClusters");
  const labelsStart = dashboardSource.indexOf("const rankClusterLabels", hookStart);
  assert.ok(hookStart >= 0 && labelsStart > hookStart, "Dashboard must create runtime clusters before resolving labels");
  const hookBlock = dashboardSource.slice(hookStart, labelsStart);

  assert.match(hookBlock, /taskId: selectedTaskId/);
  assert.match(hookBlock, /scheme: rankClusterScheme/);
  assert.doesNotMatch(
    hookBlock,
    /pcpValueKind|pcpFullExpanded|expandedPcpAttributes|resultScope|pcpAxes|pcpValues/,
    "expand/collapse, Value, visible axes, and data-purpose scope must never redefine the complete rank feature space",
  );

  const invalidationStart = dashboardSource.indexOf("const previousSource = rankClusterInteractionSourceRef.current");
  const invalidationEnd = dashboardSource.indexOf("const zoomCurrentPcpBrush", invalidationStart);
  const invalidationBlock = dashboardSource.slice(invalidationStart, invalidationEnd);
  assert.match(
    invalidationBlock,
    /rankClusterSourceKey[\s\S]*clearPcpSelection\(\)/,
    "a newly clustered ranking must invalidate an old PCP brush even when its scheme is unchanged",
  );

  const schemeHandlerStart = dashboardSource.indexOf("const changeRankClusterScheme");
  const schemeHandlerEnd = dashboardSource.indexOf("const changeVisualClusterScheme", schemeHandlerStart);
  const schemeHandler = dashboardSource.slice(schemeHandlerStart, schemeHandlerEnd);
  assert.match(schemeHandler, /setRankClusterScheme\(scheme\)/);
  assert.match(schemeHandler, /setRankClusterFilter\("all"\)|clearRankClusterInteraction\(\)/);
  assert.match(schemeHandler, /setClusterDrilldown\(null\)|clearRankClusterInteraction\(\)/);
  assert.match(schemeHandler, /clearPcpSelection\(\)|clearRankClusterInteraction\(\)/);
});

test("Fusion Weight v4 summaries become complete normalized PCP weight configs", () => {
  const attributeIds = ["cat", "hugging"];
  const learnedWeights = {
    cat: Object.fromEntries(
      HIERARCHICAL_PCP_METHODS.map((method, index) => [method, index + 1]),
    ),
    hugging: Object.fromEntries(
      HIERARCHICAL_PCP_METHODS.map((method) => [method, 2]),
    ),
  };

  const config = hierarchicalPcpConfigFromTuningWeights(learnedWeights, attributeIds);
  assert.deepEqual(config.attributeWeights, { cat: 0.5, hugging: 0.5 });
  const weightedConfig = hierarchicalPcpConfigFromTuningWeights(
    learnedWeights,
    attributeIds,
    { cat: 0.5, hugging: 1.5 },
  );
  assert.deepEqual(weightedConfig.attributeWeights, { cat: 0.25, hugging: 0.75 });
  const catTotal = HIERARCHICAL_PCP_METHODS.reduce(
    (sum, _method, index) => sum + index + 1,
    0,
  );
  for (const [index, method] of HIERARCHICAL_PCP_METHODS.entries()) {
    assert.ok(
      Math.abs(config.methodWeightsByAttribute.cat[method] - (index + 1) / catTotal) < 1e-12,
      `${method} must preserve its learned cat contribution after normalization`,
    );
    assert.equal(config.methodWeightsByAttribute.hugging[method], 1 / 13);
  }
  assert.throws(
    () => hierarchicalPcpConfigFromTuningWeights({ cat: learnedWeights.cat }, attributeIds),
    /hugging|missing/i,
    "a partial tuning summary must not silently leave a PCP attribute at stale weights",
  );
  assert.throws(
    () => hierarchicalPcpConfigFromTuningWeights(
      learnedWeights,
      attributeIds,
      { cat: 1 },
    ),
    /attribute weights|hugging|missing/i,
  );
  assert.throws(
    () => hierarchicalPcpConfigFromTuningWeights(
      {
        ...learnedWeights,
        cat: Object.fromEntries(HIERARCHICAL_PCP_LEARNERS.map((method) => [method, 1])),
      },
      attributeIds,
    ),
    /Image Prototype|exactly|method weights/i,
    "an eight-learner historical matrix must never invent five baseline weights",
  );
});

test("Fusion Weight v4 exposes one peer 13-method rank simplex per attribute", async () => {
  const [apiSource, panelSource] = await Promise.all([
    source("app/lib/tuningApi.ts"),
    source("app/components/TuningPanel.tsx"),
  ]);

  assert.equal(HIERARCHICAL_PCP_METHODS.length, 13);
  assert.deepEqual(
    HIERARCHICAL_PCP_METHODS.slice(0, 5),
    [
      "Image Prototype",
      "Query MaxSim",
      "Image--Text Fusion",
      "Text Prompt Ensemble",
      "Z-score Image--Text Fusion",
    ],
    "the five canonical embedding baselines must precede the eight learning methods",
  );
  assert.deepEqual(HIERARCHICAL_PCP_METHODS.slice(5), HIERARCHICAL_PCP_LEARNERS);

  assert.match(apiSource, /fusion-weight-hierarchical-rank-13-v4/);
  assert.match(
    apiSource,
    /methodWeightsByAttribute\?: Record<string, Record<string, number>>/,
  );
  assert.match(apiSource, /methodWeightCount\?: number/);
  assert.match(apiSource, /optimizedMethodWeightCount\?: number/);
  const v4PredicateStart = apiSource.indexOf("export function isHierarchicalRankFusionRun");
  const legacyPredicateStart = apiSource.indexOf("export function isPerAttributeFusionRun");
  const predicateEnd = apiSource.indexOf("export interface TuningBootstrap", legacyPredicateStart);
  const v4PredicateSource = apiSource.slice(v4PredicateStart, legacyPredicateStart);
  const legacyPredicateSource = apiSource.slice(legacyPredicateStart, predicateEnd);
  assert.match(v4PredicateSource, /HIERARCHICAL_RANK_FUSION_ALGORITHM/);
  assert.match(v4PredicateSource, /modelSummary\?\.methodWeightsByAttribute/);
  assert.match(legacyPredicateSource, /isHierarchicalRankFusionRun\(run\)/);
  assert.match(legacyPredicateSource, /PER_ATTRIBUTE_FUSION_ALGORITHM_V3/);
  assert.match(legacyPredicateSource, /PER_ATTRIBUTE_FUSION_ALGORITHM_V2/);
  assert.match(legacyPredicateSource, /modelSummary\?\.learnerWeightsByAttribute/);

  assert.match(panelSource, /HIERARCHICAL_PCP_METHODS/);
  assert.match(panelSource, /modelSummary\?\.methodWeightsByAttribute/);
  assert.match(panelSource, /Learner weights by attribute/);
  assert.match(
    panelSource,
    /HIERARCHICAL_PCP_METHODS[\s\S]*?weights\[name\][\s\S]*?toFixed\(1\)/,
    "the v4 matrix must render in the shared fixed 5+8 order",
  );
  assert.match(panelSource, /Attribute weights/);
  assert.match(
    panelSource,
    /methodWeightsByAttribute[\s\S]*?\(value \* 100\)\.toFixed\(1\)/,
    "v4 outer attribute weights must be displayed as percentages",
  );
  assert.match(
    panelSource,
    /methodWeightsByAttribute \|\| displayedLearnerWeights[\s\S]*?Apply weights & ranking/,
  );
  assert.match(panelSource, /className="tuning-run-details"/);
});

test("Fusion Weight Apply atomically updates full ranks, PCP curves, and the visible weight rail", async () => {
  const [dashboardSource, panelSource] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TuningPanel.tsx"),
  ]);

  const applyStart = dashboardSource.indexOf("const applyTunedRanking");
  const revertStart = dashboardSource.indexOf("const revertTunedRanking");
  assert.ok(applyStart >= 0, "Dashboard must own the cross-view tuned-apply transaction");
  assert.ok(revertStart > applyStart, "the tuned revert handler must follow the apply handler");
  const applyBlock = dashboardSource.slice(applyStart, revertStart);
  assert.match(applyBlock, /isHierarchicalRankFusionRun\(run\)/);
  assert.match(applyBlock, /modelSummary!?\.methodWeightsByAttribute/);
  assert.match(applyBlock, /modelSummary!?\.attributeWeights/);
  assert.match(applyBlock, /hierarchicalPcpConfigFromTuningWeights/);
  const configAt = applyBlock.indexOf("hierarchicalPcpConfigFromTuningWeights");
  const hierarchyApplyAt = applyBlock.indexOf("fusionTuneHierarchy.apply");
  const rankReclusterAt = applyBlock.indexOf("await prepareRuntimeRankClusters");
  const targetArtifactClearAt = applyBlock.indexOf("tuning.revertRun");
  const draftCommitAt = applyBlock.lastIndexOf("setHierarchicalDraft");
  const metadataCommitAt = applyBlock.indexOf("setAppliedFusionTune");
  assert.ok(hierarchyApplyAt >= 0, "Fusion Weight Apply must request the full target matrix");
  assert.ok(
    hierarchyApplyAt > configAt,
    "the learned matrix must be validated before requesting recomputed rankings",
  );
  assert.ok(
    rankReclusterAt > hierarchyApplyAt,
    "the complete recomputed rank cube must exist before fitting its Development rank-profile clusters",
  );
  const rankReclusterBlock = applyBlock.slice(rankReclusterAt, targetArtifactClearAt);
  assert.match(rankReclusterBlock, /await prepareRuntimeRankClusters\(result\.config, rankClusterScheme\)/);
  assert.match(
    rankReclusterBlock,
    /rankClusterScheme/,
    "Apply must preflight the currently selected fine, absolute, or shape scheme",
  );
  assert.match(
    rankReclusterBlock,
    /result\.config/,
    "dynamic rank clustering must address the same complete recomputed Fusion configuration",
  );
  assert.doesNotMatch(
    rankReclusterBlock,
    /groundTruth|pcpValueKind|pcpAxes|pcpValues|expandedPcpAttributes|resultScope/,
    "rank reclustering must be model-only and independent of the visible PCP presentation",
  );
  assert.ok(
    targetArtifactClearAt > rankReclusterAt,
    "both the complete cube and its current-scheme clusters must succeed before a stale artifact is cleared",
  );
  assert.ok(
    draftCommitAt > targetArtifactClearAt && metadataCommitAt > draftCommitAt,
    "the PCP weights and active ranking metadata must commit only after the full cube succeeds",
  );
  const applyFailureBlock = applyBlock.slice(applyBlock.indexOf("catch (reason)"));
  assert.doesNotMatch(
    applyFailureBlock,
    /setHierarchicalDraft|setAppliedFusionTune/,
    "a failed full-cube or reclustering request must not partially commit ranks, weights, or labels",
  );
  assert.match(applyBlock, /setRankClusterFilter\("all"\)|(?:reset|clear)RankCluster/);
  assert.match(applyBlock, /setClusterDrilldown\(null\)|(?:reset|clear)RankCluster/);

  const preflightStart = dashboardSource.indexOf("const prepareRuntimeRankClusters");
  const preflightEnd = dashboardSource.indexOf("const changePcpFullExpanded", preflightStart);
  const preflightBlock = dashboardSource.slice(preflightStart, preflightEnd);
  assert.match(preflightBlock, /requestRuntimeRankProfileClusters\(\{/);
  assert.match(preflightBlock, /taskId: selectedTaskId/);
  assert.match(preflightBlock, /rowCount: dataset\.manifest\.rowCount/);
  assert.match(preflightBlock, /targetCount: dataset\.manifest\.targetCount/);
  assert.match(preflightBlock, /attributeIds: hierarchicalAttributeIds/);
  assert.match(preflightBlock, /config,/);
  assert.match(preflightBlock, /scheme: scheme/);
  assert.doesNotMatch(preflightBlock, /groundTruth|pcpValueKind|pcpAxes|pcpValues|resultScope/);

  const revertBlock = dashboardSource.slice(
    revertStart,
    dashboardSource.indexOf("const changeRetrievalTarget", revertStart),
  );
  assert.match(revertBlock, /tuning\.revertRun\(\)/);
  assert.match(revertBlock, /fusionTuneHierarchy\.clear\(\)/);
  assert.match(revertBlock, /setAppliedFusionTune\(null\)/);
  assert.match(revertBlock, /setHierarchicalDraft\(defaultHierarchicalConfig\)/);
  assert.match(revertBlock, /setRankClusterFilter\("all"\)|(?:reset|clear)RankCluster/);
  assert.match(revertBlock, /setClusterDrilldown\(null\)|(?:reset|clear)RankCluster/);
  assert.doesNotMatch(
    `${applyBlock}\n${revertBlock}`,
    /setVisualClusterScheme|setVisualClusterFilter|visualAnalysis\.(?:clear|reset)|setProjection\(|setRankClusterScheme\(/,
    "Apply and Revert must preserve the chosen rank scheme and all visual-embedding cluster state",
  );
  assert.match(
    dashboardSource,
    /<TuningPanel[\s\S]*?onApplyRun=\{applyTunedRanking\}[\s\S]*?onRevertRun=\{revertTunedRanking\}/,
    "the visible Apply and Return buttons must use the Dashboard transaction handlers",
  );
  assert.match(panelSource, /Use in Top ranking/);
  assert.match(panelSource, /Return to baseline/);
  assert.match(panelSource, /Attribute weights/);
  assert.match(panelSource, /activeJointAttributeIds/);

  const pcpBlock = dashboardSource.slice(
    dashboardSource.indexOf("const pcpValues"),
    dashboardSource.indexOf("const pcpScopeKey"),
  );
  assert.match(pcpBlock, /activeRuntimeFusionResult/);
  assert.match(
    pcpBlock,
    /axis\.kind === "full" \|\| axis\.kind === "attribute"[\s\S]*?\? finalSource/,
    "only the fused full and per-attribute axes may read the runtime cube",
  );
  assert.match(
    pcpBlock,
    /: null;[\s\S]*?runtimeSource[\s\S]*?: source\[scoreOffset\(/,
    "embedding baselines and learner axes must remain backed by their exported static ranks",
  );
  assert.match(
    dashboardSource,
    /const displayedHierarchicalConfig[\s\S]*?hierarchicalDraft/,
    "the left PCP rail must display the committed learned config in Development",
  );

  const attributeStrengthBlock = dashboardSource.slice(
    dashboardSource.indexOf("const getAttributeStrengths"),
    dashboardSource.indexOf("const hasActivePcpBrush"),
  );
  assert.match(
    attributeStrengthBlock,
    /activeRuntimeFusionResult\.calibratedScores\[virtualOffset\]/,
    "all mini attribute profiles must use the recomputed Fusion matrix",
  );
  const ruleBlock = dashboardSource.slice(
    dashboardSource.indexOf("const selectionRuleAttributes"),
    dashboardSource.indexOf("const projectionPoints"),
  );
  assert.match(ruleBlock, /activeRuntimeFusionResult\.ranks/);
  assert.match(ruleBlock, /activeRuntimeFusionResult\.calibratedScores/);
  assert.match(ruleBlock, /summarizeSelectionRules/);
  assert.match(
    dashboardSource,
    /const finalMask = preRuleMask/,
    "the descriptive Rule summary must not feed back into candidate filtering",
  );
  assert.match(
    dashboardSource,
    /const activeRuntimeFusionResult = activeFusionTuneResult[\s\S]*?\?\? \(hierarchicalFusionSelected \? activeHierarchicalFusionResult : null\)/,
    "a committed learned cube must take precedence without masquerading as the manual hierarchy method",
  );
});

test("an applied personal tune becomes the selected runtime-only Top ranking method", async () => {
  const [dashboardSource, apiSource, hookSource] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/lib/tuningApi.ts"),
    source("app/lib/useTuningSession.ts"),
  ]);

  assert.match(apiSource, /if \(mode === "fusion-weight"\) return "Fusion Weight"/);
  assert.match(apiSource, /if \(mode === "residual"\) return "Residual"/);
  assert.match(apiSource, /if \(mode === "weight_staged"\) return "Weight-only · Staged"/);
  assert.match(apiSource, /if \(mode === "weight_joint"\) return "Weight-only · Joint"/);
  assert.match(apiSource, /"Native → Fusion · Staged"/);
  assert.match(apiSource, /"Native → Fusion · Joint"/);
  assert.match(
    apiSource,
    /export function tuningRankingLabel\(mode: string, algorithmVersion\?: string\): string \{[\s\S]*?`\$\{tuningModeLabel\(mode, algorithmVersion\)\} Tune`/,
  );
  assert.match(
    dashboardSource,
    /const activeFusionTuneRanking = activeFusionTuneResult && appliedFusionTune[\s\S]*?const activeTargetTunedRanking = activeTunedRanks && tuning\.run\?\.status === "succeeded"[\s\S]*?const activeTunedRanking = activeFusionTuneRanking \?\? activeTargetTunedRanking/,
  );
  assert.match(
    dashboardSource,
    /<span>Ranked by<\/span>[\s\S]*?<select[\s\S]*?value=\{activeTunedRanking\?\.id \?\? \(legacyBaselineSelected[\s\S]*?LEGACY_OURS_FULL_OPTION : rankMethod\)\}/,
    "the Ranked by control must select the applied tune instead of retaining its base method",
  );
  assert.match(
    dashboardSource,
    /\{activeTunedRanking && \([\s\S]*?<option value=\{activeTunedRanking\.id\}>\{activeTunedRanking\.label\}<\/option>/,
    "the applied tune must be exposed as a runtime-only selector option",
  );
  assert.match(
    dashboardSource,
    /if \(activeTunedRanking\) revertTunedRanking\(\);[\s\S]*?setRankMethod\(method\);/,
    "choosing an exported method must leave the tuned Top ranking and restore a baseline method",
  );
  assert.match(
    dashboardSource,
    /<TopGallery[\s\S]*?learner=\{\{[\s\S]*?id: activeTunedRanking\?\.id \?\? rankMethod,[\s\S]*?label: activeTunedRanking[\s\S]*?activeTunedRanking\.label/,
    "the Top Gallery caption and memo key must identify the applied tune too",
  );
  assert.match(
    hookSource,
    /const revertRun = useCallback\(\(\) => \{[\s\S]*?setAppliedRanks\(null\);[\s\S]*?setAppliedScores\(null\);[\s\S]*?setAppliedRunId\(null\);/,
    "returning to baseline must clear both tuned payloads and the applied run identity",
  );
  assert.doesNotMatch(
    dashboardSource.slice(
      dashboardSource.indexOf("const displayMethods"),
      dashboardSource.indexOf("const smartFilterLearners"),
    ),
    /activeTunedRanking|personal-tune/,
    "the runtime tune option must not enter the PCP learner list",
  );
});

test("Personal Tune reports label-isolated Clean Validation while keeping legacy scopes", async () => {
  const [dashboardSource, panelSource, hookSource, apiSource] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TuningPanel.tsx"),
    source("app/lib/useTuningSession.ts"),
    source("app/lib/tuningApi.ts"),
  ]);

  assert.match(panelSource, /Clean Validation/);
  assert.match(panelSource, /Probe Val/);
  assert.match(panelSource, /<summary>\s*\{vqaValidation \? "Evaluation details" : "Historical evaluation · original scope"\}[\s\S]*?<\/summary>/);
  assert.match(panelSource, /aria-label="Validation AP comparison"/);
  assert.doesNotMatch(panelSource, /testEvaluation/);
  assert.doesNotMatch(panelSource, /Val \/ Test results|H · Human Feedback/);
  assert.match(panelSource, /label-isolated/);
  assert.match(panelSource, /\{tuningModeLabel\(run\.mode, run\.algorithmVersion\)\} · \{run\.status === "queued"/);
  assert.match(panelSource, /Best F1/);
  assert.match(panelSource, /original attribute split/);
  assert.match(panelSource, /Joint-derived/);
  assert.match(apiSource, /splitAudit\?: TuningSplitAudit/);
  assert.match(apiSource, /"clean-validation" \| "probe-validation"/);
  assert.doesNotMatch(panelSource, /Session-only; source unchanged/);
  assert.match(
    hookSource,
    /const runBaseMethod = tuningBaseMethod\(mode, baseMethod\);[\s\S]*?createTuningRun\(sessionId, \{[\s\S]*?mode,[\s\S]*?baseMethod: runBaseMethod/,
    "the tuning request must not inherit the current UI result mask",
  );
  assert.doesNotMatch(panelSource, /resultMask/);
  assert.doesNotMatch(hookSource, /resultMask/);
  assert.doesNotMatch(apiSource, /resultMask/);
  const validationStart = dashboardSource.indexOf("const validationComparisonRows");
  const validationEnd = dashboardSource.indexOf("const getRank", validationStart);
  assert.ok(validationStart >= 0 && validationEnd > validationStart);
  const validationBlock = dashboardSource.slice(validationStart, validationEnd);
  assert.doesNotMatch(
    validationBlock,
    /activeTunedRanks|activeFusionTuneResult|appliedFusionTune|personal-tune/,
    "personal runs must not be re-evaluated inside the separate Web Validation table",
  );
});

test("Functions separates Probe updates and two Weight schedules while TuningPanel keeps legacy results", async () => {
  const [dashboardSource, actionsSource, panelSource, apiSource] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TuningFunctionActions.tsx"),
    source("app/components/TuningPanel.tsx"),
    source("app/lib/tuningApi.ts"),
  ]);

  assert.match(dashboardSource, /<TuningPanel[\s\S]*?state=\{tuning\}/);
  assert.match(
    dashboardSource,
    /developmentMode && !hierarchicalFusionSelected && !legacyBaselineSelected && \([\s\S]*?<TuningFunctionActions[\s\S]*?state=\{tuning\}[\s\S]*?taskId=\{selectedTaskId\}[\s\S]*?targetId=\{retrievalTarget\}[\s\S]*?baseMethod=\{rankMethod\}/,
  );
  assert.match(actionsSource, /mode: "weight_staged"/);
  assert.match(actionsSource, /mode: "weight_joint"/);
  assert.doesNotMatch(actionsSource, /mode: "probe_staged"|mode: "probe_joint"/);
  assert.match(actionsSource, /label: "Staged Weight Refinement"/);
  assert.match(actionsSource, /label: "Joint Weight Tune"/);
  assert.match(actionsSource, /Update Probes/);
  assert.match(actionsSource, /冻结所选 Probe 快照/);
  assert.match(actionsSource, /refinementCapability\(state\.refinementCapabilities, mode\)/);
  assert.match(actionsSource, /targetId !== "joint"/);
  assert.match(actionsSource, /Select Joint as the retrieval target first/);
  assert.match(actionsSource, /disabled=\{!refinementCapability\(state\.refinementCapabilities, action\.mode\)\.available \|\| !tuningReady \|\| commonDisabled \|\| missingSnapshot\}/);
  assert.match(actionsSource, /onClick=\{\(\) => launch\(action\.mode\)\}/);
  assert.doesNotMatch(actionsSource, /launch\("fusion-weight"\)|launch\("residual"\)/);
  assert.match(actionsSource, /fusionBaseReady = baseMethod === "Ours-Full"/);
  assert.match(actionsSource, /state\.counts\.usablePositive \+ state\.counts\.usableNegative >= 1/);
  assert.match(actionsSource, /unconfirmedJointNegativeCount = \[\.\.\.state\.annotations\.values\(\)\]\.filter/);
  assert.match(actionsSource, /!annotation\.failureAttributionConfirmed/);
  assert.match(actionsSource, /!annotation\.failedAttributeIds\?\.length/);
  assert.match(actionsSource, /Confirm failed attributes for/);
  assert.match(actionsSource, /tuningReady = jointTargetReady[\s\S]*?fusionBaseReady[\s\S]*?hasUsableFeedback[\s\S]*?supervisionResolved/);
  assert.match(actionsSource, /Add at least one positive or negative correction/);
  assert.match(actionsSource, /Select Ours-Full in Top Results first/);
  assert.match(actionsSource, /state\.serviceState !== "ready"/);
  assert.match(actionsSource, /state\.session\.taskId === taskId/);
  assert.match(actionsSource, /state\.session\.targetId === targetId/);
  assert.match(actionsSource, /state\.session\.baseMethod === baseMethod/);
  assert.match(actionsSource, /\|\| !sessionMatches/);
  assert.match(actionsSource, /Preparing the selected tuning session/);
  assert.doesNotMatch(panelSource, /runTuning|tuning-mode-card/);
  assert.match(apiSource, /export type TuningLaunchMode =[\s\S]*?"weight_staged"[\s\S]*?"weight_joint"[\s\S]*?"probe_staged"[\s\S]*?"probe_joint"/);
  assert.match(apiSource, /export type LegacyTuningMode = "prototype" \| "label" \| "fusion-weight" \| "residual"/);
  assert.match(apiSource, /if \(mode === "weight_staged"\) return "Weight-only · Staged"/);
  assert.match(apiSource, /if \(mode === "weight_joint"\) return "Weight-only · Joint"/);
  assert.match(apiSource, /"Native → Fusion · Staged"/);
  assert.match(apiSource, /"Native → Fusion · Joint"/);
  assert.match(apiSource, /return "Ours-Full"/);
  assert.match(apiSource, /if \(mode === "prototype"\) return "Legacy Prototype"/);
  assert.match(apiSource, /if \(mode === "label"\) return "Legacy Label Head"/);
  assert.match(panelSource, /Before AP/);
  assert.match(panelSource, /After AP/);
  assert.match(panelSource, /ΔAP/);
  assert.match(panelSource, /Clean Validation/);
  assert.match(panelSource, /Attribute exponents γ/);
  assert.match(panelSource, /independent \[0\.05, 3\.00\]/);
  assert.match(panelSource, /onClick=\{\(\) => void applyCurrentRun\(\)/);
  assert.match(panelSource, /onClick=\{revertCurrentRun\}/);
  assert.match(panelSource, /Use in Top ranking/);
  assert.match(panelSource, /isCurrentTuningMode\(run\.mode\)[\s\S]*?"Use in Top ranking"/);
  assert.match(panelSource, /Return to baseline/);
  assert.doesNotMatch(panelSource, /Session-only; source unchanged/);
  assert.match(apiSource, /learnerWeightsByAttribute\?: Record<string, Record<string, number>>/);
  assert.match(apiSource, /activeAttributeIds\?: string\[\]/);
  assert.match(panelSource, /modelSummary\?\.learnerWeightsByAttribute/);
  assert.match(panelSource, /Learner weights by attribute/);
  assert.match(panelSource, /activeAttributeIds\.has\(attribute\)/);
  assert.match(panelSource, /kept equal/);
  assert.match(panelSource, /className="tuning-run-details"/);
  assert.match(panelSource, /Legacy shared weights/);
  assert.match(panelSource, /Residual coefficients/);
  assert.match(panelSource, /Embedding fusion λ/);
  assert.match(panelSource, /probes updated/);
  assert.doesNotMatch(panelSource, /launch\("prototype"\)|launch\("label"\)/);
  assert.doesNotMatch(dashboardSource, /run\.mode === "prototype"/);
});

test("tuning API derives ownership from credentials and never accepts a client userId", async () => {
  const apiSource = await source("app/lib/tuningApi.ts");

  assert.match(apiSource, /credentials: "same-origin"/);
  assert.match(
    apiSource,
    /bootstrapTuning\(input: \{[\s\S]*?taskId: string;[\s\S]*?targetId: string;[\s\S]*?baseMethod: string;/,
  );
  assert.doesNotMatch(
    apiSource,
    /\buserId\b/,
    "the browser must not choose or transmit the server-owned user identifier",
  );
  assert.match(apiSource, /JSON\.stringify\(input\)/);
  assert.match(apiSource, /JSON\.stringify\(\{[\s\S]*?imageId: annotation\.imageId,[\s\S]*?label: annotation\.label/);
});

test("pending feedback is confirmed before a tuning snapshot can launch", async () => {
  const [hookSource, actionsSource, panelSource] = await Promise.all([
    source("app/lib/useTuningSession.ts"),
    source("app/components/TuningFunctionActions.tsx"),
    source("app/components/TuningPanel.tsx"),
  ]);

  assert.match(hookSource, /pendingAnnotationCount: number/);
  assert.match(hookSource, /pendingAnnotationWritesRef/);
  assert.match(hookSource, /Promise\.allSettled\(pending\)/);
  assert.match(
    hookSource,
    /const runTuning = useCallback[\s\S]*?await waitForPendingAnnotations\(generation, sessionId\);[\s\S]*?createTuningRun\(sessionId/,
    "Run must flush the active session's annotation requests before creating its immutable snapshot",
  );
  assert.match(hookSource, /targetId !== "joint"/);
  assert.match(hookSource, /fetchRefinementCapabilities\(taskId\)/);
  assert.match(hookSource, /if \(!capability\.available\) throw new Error/);
  assert.match(hookSource, /confirmedAnnotationsRef\.current\.values\(\)/);
  assert.match(hookSource, /Confirm failed attributes for every Joint negative before tuning/);
  assert.match(
    hookSource,
    /const runTuning = useCallback[\s\S]*?!enabled[\s\S]*?session\.taskId !== taskId[\s\S]*?session\.targetId !== targetId[\s\S]*?session\.baseMethod !== baseMethod/,
    "Run must not launch against a stale task, target, or base-method session",
  );
  assert.match(actionsSource, /const annotationsPending = state\.pendingAnnotationCount > 0/);
  assert.match(actionsSource, /\|\| annotationsPending/);
  assert.match(actionsSource, /disabled=\{!refinementCapability\(state\.refinementCapabilities, action\.mode\)\.available \|\| !tuningReady \|\| commonDisabled \|\| missingSnapshot\}/);
  assert.match(panelSource, /Saving \{pendingAnnotationCount\}/);
});

test("Tune feedback reports original-supervision membership and action before Run", async () => {
  const [apiSource, hookSource, dashboardSource, actionsSource, panelSource, css] = await Promise.all([
    source("app/lib/tuningApi.ts"),
    source("app/lib/useTuningSession.ts"),
    source("app/Dashboard.tsx"),
    source("app/components/TuningFunctionActions.tsx"),
    source("app/components/TuningPanel.tsx"),
    source("app/globals.css"),
  ]);

  assert.match(apiSource, /TuningFeedbackOrigin = "new" \| "existing"/);
  assert.match(apiSource, /TuningFeedbackRelation = "new" \| "override" \| "reinforce" \| "uncertain"/);
  assert.match(apiSource, /supervision\?: TuningAnnotationSupervision/);
  assert.match(hookSource, /feedbackBreakdownFromAnnotations/);
  assert.match(hookSource, /summary\.newCount \+= 1/);
  assert.match(hookSource, /summary\.existingCount \+= 1/);
  assert.match(hookSource, /summary\.overrideCount \+= 1/);
  assert.match(hookSource, /summary\.reinforceCount \+= 1/);
  assert.match(hookSource, /summary\.reviewOnlyCount \+= 1/);
  assert.match(panelSource, /\{feedbackBreakdown\.totalCount\} feedback/);
  assert.match(panelSource, /\{feedbackBreakdown\.newCount\} new/);
  assert.match(panelSource, /\{feedbackBreakdown\.existingCount\} existing/);
  assert.match(panelSource, /feedbackRelationSummary/);
  assert.match(panelSource, /Override/);
  assert.match(panelSource, /Reinforce/);
  assert.match(panelSource, /review-only \(\?\)/);
  assert.match(actionsSource, /Waiting for original-supervision checks/);
  assert.doesNotMatch(dashboardSource, /feedback-mini-supervision|feedback-strip/);
  assert.doesNotMatch(dashboardSource, /className="feedback-summary"|\{positiveCount\} positive|\{negativeCount\} negative/);
  assert.doesNotMatch(dashboardSource, /\{tuning\.counts\.uncertain\} uncertain/);
  assert.match(dashboardSource, /onClearAll=\{\(\) => tuning\.clearAnnotations\(\)\}/);
  assert.match(dashboardSource, /clearDisabled=\{tuning\.busy \|\| fixedValidation\.status !== "ready"\}/);
  assert.doesNotMatch(panelSource, /className="tuning-counts"/);
  assert.match(css, /\.feedback-summary \.positive/);
  assert.match(css, /\.feedback-summary \.negative/);
  assert.doesNotMatch(
    hookSource,
    /groundTruth|ground-truth/i,
    "the browser must use the server's recovered original supervision, not public GT",
  );
});

test("Joint negative feedback can confirm or replace the suggested failed attributes", async () => {
  const [apiSource, hookSource, dashboardSource, reviewSource, css] = await Promise.all([
    source("app/lib/tuningApi.ts"),
    source("app/lib/useTuningSession.ts"),
    source("app/Dashboard.tsx"),
    source("app/components/AnnotationReviewGallery.tsx"),
    source("app/globals.css"),
  ]);

  assert.match(apiSource, /failedAttributeIds\?: string\[\]/);
  assert.match(apiSource, /suggestedFailedAttributeId\?: string \| null/);
  assert.match(apiSource, /failureAttributionConfirmed\?: boolean/);
  assert.match(
    apiSource,
    /putTuningAnnotation[\s\S]*?failedAttributeIds: annotation\.failedAttributeIds[\s\S]*?suggestedFailedAttributeId: annotation\.suggestedFailedAttributeId[\s\S]*?failureAttributionConfirmed: annotation\.failureAttributionConfirmed/,
  );
  assert.match(hookSource, /updateFailureAttributes: \([\s\S]*?failedAttributeIds: readonly string\[\]/);
  assert.match(hookSource, /const updateFailureAttributes = useCallback/);
  assert.match(hookSource, /Failed attributes can only be confirmed for a negative Joint label/);
  assert.match(hookSource, /failureAttributionConfirmed: true/);
  assert.match(hookSource, /advanceAnnotationRevision\(\)/);
  assert.match(reviewSource, /function FailureAttributeSelector/);
  assert.match(reviewSource, /annotation\.label >= 0/);
  assert.match(reviewSource, /Suggested · unconfirmed/);
  assert.match(reviewSource, /Choose at least one failed attribute|selectedIds\.length === 0/);
  assert.match(reviewSource, /type="checkbox"/);
  assert.match(reviewSource, /renderFeedback=\{\(item\) => renderReviewControls\(item, true\)\}/);
  assert.match(dashboardSource, /jointSession=\{retrievalTarget === jointTarget\?\.id\}/);
  assert.match(dashboardSource, /attributeTargets=\{attributeTargets\}/);
  assert.match(dashboardSource, /tuning\.updateFailureAttributes\(item, failedAttributeIds\)/);
  assert.match(css, /\.failure-attribute-selector/);
  assert.match(css, /\.failure-attribute-options/);
});

test("review-only feedback does not invalidate a trained snapshot by itself", async () => {
  const hookSource = await source("app/lib/useTuningSession.ts");
  assert.match(
    hookSource,
    /function trainableFeedbackValue[\s\S]*?annotation\.label !== 0 \? annotation\.label : null/,
  );
  assert.match(
    hookSource,
    /trainableFeedbackValue\(previousConfirmed\) !== trainableFeedbackValue\(confirmed\)/,
  );
  assert.match(
    hookSource,
    /snapshot\.values\(\)\]\.some\(\(annotation\) => annotation\.label !== 0\)/,
  );
});

test("session generations quarantine stale mutation and tuned-rank responses", async () => {
  const hookSource = await source("app/lib/useTuningSession.ts");
  const annotationBlock = hookSource.match(
    /const updateAnnotation = useCallback[\s\S]*?\n  const clearAnnotations/,
  )?.[0] ?? "";
  const runBlock = hookSource.match(
    /const runTuning = useCallback[\s\S]*?\n  const applyRun/,
  )?.[0] ?? "";
  const applyBlock = hookSource.match(
    /const applyRun = useCallback[\s\S]*?\n  const revertRun/,
  )?.[0] ?? "";

  assert.match(annotationBlock, /isActiveSessionOperation\(generation, sessionId\)/);
  assert.match(annotationBlock, /annotationSequenceRef/);
  assert.match(annotationBlock, /confirmedAnnotationsRef/);
  assert.match(runBlock, /response\.run\.sessionId !== sessionId/);
  assert.match(runBlock, /currentRunIdRef\.current = response\.run\.id/);
  assert.match(applyBlock, /rankGeneration !== rankRequestGenerationRef\.current/);
  assert.match(applyBlock, /currentRunIdRef\.current !== run\.id/);
  assert.match(applyBlock, /Promise\.all\(\[[\s\S]*?fetchTunedRanks\(run, rowCount\),[\s\S]*?fetchTunedScores\(run, rowCount\)/);
  assert.match(applyBlock, /setAppliedRanks\(ranks\)/);
  assert.match(applyBlock, /setAppliedScores\(scores\)/);
});

test("polling retries transient failures and Revert invalidates an in-flight rank load", async () => {
  const [hookSource, panelSource] = await Promise.all([
    source("app/lib/useTuningSession.ts"),
    source("app/components/TuningPanel.tsx"),
  ]);
  const pollingBlock = hookSource.match(
    /useEffect\(\(\) => \{\n    if \(!run \|\| run\.status[\s\S]*?\n  \}, \[isActiveSessionOperation, run, runRevision\]\);/,
  )?.[0] ?? "";
  const revertBlock = hookSource.match(
    /const revertRun = useCallback\(\(\) => \{[\s\S]*?\n  \}, \[\]\);/,
  )?.[0] ?? "";

  assert.match(pollingBlock, /retryDelay = Math\.min\(retryDelay \* 2, 9600\)/);
  assert.match(pollingBlock, /schedulePoll\(retryDelay\)/);
  assert.match(pollingBlock, /cancelled = true/);
  assert.match(revertBlock, /rankRequestGenerationRef\.current \+= 1/);
  assert.match(revertBlock, /setAppliedRanks\(null\)/);
  assert.match(revertBlock, /setAppliedScores\(null\)/);
  assert.match(panelSource, /const applying = busy \|\| applyBusy/);
  assert.match(panelSource, /disabled=\{applying \|\| resultStale \|\| isApplied\}/);
});

test("a failed annotation restores confirmed state without advancing its revision", async () => {
  const hookSource = await source("app/lib/useTuningSession.ts");
  const annotationBlock = hookSource.match(
    /const updateAnnotation = useCallback[\s\S]*?\n  const clearAnnotations/,
  )?.[0] ?? "";
  const failureBlock = annotationBlock.match(/\} catch \(reason\) \{[\s\S]*?throw reason;\n      \}/)?.[0] ?? "";

  assert.match(failureBlock, /confirmedAnnotationsRef\.current\.get\(item\.id\)/);
  assert.doesNotMatch(failureBlock, /advanceAnnotationRevision|setAnnotationRevision/);
  assert.match(
    annotationBlock,
    /confirmedAnnotationsRef\.current[\s\S]*?advanceAnnotationRevision\(\);[\s\S]*?\} catch \(reason\)/,
    "the revision advances only on a confirmed server mutation",
  );
});
