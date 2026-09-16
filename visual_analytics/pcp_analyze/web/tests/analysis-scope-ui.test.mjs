import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = () => readFile(
  new URL("../app/Dashboard.tsx", import.meta.url),
  "utf8",
);

function componentCall(contents, name) {
  const start = contents.indexOf(`<${name}`);
  assert.notEqual(start, -1, `${name} must be rendered`);
  const end = contents.indexOf("/>", start);
  assert.notEqual(end, -1, `${name} must have a closing tag`);
  return contents.slice(start, end + 2);
}

test("explicit Development and Frozen Test scopes reach analysis views", async () => {
  const dashboard = await source();
  const samplePcp = componentCall(dashboard, "ParallelCoordinates");
  const clusterPcp = componentCall(dashboard, "ClusterSummaryParallelCoordinates");
  const projection = componentCall(dashboard, "ProjectionScatter");
  const maskPipeline = dashboard.slice(
    dashboard.indexOf("const analysisMask"),
    dashboard.indexOf("const galleryItems"),
  );

  assert.match(maskPipeline, /analysisScopeMask\(dataset\.manifest\.rowCount, \{[\s\S]*?developmentMask: developmentWithoutFixedVal\(dataset\.developmentMask, fixedValidation\.mask\),[\s\S]*?validationMask: dataset\.validationMask,[\s\S]*?testMask: dataset\.testMask,[\s\S]*?\}, resultScope\)/);
  assert.match(maskPipeline, /const rankStructuralMask = useMemo[\s\S]*?rankClusterFilter[\s\S]*?rankClusterLabels/);
  assert.match(maskPipeline, /const visualStructuralMask = useMemo[\s\S]*?visualClusterFilter[\s\S]*?visualClusterLabels/);
  assert.match(maskPipeline, /intersectMasks\(pcpEligibleMask, visualStructuralMask\)/);
  assert.match(maskPipeline, /filterAnalysisRows\([\s\S]*?projectionPoints,[\s\S]*?analysisMask/);
  assert.match(
    samplePcp,
    /labels=\{rankClustersReady \? rankClusterLabels : undefined\}/,
  );
  assert.match(samplePcp, /candidateMask=\{pcpCandidateMask\}/);
  assert.match(clusterPcp, /candidateMask=\{analysisMask\}/);
  assert.match(clusterPcp, /labels=\{rankClusterLabels\}/);
  assert.match(projection, /points=\{scopedProjectionPoints\}/);
  assert.match(projection, /candidateMask=\{projectionEligibleMask\}/);
  assert.match(projection, /highlightMask=\{activePcpHighlightMask\}/);
  assert.match(maskPipeline, /const activePcpHighlightMask = hasPcpRestriction \? pcpEligibleMask : null/);
  assert.match(dashboard, /<span>Data purpose<\/span>/);
  assert.match(
    dashboard,
    /changeResultScope\(event\.target\.value as ResultScope\)/,
  );
  assert.match(dashboard, /validationMode[\s\S]*?<ValidationComparison/);
  assert.match(dashboard, /testMode[\s\S]*?<h2 id="test-audit-heading">Diagnosis and Refinement<\/h2>/);
});

test("visual cluster legend reports PCP-selected and scoped totals independently", async () => {
  const dashboard = await source();

  assert.match(dashboard, /const analysisVisualClusterCounts = useMemo/);
  assert.match(dashboard, /const pcpSelectedVisualClusterCounts = useMemo/);
  assert.match(dashboard, /if \(!analysisMask\[index\]\) continue;/);
  assert.match(dashboard, /counts\.set\(key, \(counts\.get\(key\) \?\? 0\) \+ 1\)/);
  assert.match(dashboard, /const scopedSize = analysisVisualClusterCounts\.counts\.get/);
  assert.match(dashboard, /const pcpSelectedSize = pcpSelectedVisualClusterCounts\.get/);
  assert.match(dashboard, /activePcpHighlightMask[\s\S]*?`PCP \$\{pcpSelectedSize\.toLocaleString\(\)\}\/\$\{scopedSize\.toLocaleString\(\)\}/);
  assert.match(dashboard, /: `\$\{scopedSize\.toLocaleString\(\)\} · `/);
});

test("Refinement exposes training actions beside feedback only in Development", async () => {
  const dashboard = await source();
  const start = dashboard.indexOf('<div className="refinement-actions-block">');
  const end = dashboard.indexOf('</section>', start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const functionsBlock = dashboard.slice(start, end);

  assert.match(functionsBlock, />Refinement and Validation</);
  assert.doesNotMatch(functionsBlock, />Reload</);
  assert.match(functionsBlock, /refinement-training-actions/);
  assert.match(functionsBlock, /developmentMode && !hierarchicalFusionSelected && !legacyBaselineSelected && \(/);
  assert.match(functionsBlock, /<TuningFunctionActions[\s\S]*?state=\{tuning\}[\s\S]*?taskId=\{selectedTaskId\}[\s\S]*?targetId=\{retrievalTarget\}[\s\S]*?baseMethod=\{rankMethod\}/);
  assert.doesNotMatch(functionsBlock, /Tune model|Load attributes|attribute-picker|Loaded:/);
});

test("Results does not duplicate the personal tuning controls", async () => {
  const dashboard = await source();

  assert.doesNotMatch(dashboard, />Tune model<|focusTuning|revise-button/);
  assert.match(dashboard, /<span>Human Feedback<\/span>/);
  const start = dashboard.indexOf('<div className="tuning-block"');
  const end = dashboard.indexOf('</section>', start);
  const tuningBlock = dashboard.slice(start, end);
  assert.equal((tuningBlock.match(/<TuningFunctionActions/g) ?? []).length, 1);
  assert.equal((dashboard.match(/<TuningFunctionActions/g) ?? []).length, 1);
  assert.doesNotMatch(tuningBlock, /Fusion Weight tune|Residual tune/);
});
