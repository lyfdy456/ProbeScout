import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  effectivePcpSelectionRanges,
  summarizeSelectionOverlap,
} from "../app/lib/selectionOverlap.ts";
import { summarizeSelectionRules } from "../app/lib/selectionRuleSummary.ts";

function mask(...values) {
  return Uint8Array.from(values);
}

function rangeRule(
  axisId,
  lower = 0.8,
  upper = 1,
  source = "brush",
  valueKind = "rank",
) {
  return { kind: "range", axisId, valueKind, lower, upper, source };
}

function categoryRule(field, value) {
  return { kind: "category", field, value };
}

function projectionRule(
  projection = "umap",
  xDomain = [-1, 1],
  yDomain = [-2, 2],
) {
  return { kind: "projection", projection, xDomain, yDomain };
}

function condition(
  id,
  conditionMask,
  rule = rangeRule(id),
  ruleResult = { kind: "none", reason: "no-attributes" },
) {
  return { id, label: id.toUpperCase(), rule, ruleResult, mask: conditionMask };
}

function summaryOf(input) {
  const result = summarizeSelectionOverlap(input);
  assert.equal(result.kind, "summary");
  return result.summary;
}

function regionsByKey(summary) {
  return new Map(summary.regions.map((region) => [region.key, region]));
}

function assertClose(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 1e-6, `${message}: ${actual} != ${expected}`);
}

test("keeps every active PCP axis as its own stable condition", () => {
  const ranges = effectivePcpSelectionRanges(
    ["full", "attribute-a", "attribute-b"],
    { full: [0.8, 1], "attribute-a": [0.2, 0.9], ignored: [0.1, 0.2] },
    { "attribute-b": [0.95, 1] },
  );

  assert.deepEqual(ranges, [
    { axisId: "full", range: [0.8, 1], source: "zoom" },
    { axisId: "attribute-a", range: [0.2, 0.9], source: "zoom" },
    { axisId: "attribute-b", range: [0.95, 1], source: "brush" },
  ]);
});

test("merges committed zoom and working brush on the same PCP axis", () => {
  assert.deepEqual(
    effectivePcpSelectionRanges(
      ["full", "attribute-a"],
      { full: [0.9, 1], "attribute-a": [0.1, 0.8] },
      { full: [0.99, 0.95], "attribute-a": [0.4, 0.95] },
    ),
    [
      { axisId: "full", range: [0.95, 0.99], source: "zoom+brush" },
      { axisId: "attribute-a", range: [0.4, 0.8], source: "zoom+brush" },
    ],
  );
  assert.throws(
    () => effectivePcpSelectionRanges(["full"], { full: [0.9, 1] }, { full: [0.1, 0.2] }),
    /do not overlap/,
  );
});

test("cluster and PCP-axis conditions report exact mutually exclusive Venn regions", () => {
  const summary = summaryOf({
    baseMask: mask(1, 1, 1, 1, 1, 1, 1, 1, 0),
    conditions: [
      condition(
        "rank-cluster",
        mask(1, 1, 0, 1, 1, 0, 0, 0, 1),
        categoryRule("rank-cluster", "K50 · 05"),
      ),
      condition(
        "pcp-axis:full",
        mask(0, 0, 1, 1, 1, 0, 1, 0, 1),
        rangeRule("full", 0.9, 1, "zoom+brush"),
      ),
    ],
    finalMask: mask(0, 0, 0, 1, 1, 0, 0, 0, 0),
  });

  assert.equal(summary.baseCount, 8);
  assert.deepEqual(
    summary.conditions.map(({ id, code, count, cumulativeCount }) => ({
      id,
      code,
      count,
      cumulativeCount,
    })),
    [
      { id: "rank-cluster", code: "A", count: 4, cumulativeCount: 4 },
      { id: "pcp-axis:full", code: "B", count: 4, cumulativeCount: 2 },
    ],
  );
  assert.equal(summary.conditions[0].rule.kind, "category");
  assert.equal(summary.conditions[1].rule.source, "zoom+brush");
  const regions = regionsByKey(summary);
  assert.deepEqual(regions.get(1), { key: 1, codes: ["A"], count: 2 });
  assert.deepEqual(regions.get(2), { key: 2, codes: ["B"], count: 2 });
  assert.deepEqual(regions.get(3), { key: 3, codes: ["A", "B"], count: 2 });
  assert.equal(summary.outsideVisibleCount, 2);
  assert.equal(summary.coreIntersectionCount, 2);
  assert.equal(summary.finalCount, 2);
});

test("condition-alone ruleResult reports every attribute's exact 5th–95th percentile ranges", () => {
  const selected = mask(1, 1, 1, 1, 0, 0);
  const conditionRuleResult = summarizeSelectionRules({
    selectionMask: selected,
    baseMask: mask(1, 1, 1, 1, 1, 1),
    attributes: [
      {
        id: "shape",
        label: "Shape",
        ranks: Float32Array.from([0, 0.2, 0.4, 0.6, 0.8, 1]),
        calibratedScores: Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      },
      {
        id: "color",
        label: "Color",
        ranks: Float32Array.from([0.9, 0.7, 0.5, 0.3, 0.1, 0]),
        calibratedScores: Float32Array.from([0.8, 0.6, 0.4, 0.2, 0.1, 0]),
      },
    ],
  });
  const summary = summaryOf({
    baseMask: mask(1, 1, 1, 1, 1, 1),
    conditions: [condition("pcp-axis:full", selected, rangeRule("full", 0, 0.6), conditionRuleResult)],
    finalMask: selected,
  });

  const ruleResult = summary.conditions[0].ruleResult;
  assert.equal(ruleResult.kind, "summary");
  assert.equal(ruleResult.summary.selectedCount, 4);
  assert.deepEqual(
    ruleResult.summary.attributes.map(({ attributeId, sampleCount }) => ({ attributeId, sampleCount })),
    [
      { attributeId: "shape", sampleCount: 4 },
      { attributeId: "color", sampleCount: 4 },
    ],
  );
  const [shape, color] = ruleResult.summary.attributes;
  assertClose(shape.rank.lowerQuantile, 0.03, "shape rank q05");
  assertClose(shape.rank.upperQuantile, 0.57, "shape rank q95");
  assertClose(shape.calibratedScore.lowerQuantile, 0.115, "shape score q05");
  assertClose(shape.calibratedScore.upperQuantile, 0.385, "shape score q95");
  assertClose(color.rank.lowerQuantile, 0.33, "color rank q05");
  assertClose(color.rank.upperQuantile, 0.87, "color rank q95");
});

test("three independent conditions expose all seven exact Venn regions", () => {
  const summary = summaryOf({
    baseMask: mask(1, 1, 1, 1, 1, 1, 1, 1),
    conditions: [
      condition("a", mask(0, 1, 0, 1, 0, 1, 0, 1)),
      condition("b", mask(0, 0, 1, 1, 0, 0, 1, 1)),
      condition(
        "projection-brush",
        mask(0, 0, 0, 0, 1, 1, 1, 1),
        projectionRule("umap", [-0.25, 0.75], [-1.5, 2.5]),
      ),
    ],
    finalMask: mask(0, 0, 0, 0, 0, 0, 0, 1),
  });

  assert.equal(summary.regions.length, 7);
  for (let key = 1; key <= 7; key += 1) {
    assert.equal(regionsByKey(summary).get(key)?.count, 1, `region ${key}`);
  }
  assert.equal(summary.outsideVisibleCount, 1);
  assert.deepEqual(summary.conditions[2].rule, {
    kind: "projection",
    projection: "umap",
    xDomain: [-0.25, 0.75],
    yDomain: [-1.5, 2.5],
  });
  assert.deepEqual(summary.conditions.map((entry) => entry.cumulativeCount), [4, 2, 1]);
  assert.equal(summary.coreIntersectionCount, 1);
  assert.equal(summary.finalCount, 1);
});

test("more than three conditions remain available and expose an exact cumulative AND", () => {
  const summary = summaryOf({
    baseMask: mask(1, 1, 1, 1, 1, 1),
    conditions: [
      condition("a", mask(1, 1, 1, 1, 0, 0)),
      condition("b", mask(1, 1, 1, 0, 1, 0)),
      condition("c", mask(1, 1, 0, 1, 1, 0)),
      condition("d", mask(0, 1, 1, 1, 1, 0)),
    ],
    finalMask: mask(0, 1, 0, 0, 0, 0),
  });

  assert.deepEqual(summary.conditions.map(({ id, code }) => ({ id, code })), [
    { id: "a", code: "A" },
    { id: "b", code: "B" },
    { id: "c", code: "C" },
    { id: "d", code: "D" },
  ]);
  assert.deepEqual(summary.conditions.map((entry) => entry.count), [4, 4, 4, 4]);
  assert.deepEqual(summary.conditions.map((entry) => entry.cumulativeCount), [4, 3, 2, 1]);
  assert.equal(summary.visibleConditions.length, 0, "four conditions use the many-set fallback");
  assert.equal(summary.hiddenConditions.length, 4, "the fallback retains every condition summary");
  assert.equal(summary.coreIntersectionCount, 1);
  assert.equal(summary.finalCount, 1);
});

test("rows outside the base population never contribute to conditions, stages, or final", () => {
  const summary = summaryOf({
    baseMask: mask(1, 1, 0, 0),
    conditions: [
      condition("a", mask(1, 0, 1, 1)),
      condition("b", mask(1, 1, 1, 1)),
    ],
    stages: [{ id: "stage", label: "Projection box", mask: mask(1, 0, 1, 1) }],
    finalMask: mask(1, 0, 1, 1),
  });

  assert.equal(summary.baseCount, 2);
  assert.deepEqual(summary.conditions.map((entry) => entry.count), [1, 2]);
  assert.equal(summary.coreIntersectionCount, 1);
  assert.deepEqual(summary.stages.map(({ id, count }) => ({ id, count })), [
    { id: "stage", count: 1 },
  ]);
  assert.equal(summary.finalCount, 1);
});

test("downstream diagnostic reduces the independent AND and finalMask equals the last stage", () => {
  const summary = summaryOf({
    baseMask: mask(1, 1, 1, 1, 1, 1),
    conditions: [
      condition("a", mask(1, 1, 1, 1, 0, 0)),
      condition("b", mask(1, 1, 1, 0, 1, 0)),
    ],
    stages: [
      { id: "diagnostic", label: "Diagnostic Top 50", mask: mask(0, 1, 0, 0, 0, 0) },
    ],
    finalMask: mask(0, 1, 0, 0, 0, 0),
  });

  assert.equal(summary.coreIntersectionCount, 3);
  assert.deepEqual(summary.stages.map(({ label, count }) => ({ label, count })), [
    { label: "Diagnostic Top 50", count: 1 },
  ]);
  assert.equal(summary.finalCount, 1);
});

test("mask contracts fail explicitly and finalMask must strictly equal the last intersection", () => {
  const validBase = mask(1, 1, 1);
  const validCondition = condition("a", mask(1, 1, 0));

  assert.throws(
    () => summarizeSelectionOverlap({
      baseMask: validBase,
      conditions: [{ ...validCondition, mask: mask(1, 1) }],
      finalMask: mask(1, 0, 0),
    }),
    /row-aligned/,
  );
  assert.throws(
    () => summarizeSelectionOverlap({
      baseMask: mask(1, 2, 1),
      conditions: [],
      finalMask: mask(0, 0, 0),
    }),
    /binary mask/,
  );
  assert.throws(
    () => summarizeSelectionOverlap({
      baseMask: validBase,
      conditions: [validCondition, condition("a", mask(1, 0, 0))],
      finalMask: mask(1, 0, 0),
    }),
    /duplicated/,
  );
  assert.throws(
    () => summarizeSelectionOverlap({
      baseMask: validBase,
      conditions: [validCondition],
      stages: [{ id: "stage", label: "Stage", mask: mask(1, 0, 1) }],
      finalMask: mask(1, 0, 1),
    }),
    /subset of the preceding selection stage/,
  );

  const intersectionConditions = [
    condition("a", mask(1, 1, 0)),
    condition("b", mask(1, 0, 1)),
  ];
  assert.throws(
    () => summarizeSelectionOverlap({
      baseMask: validBase,
      conditions: intersectionConditions,
      finalMask: mask(1, 1, 0),
    }),
    /must equal the preceding selection stage/,
    "a strict superset of the AND must fail",
  );
  assert.throws(
    () => summarizeSelectionOverlap({
      baseMask: validBase,
      conditions: intersectionConditions,
      finalMask: mask(0, 0, 0),
    }),
    /must equal the preceding selection stage/,
    "a strict subset of the AND must fail",
  );
});

test("Dashboard, panel, and CSS expose per-condition overlap semantics", async () => {
  const [dashboardSource, panelSource, cssSource] = await Promise.all([
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/SelectionOverlapPanel.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(
    dashboardSource,
    /effectivePcpSelectionRanges\(pcpAxisIds,\s*pcpBrushZoom\.ranges,\s*brushes\)/,
    "zoom and working brushes must first become one effective interval per axis",
  );
  assert.match(
    dashboardSource,
    /for \(const \{ axisId, range, source \} of effectivePcpRanges\)[\s\S]*?id: `pcp-axis:\$\{axisId\}`[\s\S]*?candidateMask: analysisMask,[\s\S]*?ranges: \{ \[axisId\]: range \}/,
    "each PCP axis must become its own common-base condition",
  );
  const conditionBlock = dashboardSource.slice(
    dashboardSource.indexOf("const selectionOverlapConditions"),
    dashboardSource.indexOf("const selectionOverlapStages"),
  );
  assert.ok(conditionBlock.indexOf('id: "rank-cluster"') < conditionBlock.indexOf("for (const { axisId"));
  assert.ok(conditionBlock.indexOf("for (const { axisId") < conditionBlock.indexOf('id: "visual-cluster"'));
  assert.ok(conditionBlock.indexOf('id: "visual-cluster"') < conditionBlock.indexOf('id: "projection-brush"'));
  assert.match(
    dashboardSource,
    /const projectionMask = useMemo\(\(\) => \{[\s\S]*?analysisMask\[rowIndex\][\s\S]*?projectionSelection\.xDomain[\s\S]*?projectionSelection\.yDomain/,
    "the projection rectangle must be evaluated independently against the common analysis population",
  );
  assert.match(
    conditionBlock,
    /id: "projection-brush"[\s\S]*?kind: "projection"[\s\S]*?mask: projectionMask/,
    "the embedding rectangle must be an independent overlap condition",
  );
  const stageBlock = dashboardSource.slice(
    dashboardSource.indexOf("const selectionOverlapStages"),
    dashboardSource.indexOf("const selectionRuleAttributes"),
  );
  assert.match(stageBlock, /Diagnostic Top 50/);
  assert.doesNotMatch(stageBlock, /Embedding brush|id: "projection"/);
  assert.match(
    dashboardSource,
    /deferredSelectionOverlapInput\.conditions\.map\(\(condition\) => \(\{[\s\S]*?ruleResult:\s*summarizeSelectionRules\(\{[\s\S]*?selectionMask:\s*condition\.mask,[\s\S]*?baseMask:\s*deferredSelectionOverlapInput\.baseMask,[\s\S]*?attributes:\s*deferredSelectionOverlapInput\.ruleAttributes/,
    "Dashboard must compute every ruleResult from that condition's own mask",
  );
  assert.match(
    dashboardSource,
    /selectionOverlapReady\s*=\s*deferredSelectionOverlapInput\s*===\s*selectionOverlapInput;[\s\S]*?selectionOverlapActive\s*=\s*selectionOverlapReady\s*&&/,
    "a deferred snapshot must not render after its scope/filter input changes",
  );
  const overlapReducerBlock = dashboardSource.slice(
    dashboardSource.indexOf("const selectionOverlapSummary"),
    dashboardSource.indexOf("const selectionOverlapReady"),
  );
  assert.doesNotMatch(
    overlapReducerBlock,
    /ruleAttributes/,
    "the overlap reducer must remain a Node-testable leaf without range computation",
  );
  assert.match(dashboardSource, /<SelectionOverlapPanel[\s\S]*?result=\{selectionOverlapSummary\}/);

  assert.match(panelSource, /function RuleProfile/);
  assert.match(panelSource, /function briefRuleText/);
  const briefRule = panelSource.slice(
    panelSource.indexOf("function briefRuleText("),
    panelSource.indexOf("function RuleProfile("),
  );
  assert.match(briefRule, /rule\.kind === "category"[\s\S]*?return directRuleText\(rule\)/);
  assert.match(briefRule, /rule\.projection\.toUpperCase\(\)\} box/);
  assert.match(briefRule, /≈ \$\{rule\.lower\.toFixed\(3\)\}–\$\{rule\.upper\.toFixed\(3\)\}/);
  assert.doesNotMatch(briefRule, /≤|≥/, "rounded preview endpoints must not be presented as exact bounds");
  const conditionCard = panelSource.slice(
    panelSource.indexOf("function ConditionCard("),
    panelSource.indexOf("function ExactVennDiagram("),
  );
  assert.match(conditionCard, /condition\.label[\s\S]*?condition\.count\.toLocaleString\(\)/);
  assert.match(conditionCard, /selection-overlap-rule-brief[\s\S]*?briefRuleText\(condition\.rule\)/);
  assert.match(
    conditionCard,
    /directRuleText\(condition\.rule\)[\s\S]*?<RuleProfile condition=\{condition\} \/>/,
  );
  assert.match(panelSource, /<details className="selection-overlap-details">[\s\S]*?<summary>Rules &amp; counts<\/summary>[\s\S]*?<ConditionCard/);
  assert.doesNotMatch(panelSource, /<details[^>]*\bopen\b/, "shared rule details should start collapsed");
  assert.match(panelSource, /condition\.ruleResult\.summary\.attributes\.map/);
  assert.match(panelSource, /Direct rule/);
  assert.match(panelSource, /rule\.kind === "projection"/);
  assert.match(panelSource, /rule\.xDomain/);
  assert.match(panelSource, /rule\.yDomain/);
  assert.match(panelSource, /summary\.conditions\.map/);
  assert.match(panelSource, /condition\.cumulativeCount/);
  assert.match(panelSource, /conditionCount > 3/);
  assert.match(panelSource, /selection-overlap-waterfall/);
  assert.match(panelSource, /viewBox=\{viewBox\}/);
  assert.match(panelSource, /aria-hidden="true"/);
  assert.match(panelSource, /focusable="false"/);
  assert.match(panelSource, /className="sr-only"/);
  assert.match(panelSource, /Exact independent filter overlap counts/);
  assert.doesNotMatch(panelSource, /onClick|onChange|onPointer|onMouse|<button\b|tabIndex=/);

  assert.match(
    cssSource,
    /\.selection-overlap-panel\s*\{[\s\S]*?min-width:\s*0;[\s\S]*?overflow:\s*hidden;/,
  );
  assert.match(
    cssSource,
    /\.selection-overlap-venn\s*\{[\s\S]*?height:\s*auto;[\s\S]*?max-width:\s*176px;[\s\S]*?width:\s*100%;/,
  );
  assert.doesNotMatch(cssSource, /\.selection-overlap-venn\s*\{[^}]*min-width:/);
  assert.match(
    cssSource,
    /\.selection-overlap-condition-grid\s*\{[\s\S]*?display:\s*grid;[\s\S]*?minmax\(/,
    "condition explanations must reflow rather than overflow the narrow column",
  );
  assert.match(cssSource, /\.selection-overlap-condition\s*\{[\s\S]*?min-width:\s*0;/);
  assert.match(cssSource, /\.selection-overlap-direct-rule[\s\S]*?overflow-wrap:\s*anywhere;/);
  assert.match(cssSource, /\.selection-overlap-profile\s*\{/);
  assert.match(cssSource, /\.selection-overlap-waterfall\s*\{/);
});
