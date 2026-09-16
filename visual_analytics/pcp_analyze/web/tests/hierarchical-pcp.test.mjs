import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  HIERARCHICAL_PCP_EMBEDDING_BASELINES,
  HIERARCHICAL_PCP_LEARNERS,
  HIERARCHICAL_PCP_METHODS,
  REFINEMENT_PCP_DEFAULT_EMBEDDING_METHODS,
  areHierarchicalPcpConfigsEquivalent,
  buildHierarchicalPcpAxes,
  buildRefinementPcpAxes,
  createDefaultHierarchicalPcpConfig,
  hierarchicalPcpConfigFingerprint,
  hierarchicalPcpConfigFromTuningWeights,
  normalizeHierarchicalPcpConfig,
  pruneHierarchicalPcpBrushes,
  refinementPcpComponentIds,
  updateHierarchicalAttributeWeight,
  updateHierarchicalMethodWeight,
} from "../app/lib/hierarchicalPcp.ts";
import { WEIGHTED_FUSION_LEARNERS } from "../app/lib/weightedFusion.ts";

const attributes = [
  { id: "cat", label: "Cat" },
  { id: "hugging", label: "Hugging" },
];

function methodRow(value = 1) {
  return Object.fromEntries(HIERARCHICAL_PCP_METHODS.map((method) => [method, value]));
}

test("the canonical inner method order is five embedding baselines plus eight learners", () => {
  assert.deepEqual([...HIERARCHICAL_PCP_LEARNERS], [...WEIGHTED_FUSION_LEARNERS]);
  assert.deepEqual([...HIERARCHICAL_PCP_EMBEDDING_BASELINES], [
    "Image Prototype",
    "Query MaxSim",
    "Image--Text Fusion",
    "Text Prompt Ensemble",
    "Z-score Image--Text Fusion",
  ]);
  assert.deepEqual([...HIERARCHICAL_PCP_METHODS], [
    ...HIERARCHICAL_PCP_EMBEDDING_BASELINES,
    ...HIERARCHICAL_PCP_LEARNERS,
  ]);
  assert.equal(HIERARCHICAL_PCP_METHODS.length, 13);
});

test("axes form Full -> attribute -> thirteen methods and preserve baseline Joint semantics", () => {
  const collapsed = buildHierarchicalPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: false,
    expandedAttributeIds: new Set(["cat"]),
  });
  assert.deepEqual(collapsed.map((axis) => axis.kind), ["full"]);
  assert.equal(collapsed[0].label, "Weighted Fusion overall");

  const attributesOnly = buildHierarchicalPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: true,
    expandedAttributeIds: [],
  });
  assert.deepEqual(
    attributesOnly.map((axis) => [axis.kind, axis.depth, axis.targetId]),
    [
      ["full", 0, "joint"],
      ["attribute", 1, "cat"],
      ["attribute", 1, "hugging"],
    ],
  );
  assert.equal(attributesOnly.length, 1 + attributes.length);

  const catExpanded = buildHierarchicalPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: true,
    expandedAttributeIds: ["cat"],
  });
  const catMethods = catExpanded.slice(2, 2 + HIERARCHICAL_PCP_METHODS.length);
  assert.equal(catExpanded.length, 1 + attributes.length + HIERARCHICAL_PCP_METHODS.length);
  assert.deepEqual(catMethods.map((axis) => axis.methodId), [...HIERARCHICAL_PCP_METHODS]);
  assert.deepEqual(
    catMethods.map((axis) => axis.kind),
    [
      ...HIERARCHICAL_PCP_EMBEDDING_BASELINES.map(() => "baseline"),
      ...HIERARCHICAL_PCP_LEARNERS.map(() => "learner"),
    ],
  );
  assert.ok(catMethods.every((axis) => axis.depth === 2 && axis.attributeId === "cat"));
  assert.ok(
    catMethods.slice(0, HIERARCHICAL_PCP_EMBEDDING_BASELINES.length)
      .every((axis) => axis.targetId === "joint"),
  );
  assert.ok(
    catMethods.slice(HIERARCHICAL_PCP_EMBEDDING_BASELINES.length)
      .every((axis) => axis.targetId === "cat"),
  );
  assert.ok(catMethods.every((axis) => axis.id.includes("cat")));
  assert.equal(catExpanded.at(-1).attributeId, "hugging");

  const complete = buildHierarchicalPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: true,
    expandedAttributeIds: attributes.map((attribute) => attribute.id),
  });
  assert.equal(complete.length, 1 + 14 * attributes.length);
  assert.equal(complete.length, 29);
  assert.equal(new Set(complete.map((axis) => axis.id)).size, complete.length);
});

test("refinement axes preserve Overall -> C -> attributes/probes and sibling H/embeddings", () => {
  const ids = attributes.map((attribute) => attribute.id);
  const complete = buildRefinementPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: true,
    expandedAttributeIds: ids,
    holisticExpanded: true,
  });

  assert.equal(complete[0].kind, "full");
  assert.equal(complete[0].targetId, "joint");
  assert.equal(complete[0].label, "Overall");
  assert.deepEqual(complete[1], {
    id: "refinement::attribute-evidence::joint",
    kind: "attribute-evidence", depth: 1, label: "Attribute evidence",
    description: "Attribute evidence C = ∏ gₐ^γₐ; conjunctive support from all required attributes.",
    targetId: "joint",
  });
  const attributeAxes = complete.filter((axis) => axis.kind === "attribute");
  const learnerAxes = complete.filter((axis) => axis.kind === "learner");
  const holisticAxes = complete.filter((axis) => axis.kind === "holistic");
  const embeddingAxes = complete.filter((axis) => axis.kind === "baseline");
  assert.deepEqual(attributeAxes.map((axis) => axis.attributeId), ids);
  assert.equal(learnerAxes.length, ids.length * HIERARCHICAL_PCP_LEARNERS.length);
  assert.equal(holisticAxes.length, 1);
  assert.equal(embeddingAxes.length, REFINEMENT_PCP_DEFAULT_EMBEDDING_METHODS.length);
  assert.ok(attributeAxes.every((axis) => axis.depth === 2));
  assert.ok(learnerAxes.every((axis) => axis.depth === 3));
  assert.equal(holisticAxes[0].depth, 1);
  assert.equal(holisticAxes[0].label, "Embedding evidence");
  assert.ok(embeddingAxes.every((axis) => axis.depth === 2));
  assert.ok(
    learnerAxes.every((axis) => axis.attributeId && axis.targetId === axis.attributeId),
    "each beta member must remain inside its own attribute",
  );
  assert.ok(
    embeddingAxes.every((axis) => axis.attributeId === undefined && axis.targetId === "joint"),
    "eta members form one global embedding branch and must not be copied per attribute",
  );
  assert.equal(
    complete.length,
    3 + ids.length * (1 + HIERARCHICAL_PCP_LEARNERS.length)
      + REFINEMENT_PCP_DEFAULT_EMBEDDING_METHODS.length,
  );
  assert.deepEqual(
    refinementPcpComponentIds(ids, "joint"),
    complete.map((axis) => axis.id),
    "the backend snapshot column contract must match every fully expanded PCP axis exactly",
  );

  const collapsed = buildRefinementPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: false,
    expandedAttributeIds: ids,
    holisticExpanded: true,
  });
  assert.deepEqual(collapsed.map((axis) => axis.kind), ["full"]);

  const branchOptions = {
    attributes, jointTargetId: "joint", fullExpanded: true,
    expandedAttributeIds: ids, attributeEvidenceExpanded: false, holisticExpanded: true,
  };
  const hiddenAttributes = buildRefinementPcpAxes(branchOptions);
  assert.deepEqual(hiddenAttributes.map(({ kind }) => kind), ["full", "attribute-evidence", "holistic", "baseline", "baseline"]);
  const branchesOnly = buildRefinementPcpAxes({ ...branchOptions, holisticExpanded: false });
  assert.deepEqual(branchesOnly.map(({ kind }) => kind), ["full", "attribute-evidence", "holistic"]);
  const restoredAttributes = buildRefinementPcpAxes({ ...branchOptions, attributeEvidenceExpanded: true });
  assert.deepEqual(restoredAttributes, complete, "independent branch collapse preserves individual attribute expansion");
});

test("two-channel refinement expands only its run-pinned holistic methods without changing static baselines", () => {
  const embeddingMethods = ["Query MaxSim", "Text Prompt Ensemble"];
  const ids = attributes.map(({ id }) => id);
  const options = {
    attributes, jointTargetId: "joint", fullExpanded: true,
    expandedAttributeIds: ids, holisticExpanded: true, embeddingMethods,
  };
  const axes = buildRefinementPcpAxes(options);
  assert.equal(axes.length, 23);
  assert.deepEqual(axes.filter(({ kind }) => kind === "baseline").map(({ methodId }) => methodId), embeddingMethods);
  assert.equal(axes.filter(({ kind }) => kind === "learner").length, 16);
  assert.deepEqual(refinementPcpComponentIds(ids, "joint", embeddingMethods), axes.map(({ id }) => id));
  assert.equal(refinementPcpComponentIds(ids, "joint").length, 23, "new layout defaults to two embedding heads");
  assert.equal(refinementPcpComponentIds(ids, "joint", HIERARCHICAL_PCP_EMBEDDING_BASELINES).length, 26, "explicit legacy five-channel layout stays available");
  assert.equal(HIERARCHICAL_PCP_EMBEDDING_BASELINES.length, 5);
  assert.equal(HIERARCHICAL_PCP_METHODS.length, 13);
  assert.throws(() => buildRefinementPcpAxes({ ...options, embeddingMethods: [] }), /unique supported methods/);
  assert.throws(() => buildRefinementPcpAxes({ ...options, embeddingMethods: ["Query MaxSim", "Query MaxSim"] }), /unique supported methods/);
});

test("default raw weights are all one and normalize on both simplex levels", () => {
  const ids = attributes.map((attribute) => attribute.id);
  const defaults = createDefaultHierarchicalPcpConfig(ids);
  assert.deepEqual(defaults.attributeWeights, { cat: 1, hugging: 1 });
  for (const id of ids) {
    for (const method of HIERARCHICAL_PCP_METHODS) {
      assert.equal(defaults.methodWeightsByAttribute[id][method], 1);
    }
  }

  const normalized = normalizeHierarchicalPcpConfig(defaults, ids);
  assert.deepEqual(normalized.attributeWeights, { cat: 0.5, hugging: 0.5 });
  for (const id of ids) {
    for (const method of HIERARCHICAL_PCP_METHODS) {
      assert.equal(normalized.methodWeightsByAttribute[id][method], 1 / 13);
    }
  }

  const scaled = createDefaultHierarchicalPcpConfig(ids, 7);
  assert.equal(
    hierarchicalPcpConfigFingerprint(defaults, ids),
    hierarchicalPcpConfigFingerprint(scaled, ids),
  );
  assert.equal(areHierarchicalPcpConfigsEquivalent(defaults, scaled, ids), true);
});

test("Tune conversion requires complete thirteen-method rows and honors outer weights", () => {
  const ids = attributes.map((attribute) => attribute.id);
  const cat = Object.fromEntries(
    HIERARCHICAL_PCP_METHODS.map((method, index) => [method, index + 1]),
  );
  const config = hierarchicalPcpConfigFromTuningWeights(
    { cat, hugging: methodRow() },
    ids,
    { cat: 2, hugging: 6 },
  );
  assert.deepEqual(config.attributeWeights, { cat: 0.25, hugging: 0.75 });
  const catTotal = HIERARCHICAL_PCP_METHODS.reduce(
    (sum, _method, index) => sum + index + 1,
    0,
  );
  HIERARCHICAL_PCP_METHODS.forEach((method, index) => {
    assert.equal(config.methodWeightsByAttribute.cat[method], (index + 1) / catTotal);
    assert.equal(config.methodWeightsByAttribute.hugging[method], 1 / 13);
  });

  const oldLearnerOnly = Object.fromEntries(
    HIERARCHICAL_PCP_LEARNERS.map((method) => [method, 1]),
  );
  assert.throws(
    () => hierarchicalPcpConfigFromTuningWeights(
      { cat: oldLearnerOnly, hugging: oldLearnerOnly },
      ids,
    ),
    /tuning method weights are missing: Image Prototype/i,
  );
  assert.throws(
    () => hierarchicalPcpConfigFromTuningWeights(
      { cat: { ...methodRow(), Unknown: 1 }, hugging: methodRow() },
      ids,
    ),
    /contain unknown entries: Unknown/i,
  );
});

test("attribute and inner-method updates are immutable and fingerprinted", () => {
  const ids = attributes.map((attribute) => attribute.id);
  const defaults = createDefaultHierarchicalPcpConfig(ids);
  const changedAttribute = updateHierarchicalAttributeWeight(defaults, "cat", 3);
  const changedMethod = updateHierarchicalMethodWeight(
    changedAttribute,
    "hugging",
    "MLP",
    0,
  );

  assert.equal(defaults.attributeWeights.cat, 1);
  assert.equal(defaults.methodWeightsByAttribute.hugging.MLP, 1);
  assert.equal(changedMethod.attributeWeights.cat, 3);
  assert.equal(changedMethod.methodWeightsByAttribute.hugging.MLP, 0);
  assert.equal(changedMethod.methodWeightsByAttribute.cat, defaults.methodWeightsByAttribute.cat);
  assert.notEqual(
    hierarchicalPcpConfigFingerprint(defaults, ids),
    hierarchicalPcpConfigFingerprint(changedMethod, ids),
  );

  const normalized = normalizeHierarchicalPcpConfig(changedMethod, ids);
  assert.equal(normalized.attributeWeights.cat, 0.75);
  assert.equal(normalized.attributeWeights.hugging, 0.25);
  assert.equal(normalized.methodWeightsByAttribute.hugging.MLP, 0);
  assert.equal(normalized.methodWeightsByAttribute.hugging["K-Fold"], 1 / 12);
});

test("invalid, incomplete, unknown, and all-zero weight groups are rejected", () => {
  const ids = attributes.map((attribute) => attribute.id);
  const defaults = createDefaultHierarchicalPcpConfig(ids);
  assert.throws(
    () => normalizeHierarchicalPcpConfig({
      ...defaults,
      attributeWeights: { cat: 0, hugging: 0 },
    }, ids),
    /At least one attribute weight/,
  );
  assert.throws(
    () => normalizeHierarchicalPcpConfig({
      ...defaults,
      methodWeightsByAttribute: {
        ...defaults.methodWeightsByAttribute,
        cat: methodRow(0),
      },
    }, ids),
    /At least one method weight for cat/,
  );

  const incomplete = methodRow();
  delete incomplete["Image Prototype"];
  assert.throws(
    () => normalizeHierarchicalPcpConfig({
      ...defaults,
      methodWeightsByAttribute: {
        ...defaults.methodWeightsByAttribute,
        cat: incomplete,
      },
    }, ids),
    /cat method weights are missing: Image Prototype/i,
  );
  assert.throws(
    () => normalizeHierarchicalPcpConfig({
      ...defaults,
      methodWeightsByAttribute: {
        ...defaults.methodWeightsByAttribute,
        cat: { ...methodRow(), Unknown: 1 },
      },
    }, ids),
    /cat method weights contain unknown entries: Unknown/i,
  );
});

test("hidden-axis brushes are removed after hierarchy disclosure changes", () => {
  const visibleAxes = buildHierarchicalPcpAxes({
    attributes,
    jointTargetId: "joint",
    fullExpanded: true,
    expandedAttributeIds: [],
  });
  assert.deepEqual(
    pruneHierarchicalPcpBrushes({
      [visibleAxes[0].id]: [0.4, 1],
      "hierarchical::method::cat::MLP": [0.8, 1],
    }, visibleAxes),
    { [visibleAxes[0].id]: [0.4, 1] },
  );
});

test("hierarchy rail exposes root and attribute disclosures plus both weight levels", async () => {
  const source = await readFile(
    new URL("../app/components/HierarchicalPcpRail.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /axis\.kind === "full" \|\| axis\.kind === "attribute-evidence" \|\| axis\.kind === "attribute"/);
  assert.match(source, /onFullExpandedChange\(!fullExpanded\)/);
  assert.match(source, /onAttributeExpandedChange\(attributeId, !expanded\.has\(attributeId\)\)/);
  assert.match(source, /onAttributeWeightChange\(attributeId, nextWeight\)/);
  assert.match(source, /onMethodWeightChange\(attributeId, methodId, nextWeight\)/);
  assert.match(source, /config\.methodWeightsByAttribute\[attributeId\]\?\.\[methodId\]/);
  assert.match(source, /uses the canonical Joint ranking/);
  assert.doesNotMatch(source, /oursExpanded|onGlobalWeightChange|onLearnerWeightChange/);
  assert.match(source, /HIERARCHICAL_PCP_TOP \+ index \* rowGap/);
  assert.match(source, /variant\?: HierarchicalPcpRailVariant/);
  assert.match(source, /variant = "aligned"/);
  assert.match(source, /summaryVariant \? undefined : \{ height \}/);
  assert.match(source, /style=\{summaryVariant[\s\S]*?\? undefined[\s\S]*?: \{ top:/);
});

test("refined rail gives C its own disclosure and badge, with compact named weight readouts", async () => {
  const source = await readFile(new URL("../app/components/HierarchicalPcpRail.tsx", import.meta.url), "utf8");
  assert.match(source, /attributeEvidenceExpanded\?: boolean/);
  assert.match(source, /onAttributeEvidenceExpandedChange\?\.\(!attributeEvidenceExpanded\)/);
  assert.match(source, /onHolisticExpandedChange\?\.\(!holisticExpanded\)/);
  assert.match(source, /data-refinement=\{Boolean\(refinementWeights\)\}/);
  assert.match(source, /axis\.kind === "attribute-evidence" \? \([\s\S]*?hierarchical-pcp-evidence-badge[\s\S]*?>C<\/span>[\s\S]*?\) : refinementWeights \? \(/);
  assert.match(source, /<output[\s\S]*?hierarchical-pcp-weight-readout/);
  assert.match(source, /axis\.kind === "attribute" \? "γ" : axis\.kind === "holistic" \? "λ" : axis\.kind === "learner" \? "β" : "η"/);
  assert.match(source, /weight!\.toFixed\(3\)/);
  assert.match(source, /\(weight! \* 100\)\.toFixed\(1\)/);
  assert.match(source, /title=\{axis\.description \?\? axis\.label\}/);
});

test("hierarchy rail styles distinguish outer attributes and indent both method types", async () => {
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  assert.match(css, /\.hierarchical-pcp-row--attribute\s*\{/);
  assert.match(css, /\.hierarchical-pcp-row--baseline\s*\{/);
  assert.match(css, /\.hierarchical-pcp-row--learner\s*\{/);
  assert.match(
    css,
    /\.hierarchical-pcp-row--baseline\s*\{[^}]*padding-left:\s*13px;/s,
  );
  assert.match(
    css,
    /\.hierarchical-pcp-row--learner\s*\{[^}]*padding-left:\s*13px;/s,
  );
  assert.doesNotMatch(css, /\.hierarchical-pcp-row--ours\s*\{/);
});

test("cluster summaries align the same hierarchy rail with their PCP axes", async () => {
  const dashboard = await readFile(
    new URL("../app/Dashboard.tsx", import.meta.url),
    "utf8",
  );
  const summary = await readFile(
    new URL("../app/components/ClusterSummaryParallelCoordinates.tsx", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  assert.match(dashboard, /className="cluster-summary-pcp"/);
  assert.doesNotMatch(dashboard, /variant="summary"/);
  assert.match(dashboard, /showAxisLabels=\{false\}/);
  assert.match(dashboard, /axisRail=\{\([\s\S]*?<HierarchicalPcpRail[\s\S]*?height=\{summaryPcpHeight\}/);
  assert.match(summary, /cluster-summary-chart-stage--with-rail/);
  assert.match(summary, /ref=\{chartPaneRef\} className="cluster-summary-chart-pane"/);
  assert.match(summary, /const top = hasAxisRail \? HIERARCHICAL_PCP_TOP : 24/);
  assert.match(summary, /viewport\.height - \(hasAxisRail \? HIERARCHICAL_PCP_BOTTOM_PADDING : 34\)/);
  assert.match(summary, /\{showAxisLabels && \([\s\S]*?truncateLabel\(label\)/);
  assert.match(styles, /\.cluster-summary-chart-stage--with-rail \.cluster-summary-chart-pane \{[\s\S]*?margin-left: var\(--learner-rail-width\);[\s\S]*?width: calc\(100% - var\(--learner-rail-width\)\);/);
});

test("the floating PCP status does not block the first axis brush", async () => {
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(
    css,
    /\.pcp-toolbar\s*\{[^}]*pointer-events:\s*none;/s,
    "the toolbar wrapper must pass drag events through to the first PCP axis",
  );
  assert.match(
    css,
    /\.pcp-clear-brushes\s*\{[^}]*pointer-events:\s*auto;/s,
    "the Clear brushes button must remain interactive",
  );
});
