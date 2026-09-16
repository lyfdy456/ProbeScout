import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { methodDisplayLabel } from "../app/lib/methodDisplay.js";
import {
  HIERARCHICAL_PCP_LEARNERS,
  buildHierarchicalPcpAxes,
  buildRefinementPcpAxes,
  createDefaultHierarchicalPcpConfig,
} from "../app/lib/hierarchicalPcp.ts";

test("PURA is a display alias, not a change to canonical method identities", () => {
  assert.equal(methodDisplayLabel("Ours-PURA"), "PURA");
  for (const method of ["PURA", "Ours-Full", "nnPU", "Ours-PURA-extra"]) {
    assert.equal(methodDisplayLabel(method), method);
  }
  assert.ok(HIERARCHICAL_PCP_LEARNERS.includes("Ours-PURA"));
  assert.ok(!HIERARCHICAL_PCP_LEARNERS.includes("PURA"));
  const config = createDefaultHierarchicalPcpConfig(["cat"]);
  assert.equal(config.methodWeightsByAttribute.cat["Ours-PURA"], 1);
  assert.equal(config.methodWeightsByAttribute.cat.PURA, undefined);
});

test("legacy and current PCP show PURA while retaining stable IDs and score lookup keys", () => {
  const input = {
    attributes: [{ id: "cat", label: "Cat" }],
    jointTargetId: "joint",
    fullExpanded: true,
    expandedAttributeIds: ["cat"],
    holisticExpanded: true,
  };
  for (const axes of [buildHierarchicalPcpAxes(input), buildRefinementPcpAxes(input)]) {
    const axis = axes.find((item) => item.methodId === "Ours-PURA");
    assert.ok(axis);
    assert.equal(axis.label, "PURA");
    assert.equal(axis.id, "hierarchical::method::cat::Ours-PURA");
    assert.equal(axis.attributeId, "cat");
    assert.equal(axis.targetId, "cat");
    assert.ok(!axis.description?.includes("Ours-PURA"));
  }
});

test("diagnostic learner selection submits the original ID while displaying the alias", async () => {
  const source = await readFile(new URL("../app/components/SmartFilterPanel.tsx", import.meta.url), "utf8");
  assert.match(source, /<option key=\{method\} value=\{method\}>\{methodDisplayLabel\(method\)\}<\/option>/);
  assert.match(source, /onLearnerMethodChange\(event\.target\.value\)/);
});
