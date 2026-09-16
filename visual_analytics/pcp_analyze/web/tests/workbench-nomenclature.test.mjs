import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("ProbeScout branding and column names match the paper-facing UI", async () => {
  const [dashboard, layout] = await Promise.all([source("app/Dashboard.tsx"), source("app/layout.tsx")]);
  const title = "ProbeScout: Interactive Retrieval Refinement";
  assert.ok(dashboard.includes(`<h1>${title}</h1>`));
  assert.ok(dashboard.includes(`<p className="eyebrow">${title}</p>`));
  assert.ok(layout.includes(`const title = "${title}"`));
  for (const [id, label] of [
    ["dashboard-heading", "Evidence Analysis"],
    ["visualization-heading", "Image Exploration"],
    ["optimization-heading", "Diagnosis and Refinement"],
    ["test-audit-heading", "Diagnosis and Refinement"],
  ]) assert.ok(dashboard.includes(`<h2 id="${id}">${label}</h2>`));
});

test("module labels omit A–I prefixes and retain Top-K context and compact refinement results", async () => {
  const [dashboard, overlap, tuning, gallery] = await Promise.all([
    source("app/Dashboard.tsx"), source("app/components/SelectionOverlapPanel.tsx"),
    source("app/components/TuningPanel.tsx"), source("app/components/TopGallery.tsx"),
  ]);
  for (const label of [
    "Query Specification", "Hierarchical Evidence", "Visual Embedding Exploration",
    "Ranked Gallery", "Diagnostic Filters", "Selected Image Analysis", "Human Feedback",
    "Refinement and Validation",
  ]) assert.ok(dashboard.includes(`<span>${label}</span>`) || dashboard.includes(`title="${label}"`), label);
  assert.ok(overlap.includes("<strong>Set Overlap Analysis</strong>"));
  assert.doesNotMatch(`${dashboard}\n${overlap}`, /[A-I] · (?:Query Specification|Hierarchical Evidence|Visual Embedding Exploration|Ranked Gallery|Diagnostic Filters|Selected Image Analysis|Set Overlap Analysis|Human Feedback|Refinement and Validation)/);
  assert.ok(tuning.includes("Val AP summary"));
  assert.ok(tuning.includes("Evaluation details"));
  assert.match(gallery, /title=\{`Ranked by \$\{learner\.label\}`\}>Top \{safeLimit\}<\/span>/);
  assert.match(dashboard, /limit=\{topLimit\}/);
  assert.match(dashboard, /<AnnotationReviewGallery/);
  assert.match(dashboard, /<TuningFunctionActions/);
  assert.match(dashboard, /sourceLabel=\{[\s\S]*?Legacy static scores/);
});

test("analysis panels omit visible source descriptions while diagnostics keep source and availability context", async () => {
  const [dashboard, panel] = await Promise.all([
    source("app/Dashboard.tsx"), source("app/components/SmartFilterPanel.tsx"),
  ]);
  assert.doesNotMatch(panel, /<p\b[^>]*>[\s\S]*?Source · \{sourceLabel\}[\s\S]*?<\/p>/);
  assert.match(panel, /title=\{unavailableReason \?\? [^\n]*sourceLabel[^\n]*sourceDetail/);
  assert.match(panel, /disabled = Boolean\(unavailableReason\)/);
  assert.match(panel, /role="alert">Diagnostics unavailable: \{unavailableReason\}/);
  assert.doesNotMatch(dashboard, /<p\b[^>]*>\s*\{ruleMethodLabel\}\s*<\/p>/);
  assert.doesNotMatch(dashboard, /<p\b[^>]*>\s*\{refinementSourceLabel\} · F =/);
  assert.match(dashboard, /sourceDetail=\{activeRefinementVisualization[\s\S]*?Applied snapshot only/);
  assert.match(dashboard, /unavailableReason=\{diagnosticSource\.error\}/);
});

test("diagnostic names change without changing their filter identities", async () => {
  const panel = await source("app/components/SmartFilterPanel.tsx");
  for (const [kind, label] of [
    ["softgate-boundary", "SoftGate Boundary"], ["learner-disagreement", "Probe Disagreement"],
    ["learner-fusion-gap", "Individual-Probe Rescue"], ["prototype-rank-gap", "Prototype–Ranking Mismatch"],
  ]) assert.match(panel, new RegExp(`kind: "${kind}",[\\s\\S]*?title: "${label}"`));
});
