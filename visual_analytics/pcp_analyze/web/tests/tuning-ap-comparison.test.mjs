import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  apDeltaTone, formatAp, formatApDelta, tuningApRows,
} from "../app/lib/tuningEvaluation.ts";

const run = (changes = {}) => ({
  id: "run-1", sessionId: "session-1", mode: "weight_joint", status: "succeeded",
  baseMethod: "Ours-Full", beforeMethod: "F0", createdAt: "2026-09-05T15:00:00Z",
  evaluationScope: "vqa-validation", before: { ap: 0.9 }, after: { ap: 0.95 },
  testEvaluation: { before: { ap: 0.4 }, after: { ap: 0.52 }, targetId: "joint", beforeMethod: "F0" },
  ...changes,
});

test("Val and Test use independent recorded metrics, not the same delta", () => {
  const [val, frozenTest] = tuningApRows(run({ deltaAp: 0.99 }));
  assert.equal(val.label, "Val");
  assert.equal(frozenTest.label, "Test");
  assert.equal(formatAp(val.before), "90.00%");
  assert.equal(formatAp(frozenTest.before), "40.00%");
  assert.equal(formatApDelta(val.delta), "+5.00");
  assert.equal(formatApDelta(frozenTest.delta), "+12.00");
});

test("both schedules and Original/Updated sources retain separate Test evaluation", () => {
  for (const mode of ["weight_staged", "weight_joint"]) {
    for (const probeSource of ["original", "updated"]) {
      const input = run({ mode, probeSource });
      const original = structuredClone(input);
      assert.equal(tuningApRows(input)[1].after, 0.52);
      assert.deepEqual(input, original, "display must not mutate model/session state");
    }
  }
});

test("history without Test metrics shows unrecorded, not zero or a Val fallback", () => {
  const row = tuningApRows(run({ testEvaluation: undefined }))[1];
  assert.equal(row.note, "Not recorded");
  assert.equal(formatAp(row.before), "—");
  assert.equal(formatAp(row.after), "—");
  assert.equal(formatApDelta(row.delta), "—");
});

test("unavailable Test does not hide a valid Val result", () => {
  const rows = tuningApRows(run({ testEvaluation: undefined, testEvaluationError: "No positive Test rows" }));
  assert.equal(rows[0].after, 0.95);
  assert.equal(rows[1].note, "Unavailable");
  assert.equal(tuningApRows(run({ testEvaluation: {} }))[1].note, "Not recorded");
});

test("zero AP is valid; non-finite and out-of-range APs are unavailable", () => {
  const rows = tuningApRows(run({ testEvaluation: { before: { ap: 0 }, after: { ap: 0 } } }));
  assert.equal(formatAp(rows[1].before), "0.00%");
  assert.equal(formatApDelta(rows[1].delta), "0.00");
  for (const invalid of [undefined, null, NaN, Infinity, -0.1, 1.1]) {
    assert.equal(formatAp(invalid), "—");
  }
});

test("negative, positive and unchanged AP differences have accurate pp formatting", () => {
  assert.equal(formatApDelta(-0.125), "-12.50");
  assert.equal(apDeltaTone(-0.125), "negative");
  assert.equal(apDeltaTone(0.125), "positive");
  assert.equal(apDeltaTone(0), "neutral");
  assert.equal(apDeltaTone(-1e-9), "neutral");
  assert.equal(formatApDelta(-1e-9), "0.00");
});

test("legacy evaluation scopes keep their actual identity", () => {
  for (const [scope, label] of [["clean-validation", "Clean Val (legacy)"],
    ["probe-validation", "Probe Val (legacy)"], ["validation", "Web Val (legacy)"]]) {
    assert.equal(tuningApRows(run({ evaluationScope: scope }))[0].label, label);
  }
  const rows = tuningApRows(run({ evaluationScope: "test", testEvaluation: undefined }));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].label, "Test (legacy)");
});

test("Development selects only known Validation scopes without touching secondary Test metrics", () => {
  for (const scope of ["vqa-validation", "clean-validation", "probe-validation", "validation"]) {
    const input = run({ evaluationScope: scope });
    Object.defineProperty(input, "testEvaluation", { get() { throw new Error("Test must not be read"); } });
    const rows = tuningApRows(input, "validation");
    assert.equal(rows.length, 1);
    assert.equal(rows[0].after, .95);
  }
  for (const scope of ["test", undefined, "unrecognized"]) {
    assert.deepEqual(tuningApRows(run({ evaluationScope: scope }), "validation"), []);
  }
});

test("Test-only display retains its own values and has no Val fallback", () => {
  assert.equal(tuningApRows(run(), "test")[0].after, .52);
  assert.equal(tuningApRows(run({ testEvaluation: undefined }), "test")[0].after, undefined);
  assert.equal(tuningApRows(run({ evaluationScope: "test" }), "test")[0].after, .95);
});

test("Development component uses the Validation guard for AP, F1 and TP; Test panel is mounted only in audit view", async () => {
  const source = await readFile(new URL("../app/components/TuningPanel.tsx", import.meta.url), "utf8");
  const dashboard = await readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8");
  assert.match(source, /tuningApRows\(run, "validation"\)/);
  assert.doesNotMatch(source, /testEvaluation|Val \/ Test|Test ·/);
  const start = source.indexOf('{validationMetricsAvailable ? <>');
  const end = source.indexOf('<details className="tuning-run-details">', start);
  assert.ok(start >= 0 && end > start);
  const block = source.slice(start, end);
  for (const marker of ["apRows.map", "Best F1", "TP@K"]) assert.ok(block.includes(marker));
  const testView = dashboard.indexOf('<section className="workspace-panel test-audit-panel"');
  assert.ok(testView > dashboard.indexOf('{developmentMode ? ('));
  assert.equal((dashboard.match(/<FrozenTestEvaluation/g) ?? []).length, 1);
  assert.ok(dashboard.indexOf('<FrozenTestEvaluation') > testView);
  assert.match(dashboard, /testMode \? " \/ 6 Test" : ""/);
});
