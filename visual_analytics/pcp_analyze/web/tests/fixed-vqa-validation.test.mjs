import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  fixedVqaValidationMask,
  developmentWithoutFixedVal,
  fixedValAllowsFeedback,
} from "../app/lib/fixedVqaValidation.ts";
import { fetchFixedVqaValidation } from "../app/lib/tuningApi.ts";

const payload = {
  taskId: "task-a", version: "vqa-val-v1", manifestSha256: "a".repeat(64),
  rowIndices: [1, 4], count: 2, protocol: "fixed-vqa-joint-seed0-holdout-v1",
  labelSource: "original-vqa-supervision", initialModelHoldoutIndependent: false,
};
const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("fixed Val validates provenance, exact count, unique task-aligned rows", () => {
  assert.deepEqual([...fixedVqaValidationMask(payload, "task-a", 6)], [0, 1, 0, 0, 1, 0]);
  for (const change of [
    { taskId: "task-b" }, { taskId: undefined }, { version: "" }, { manifestSha256: "bad" },
    { protocol: "" }, { protocol: "different-protocol" }, { referenceOnly: false },
    { labelSource: "ground-truth" }, { initialModelHoldoutIndependent: true },
    { count: 0, rowIndices: [] }, { count: 1 }, { rowIndices: [1, 1] },
    { rowIndices: [-1, 4] }, { rowIndices: [1, 6] }, { rowIndices: [1.5, 4] },
  ]) assert.throws(() => fixedVqaValidationMask({ ...payload, ...change }, "task-a", 6));
});

test("DG subtracts fixed Val without reclaiming old Web Validation or changing Test", () => {
  const development = Uint8Array.from([1, 1, 0, 0, 1, 0]);
  const validation = fixedVqaValidationMask(payload, "task-a", 6);
  const testMask = Uint8Array.from([0, 0, 0, 1, 0, 0]);
  assert.deepEqual([...developmentWithoutFixedVal(development, validation)], [1, 0, 0, 0, 0, 0]);
  assert.deepEqual([...development], [1, 1, 0, 0, 1, 0]);
  assert.deepEqual([...testMask], [0, 0, 0, 1, 0, 0]);
  assert.deepEqual([...developmentWithoutFixedVal(development, null)], [0, 0, 0, 0, 0, 0]);
  assert.throws(() => developmentWithoutFixedVal(development, new Uint8Array(2)));
});

test("all feedback modes fail closed on protected, pending, or misaligned rows", () => {
  const imageIds = ["a", "b", "c", "d", "e", "f"];
  const development = Uint8Array.from([1, 1, 0, 0, 1, 0]);
  const validation = fixedVqaValidationMask(payload, "task-a", 6);
  assert.equal(fixedValAllowsFeedback({ rowIndex: 0, id: "a" }, imageIds, development, validation), true);
  for (const item of [{ rowIndex: 1, id: "b" }, { rowIndex: 4, id: "e" },
    { rowIndex: 2, id: "c" }, { rowIndex: 3, id: "d" }, { rowIndex: -1, id: "a" },
    { rowIndex: 0, id: "wrong" }, { rowIndex: 0.5, id: "a" }]) {
    assert.equal(fixedValAllowsFeedback(item, imageIds, development, validation), false);
  }
  assert.equal(fixedValAllowsFeedback({ rowIndex: 0, id: "a" }, imageIds, development, null), false);
});

test("fixed Val API is task-wide, abortable, private, and uncached", async () => {
  const previousFetch = globalThis.fetch;
  const controller = new AbortController();
  try {
    globalThis.fetch = async (url, init) => {
      assert.equal(url, "/api/tuning/tasks/task%2Fa/validation");
      assert.equal(init.credentials, "same-origin");
      assert.equal(init.signal, controller.signal);
      assert.equal(init.cache, "no-store");
      return { ok: true, json: async () => payload };
    };
    assert.deepEqual(await fetchFixedVqaValidation("task/a", controller.signal), payload);
  } finally { globalThis.fetch = previousFetch; }
});

test("task changes and Reload invalidate Val before effects while late requests cannot commit", async () => {
  const hook = await source("app/lib/useFixedVqaValidation.ts");
  assert.match(hook, /if \(!active \|\| controller\.signal\.aborted\) return;/);
  assert.match(hook, /active = false;[\s\S]*?controller\.abort\(\)/);
  assert.match(hook, /state\?\.taskId !== taskId \|\| state\.rowCount !== rowCount \|\| state\.reloadKey !== reloadKey/);
  assert.match(hook, /status: "loading", mask: null/);
  assert.doesNotMatch(hook, /targetId|groundTruth/);
});

test("current UI exposes only Val results and preserves legacy scopes in closed history", async () => {
  const [dashboard, panel] = await Promise.all([source("app/Dashboard.tsx"), source("app/components/TuningPanel.tsx")]);
  assert.match(dashboard, /scopes\.filter\(\(scope\) => scope\.id !== "validation"\)/);
  assert.match(dashboard, /if \(scope !== "development" && scope !== "test"\) return/);
  assert.match(dashboard, /developmentMask: developmentWithoutFixedVal\(dataset\.developmentMask, fixedValidation\.mask\)/);
  assert.match(panel, /vqaValidation = run\?\.evaluationScope === "vqa-validation"/);
  assert.match(panel, /className="tuning-result-overview" key=\{run\.id\}/);
  assert.match(panel, /<details className="tuning-evaluation-result">/);
  assert.match(panel, /Historical evaluation · original scope/);
  assert.match(panel, /vqaValidation[\s\S]*?\? "Val"/);
  assert.match(panel, /Clean Validation/);
  assert.match(panel, /legacy Web Validation/);
  assert.match(panel, /not an independent holdout for the original model/);
  assert.match(panel, /runValidationManifestSha256/);
});

test("single, bulk, review, failure attribution, and queued writes enforce fixed Val protection", async () => {
  const [dashboard, hook, review] = await Promise.all([
    source("app/Dashboard.tsx"), source("app/lib/useTuningSession.ts"), source("app/components/AnnotationReviewGallery.tsx"),
  ]);
  assert.match(dashboard, /if \(!items\.every\(canAnnotate\)\) return/);
  assert.match(hook, /assertFeedbackAllowed\(item, session\.id, label === "unmarked"\)/);
  assert.match(hook, /assertFeedbackAllowed\(item, sessionId, label === "unmarked"\)/);
  assert.match(hook, /for \(const item of uniqueItems\) assertFeedbackAllowed\(item, sessionId\)/);
  assert.match(hook, /const updateFailureAttributes[\s\S]*?assertFeedbackAllowed\(item, session\.id\)/);
  assert.match(hook, /removal \? context\.input\.canRemoveFeedback\(item\) : context\.input\.canWriteFeedback\(item\)/);
  assert.match(hook, /context\.session\.taskId !== context\.input\.taskId/);
  assert.match(hook, /includedInTune !== false/);
  assert.match(review, /Val · excluded/);
  assert.match(review, /canAnnotate\?\.\(item\) === false && canRemoveAnnotation\?\.\(item\)/);
  assert.match(review, /changePreference\(item, "unmarked"\)/);
  assert.match(review, /return item \? \{ annotation, item \} : null/);
});
