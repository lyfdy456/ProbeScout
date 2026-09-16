import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  MAX_BULK_TUNING_ANNOTATIONS,
  putTuningAnnotationsBulk,
} from "../app/lib/tuningApi.ts";

async function source(relativePath) {
  return readFile(new URL(`../${relativePath}`, import.meta.url), "utf8");
}

const EMPTY_COUNTS = {
  strongNegative: 0,
  negative: 0,
  uncertain: 0,
  positive: 0,
  strongPositive: 0,
  usablePositive: 0,
  usableNegative: 0,
};

test("bulk feedback API submits one session-owned atomic request", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  const confirmed = [
    {
      rowIndex: 4,
      imageId: "image-4",
      label: 1,
      source: "selection-bulk",
      updatedAt: "2026-08-18T00:00:00Z",
    },
    {
      rowIndex: 9,
      imageId: "image-9",
      label: 1,
      source: "selection-bulk",
      updatedAt: "2026-08-18T00:00:00Z",
    },
  ];
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init });
    return new Response(JSON.stringify({ annotations: confirmed, counts: EMPTY_COUNTS }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const response = await putTuningAnnotationsBulk(
      "session/private",
      confirmed.map(({ rowIndex, imageId, label }) => ({ rowIndex, imageId, label })),
      "selection-bulk",
    );
    assert.deepEqual(response.annotations, confirmed);
    assert.equal(requests.length, 1, "one selection must not become N per-row writes");
    assert.equal(requests[0].url, "/api/tuning/sessions/session%2Fprivate/annotations/bulk");
    assert.equal(requests[0].init.method, "POST");
    assert.equal(requests[0].init.credentials, "same-origin");
    assert.deepEqual(JSON.parse(requests[0].init.body), {
      source: "selection-bulk",
      annotations: [
        { rowIndex: 4, imageId: "image-4", label: 1 },
        { rowIndex: 9, imageId: "image-9", label: 1 },
      ],
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("bulk feedback rejects empty and over-limit batches before network I/O", () => {
  assert.equal(MAX_BULK_TUNING_ANNOTATIONS, 1_000);
  assert.throws(
    () => putTuningAnnotationsBulk("session", []),
    /at least one image/i,
  );
  const oversized = Array.from(
    { length: MAX_BULK_TUNING_ANNOTATIONS + 1 },
    (_, rowIndex) => ({ rowIndex, imageId: `image-${rowIndex}`, label: -1 }),
  );
  assert.throws(
    () => putTuningAnnotationsBulk("session", oversized),
    /limited to 1,000 images/i,
  );
});

test("bulk feedback hook flushes pending writes and commits one confirmed revision", async () => {
  const hookSource = await source("app/lib/useTuningSession.ts");
  const block = hookSource.match(
    /const bulkUpdateAnnotations = useCallback[\s\S]*?\n  const updateFailureAttributes/,
  )?.[0] ?? "";

  assert.match(hookSource, /bulkUpdateAnnotations:\s*\(/);
  assert.match(block, /sessionMutationLockRef\.current = operationToken/);
  const waitAt = block.indexOf("await waitForPendingAnnotations(generation, sessionId)");
  const requestAt = block.indexOf("await putTuningAnnotationsBulk(");
  assert.ok(waitAt >= 0 && requestAt > waitAt, "bulk POST must wait for every earlier row write");
  assert.match(
    block.slice(requestAt),
    /if \(!isActiveSessionOperation\(generation, sessionId\)\) return;/,
    "a response from a replaced task/session must be quarantined",
  );
  assert.match(block, /const nextConfirmed = new Map\(previousConfirmed\)/);
  assert.match(block, /confirmedAnnotationsRef\.current = nextConfirmed;/);
  assert.match(block, /setAnnotations\(new Map\(nextConfirmed\)\)/);
  assert.equal(
    block.match(/advanceAnnotationRevision\(\)/g)?.length,
    1,
    "one atomic batch may invalidate a completed run at most once",
  );
  assert.match(block, /if \(revisionChanged\) advanceAnnotationRevision\(\)/);
  assert.match(
    block,
    /catch \(reason\)[\s\S]*?setAnnotations\(new Map\(confirmedAnnotationsRef\.current\)\)[\s\S]*?setError/,
    "a failed request must restore confirmed browser state and expose the error",
  );
  assert.match(block, /busyOperationRef\.current = operationToken/);
  assert.match(block, /setBusy\(true\)/);
  assert.match(block, /finally[\s\S]*?sessionMutationLockRef\.current = null/);
  assert.match(block, /finally[\s\S]*?setBusy\(false\)/);
});

test("Dashboard bulk buttons freeze the full filtered candidate set rather than Top-N", async () => {
  const dashboardSource = await source("app/Dashboard.tsx");
  const selectionBlock = dashboardSource.slice(
    dashboardSource.indexOf("const bulkSelectedItems"),
    dashboardSource.indexOf("const queryPreviewItems"),
  );
  assert.match(selectionBlock, /galleryItems\.filter/);
  assert.match(selectionBlock, /resultMask\[item\.rowIndex\] === 1/);
  assert.match(selectionBlock, /canAnnotate\(item\)/);
  assert.doesNotMatch(selectionBlock, /topLimit|rankedItems|selectedId/);
  assert.match(dashboardSource, /selectionRuleActive[\s\S]*?smartFilterKind !== null/);
  assert.match(dashboardSource, /Mark all positive/);
  assert.match(dashboardSource, /Mark all negative/);
  assert.match(dashboardSource, /bulkSelectedItems\.length > MAX_BULK_TUNING_ANNOTATIONS/);
  assert.match(
    dashboardSource,
    /bulkUpdateAnnotations\([\s\S]*?`selection-bulk-\$\{label\}`/,
  );
});
