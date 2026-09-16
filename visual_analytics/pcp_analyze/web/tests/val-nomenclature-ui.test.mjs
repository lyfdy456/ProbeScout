import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("active tuning uses the single Val name and keeps provenance in protocol details", async () => {
  const [panel, actions] = await Promise.all([
    source("app/components/TuningPanel.tsx"),
    source("app/components/TuningFunctionActions.tsx"),
  ]);
  assert.doesNotMatch(`${panel}\n${actions}`, /small[ -]?(?:vqa[ -]?)?val|小\s*val|小验证集/i);
  assert.match(panel, /Val · \{validation \? `\$\{validation\.count\} images`/);
  assert.match(panel, /<summary>Val protocol<\/summary>/);
  assert.match(panel, /Original VQA 20% · fixed per task · excluded from training, exploration, and feedback/);
  assert.match(panel, /not an independent holdout for the original model/);
  assert.match(panel, /Val · original VQA labels · fixed 20%/);
  assert.match(panel, /runValidationManifestSha256/);
});

test("Val naming does not rename stored evaluation identities or relabel historical results", async () => {
  const [panel, api] = await Promise.all([
    source("app/components/TuningPanel.tsx"),
    source("app/lib/tuningApi.ts"),
  ]);
  assert.match(api, /evaluationScope\?: "vqa-validation" \| "clean-validation" \| "probe-validation" \| "validation" \| "test"/);
  assert.match(panel, /vqaValidation = run\?\.evaluationScope === "vqa-validation"/);
  assert.match(panel, /className="tuning-result-overview" key=\{run\.id\}/);
  assert.match(panel, /<details className="tuning-evaluation-result">/);
  assert.match(panel, /vqaValidation \? "Evaluation details" : "Historical evaluation · original scope"/);
  for (const historicalLabel of ["Clean Validation", "Probe Val", "Saved run", "legacy Web Validation"]) {
    assert.ok(panel.includes(`"${historicalLabel}"`), historicalLabel);
  }
  assert.doesNotMatch(panel, /testEvaluation|legacy Frozen Test/);
});

test("Tune actions remain locked until Val resolves and exclude historical Val feedback", async () => {
  const [actions, hook] = await Promise.all([
    source("app/components/TuningFunctionActions.tsx"),
    source("app/lib/useTuningSession.ts"),
  ]);
  assert.match(actions, /validationReady = false/);
  assert.match(actions, /const commonDisabled = \([\s\S]*?\|\| !validationReady/);
  assert.match(actions, /disabled=\{!refinementCapability\(state\.refinementCapabilities, action\.mode\)\.available \|\| !tuningReady \|\| commonDisabled \|\| missingSnapshot\}/);
  assert.match(actions, /const tuningReady = jointTargetReady[\s\S]*?&& supervisionResolved/);
  assert.match(actions, /annotation\.supervision\?\.includedInTune !== false/);
  assert.match(hook, /if \(annotation\.supervision\?\.includedInTune !== false\) \{[\s\S]*?counts\.usablePositive/);
  const run = hook.slice(hook.indexOf("const runTuning = useCallback"), hook.indexOf("const applyRun = useCallback"));
  assert.match(run, /!feedbackContextRef\.current\.input\.validationReady[\s\S]*?await waitForPendingAnnotations[\s\S]*?!feedbackContextRef\.current\.input\.validationReady/);
});

test("Val guards apply to every feedback path, including already queued writes", async () => {
  const hook = await source("app/lib/useTuningSession.ts");
  assert.match(hook, /!context\.input\.validationReady/);
  assert.match(hook, /assertFeedbackAllowed\(item, session\.id, label === "unmarked"\)/);
  assert.match(hook, /assertFeedbackAllowed\(item, sessionId, label === "unmarked"\)/);
  assert.match(hook, /for \(const item of uniqueItems\) assertFeedbackAllowed\(item, sessionId\)/);
  assert.match(hook, /const updateFailureAttributes[\s\S]*?assertFeedbackAllowed\(item, session\.id\)/);
  assert.match(hook, /removal \? context\.input\.canRemoveFeedback\(item\) : context\.input\.canWriteFeedback\(item\)/);
});
