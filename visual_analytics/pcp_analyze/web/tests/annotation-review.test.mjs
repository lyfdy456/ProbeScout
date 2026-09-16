import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import * as feedbackLabels from "../app/lib/feedbackLabels.ts";
import * as tuningApi from "../app/lib/tuningApi.ts";

async function source(relativePath) {
  return readFile(new URL(`../${relativePath}`, import.meta.url), "utf8");
}

test("current-session feedback can be reviewed without inventing another data store", async () => {
  const [reviewSource, dashboardSource, gallerySource] = await Promise.all([
    source("app/components/AnnotationReviewGallery.tsx"),
    source("app/Dashboard.tsx"),
    source("app/components/TopGallery.tsx"),
  ]);

  assert.match(gallerySource, /export function GalleryThumbnail/);
  assert.match(gallerySource, /export function PreferenceControls/);
  assert.match(reviewSource, /\[\.\.\.annotations\.values\(\)\]/);
  assert.match(reviewSource, /itemById\.get\(annotation\.imageId\)/);
  assert.match(reviewSource, /<strong>Review feedback<\/strong>/);
  assert.match(reviewSource, /aria-label="Current session feedback"/);
  assert.match(reviewSource, /aria-label="Filter feedback"/);
  assert.match(reviewSource, /Open feedback image:/);
  assert.doesNotMatch(reviewSource, /Review labeled images|H · Human Feedback/);
  assert.doesNotMatch(reviewSource, /fetch\(|localStorage|sessionStorage/);

  assert.match(
    dashboardSource,
    /const updateReviewedPreference = useCallback[\s\S]*?if \(preference === "unmarked" \? !canRemoveAnnotation\(item\) : !canAnnotate\(item\)\) return false;[\s\S]*?tuning\.updateAnnotation\(item, preference, "annotation-review"\)/,
    "review edits must reuse the protected session mutation; held-out history permits removal only",
  );
  assert.match(
    dashboardSource,
    /<AnnotationReviewGallery[\s\S]*?annotations=\{tuning\.annotations\}[\s\S]*?itemById=\{galleryItemById\}[\s\S]*?onPreferenceChange=\{updateReviewedPreference\}/,
  );
  assert.match(
    dashboardSource,
    /key=\{tuning\.session\?\.id \?\?/,
    "task, target, method, or session replacement must reset transient review state",
  );
});

test("labeled-image review includes every feedback level and safe relabeling", async () => {
  const reviewSource = await source("app/components/AnnotationReviewGallery.tsx");

  assert.match(reviewSource, /if \(label > 0\) return "positive"/);
  assert.match(reviewSource, /if \(label < 0\) return "negative"/);
  assert.match(reviewSource, /return "uncertain"/);
  assert.match(reviewSource, /All/);
  assert.match(reviewSource, /Positive/);
  assert.match(reviewSource, /Negative/);
  assert.match(reviewSource, /Unsure/);
  assert.match(reviewSource, /PreferenceControls/);
  assert.match(reviewSource, /preference === "unmarked" \? "Feedback removed\."/);
  assert.match(reviewSource, /pending=\{saving \|\| busy\}/);
  assert.match(reviewSource, /<GalleryLightbox/);
  assert.match(reviewSource, /renderFeedback=\{\(item\) => renderReviewControls\(item, true\)\}/);
  assert.match(reviewSource, /getOriginalVqaLabel=\{getOriginalVqaLabel\}/);
  assert.match(reviewSource, /getAttributeStrengths=\{getAttributeStrengths\}/);
  assert.match(
    reviewSource,
    /left\.item\.rowIndex - right\.item\.rowIndex/,
    "review cards must keep a stable order while labels are corrected",
  );
});

test("feedback review is a single scrollable row without truncating annotations", async () => {
  const [css, reviewSource] = await Promise.all([
    source("app/globals.css"),
    source("app/components/AnnotationReviewGallery.tsx"),
  ]);

  assert.match(css, /\.annotation-review-scroll\s*\{[^}]*overflow-x: auto/);
  assert.match(css, /\.annotation-review-grid\s*\{[^}]*grid-auto-flow: column/);
  assert.match(css, /\.annotation-review-grid\s*\{[^}]*grid-auto-columns:/);
  assert.match(reviewSource, /filteredEntries\.map\(\(\{ annotation, item \}\) =>/);
  assert.doesNotMatch(reviewSource, /filteredEntries\.slice\(/);
  assert.match(reviewSource, /ref=\{reviewScrollRef\}/);
  assert.match(reviewSource, /reviewScrollRef\.current\.scrollLeft = 0/);
  assert.match(reviewSource, /reviewScrollRef\.current\.scrollTop = 0/);
  assert.match(reviewSource, /\}, \[expanded, filter\]\)/);
  assert.match(reviewSource, /tabIndex=\{0\}/);
});

test("expanded feedback strip keeps every thumbnail but renders one shared negative editor", async () => {
  const imports = {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "../lib/feedbackLabels": feedbackLabels,
    "../lib/tuningApi": tuningApi,
    // These dependencies are not rendered or called by this review fixture.
    "./ProjectionScatter": {},
    "../lib/resultScope": {},
    "../lib/useGalleryRowWindow": {},
    "./AttributeStrengthProfile": {},
    "./GalleryLightbox": {},
  };
  const compile = async (name, expanded = false) => {
    let text = await source(`app/components/${name}.tsx`);
    if (expanded) {
      text = text.replace(
        "const [expanded, setExpanded] = useState(false);",
        "const [expanded, setExpanded] = useState(true);",
      );
      text = text.replace(
        "const [selectedId, setSelectedId] = useState<string | null>(null);",
        'const [selectedId, setSelectedId] = useState<string | null>("fixture-3");',
      );
    }
    const compiled = ts.transpileModule(text, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const componentModule = { exports: {} };
    new Function("require", "module", "exports", compiled)((id) => {
      assert.ok(id in imports, `unexpected runtime import ${id}`);
      return imports[id];
    }, componentModule, componentModule.exports);
    return componentModule.exports;
  };
  imports["./TopGallery"] = await compile("TopGallery");
  const { AnnotationReviewGallery } = await compile("AnnotationReviewGallery", true);
  const labels = [2, 1, 0, -1, -2, 1, 0, -1];
  const items = labels.map((label, index) => ({
    id: `fixture-${index}`, rowIndex: index, label: `Feedback ${index + 1}`,
    thumbnailAtlas: { atlasUrl: "/fixture-only.webp", row: 0, col: index },
  }));
  const annotations = new Map(items.map((item, index) => [item.id, {
    imageId: item.id, rowIndex: item.rowIndex, label: labels[index],
    updatedAt: "fixture-only", suggestedFailedAttributeId: labels[index] < 0 ? "long-attribute" : undefined,
  }]));
  const preferences = new Map(items.map((item, index) => [item.id,
    feedbackLabels.FEEDBACK_LABELS.find((option) => option.numeric === labels[index]).value,
  ]));
  const html = renderToStaticMarkup(React.createElement(AnnotationReviewGallery, {
    annotations, preferences, itemById: new Map(items.map((item) => [item.id, item])),
    jointSession: true, attributeTargets: [
      { id: "long-attribute", label: "Symmetric electric-lit enclosed interior" },
      { id: "cat", label: "Cat" },
    ],
    onPreferenceChange: async () => true,
    onConfirmFailureAttributes: async () => true,
  }));
  assert.equal((html.match(/<article\b/g) ?? []).length, 8);
  assert.equal((html.match(/aria-label="Select feedback image:/g) ?? []).length, 8);
  assert.equal((html.match(/aria-label="Open feedback image:/g) ?? []).length, 8);
  assert.equal((html.match(/aria-label="Feedback label for /g) ?? []).length, 1);
  assert.equal((html.match(/class="gallery-preference-button /g) ?? []).length, 5);
  assert.equal((html.match(/class="failure-attribute-selector"/g) ?? []).length, 1);
  assert.equal((html.match(/type="checkbox"/g) ?? []).length, 2);
  assert.match(html, /Suggested · unconfirmed/);
  assert.match(html, /annotation-review-positive/);
  assert.match(html, /annotation-review-negative/);
  assert.match(html, /annotation-review-uncertain/);
  assert.match(html, /class="annotation-review-scroll" role="region" aria-label="All feedback images" tabindex="0"/);
});
