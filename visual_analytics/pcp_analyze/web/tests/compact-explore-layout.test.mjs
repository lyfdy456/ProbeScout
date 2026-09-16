import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("compact Query retains reference images, preview, and Retrieval target without text input", async () => {
  const dashboard = await source("app/Dashboard.tsx");
  assert.doesNotMatch(dashboard, /Fixed task query|query-text-readonly/);
  assert.match(dashboard, /className="query-images"/);
  assert.match(dashboard, /onClick=\{\(\) => setQueryPreviewIndex\(index\)\}/);
  assert.match(dashboard, /<GalleryLightbox[\s\S]*?items=\{queryPreviewItems\}[\s\S]*?activeIndex=\{queryPreviewIndex\}/);
  assert.match(dashboard, /className="result-controls"[\s\S]*?aria-label="Retrieval target"[\s\S]*?onChange=\{\(event\) => changeRetrievalTarget\(event.target.value as RetrievalTarget\)\}/);
  assert.doesNotMatch(dashboard, /TextQueryControl|aria-label="Text query"/);
});

test("projection and cluster controls move into closed Advanced Settings and retain their handlers", async () => {
  const [dashboard, scatter] = await Promise.all([
    source("app/Dashboard.tsx"), source("app/components/ProjectionScatter.tsx"),
  ]);
  const frame = dashboard.indexOf('className="projection-frame"');
  const controls = dashboard.slice(dashboard.indexOf('<details className="advanced-settings">'), dashboard.indexOf('className={`scope-purpose-banner'));
  assert.match(dashboard.slice(frame), /<ProjectionScatter\s+showHeading=\{false\}/);
  assert.doesNotMatch(dashboard.slice(frame), /aria-label="Projection"|aria-label="Embedding cluster scheme"/);
  assert.doesNotMatch(controls, /<details[^>]*\bopen[= >]/);
  assert.match(controls, /aria-label="Projection"[\s\S]*?setProjection\(event.target.value as ProjectionKind\);\s+setProjectionSelection\(null\)/);
  assert.match(controls, /aria-label="Embedding cluster scheme"[\s\S]*?changeVisualClusterScheme\(/);
  assert.ok(scatter.indexOf("{controls}") < scatter.indexOf('className="projection-stage"'));
  assert.match(scatter, /className="projection-toolbar"[\s\S]*?\{controls\}[\s\S]*?className="projection-meta"/);
  assert.match(dashboard.slice(frame), /height=\{250\}/);
});

test("Top viewport measures two rows and keeps exploration content independent of grid stretch", async () => {
  const css = await source("app/globals.css");
  assert.match(css, /\.top-gallery-scroll\s*\{\s*max-height: var\(--gallery-row-window-height, none\)/);
  assert.match(css, /\.exploration-content\s*\{[^}]*flex: 0 0 auto/s);
  assert.doesNotMatch(css, /max-height: clamp\(260px, 32svh, 360px\)/);
  assert.match(css, /\.projection-controls label,\s*\.projection-controls select\s*\{\s*min-width: 0;\s*width: 100%/);
  assert.match(css, /\.projection-meta\s*\{[\s\S]*?flex-wrap: wrap/);
});
