import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = async (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("narrow Top cards wrap all five feedback controls inside the card", async () => {
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(
    css,
    /@media \(max-width: 440px\) \{[\s\S]*?\.gallery-card-footer \{[\s\S]*?flex-direction: column/,
  );
  assert.match(
    css,
    /@media \(max-width: 440px\) \{[\s\S]*?\.gallery-card-footer \.gallery-preference \{[\s\S]*?grid-template-columns: repeat\(3, minmax\(0, 1fr\)\)/,
  );
  assert.match(
    css,
    /@media \(max-width: 440px\) \{[\s\S]*?\.gallery-card-footer \.gallery-preference-button \{[\s\S]*?min-width: 0;[\s\S]*?width: 100%/,
  );
});

test("Top results scroll inside a bounded keyboard-accessible region", async () => {
  const [gallery, css] = await Promise.all([
    source("app/components/TopGallery.tsx"),
    source("app/globals.css"),
  ]);
  const toolbarIndex = gallery.indexOf('className="gallery-toolbar"');
  const evaluationIndex = gallery.indexOf('className="gallery-evaluation"');
  const scrollIndex = gallery.indexOf('className="top-gallery-scroll"');
  const gridIndex = gallery.indexOf('className="gallery-grid"', scrollIndex);
  assert.ok(toolbarIndex >= 0 && toolbarIndex < scrollIndex, "the toolbar must stay outside the scroll region");
  assert.ok(evaluationIndex >= 0 && evaluationIndex < scrollIndex, "evaluation metrics must stay outside the scroll region");
  assert.ok(scrollIndex >= 0 && gridIndex > scrollIndex, "the complete card grid must live inside the scroll region");
  assert.match(
    gallery,
    /ref=\{galleryScrollRef\}[\s\S]*?className="top-gallery-scroll"[\s\S]*?role="region"[\s\S]*?tabIndex=\{0\}/,
  );
  assert.match(gallery, /galleryScrollRef\.current\.scrollTop = 0/);
  assert.match(css, /\.top-gallery-scroll\s*\{[\s\S]*?max-height: var\(--gallery-row-window-height, none\)[\s\S]*?overflow-y: auto[\s\S]*?scrollbar-gutter: stable/);
  assert.match(css, /\.top-gallery-scroll:focus-visible/);
  assert.match(
    css,
    /@media \(max-width: 760px\)[\s\S]*?\.top-gallery-scroll\s*\{[\s\S]*?max-height: var\(--gallery-row-window-height, none\)[\s\S]*?scrollbar-gutter: stable/,
  );
});
