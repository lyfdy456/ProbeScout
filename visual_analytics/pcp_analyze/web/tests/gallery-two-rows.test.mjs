import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { galleryRowWindowHeight } from "../app/lib/galleryRowWindow.ts";

function gridCards(count, columns, rowHeight, gap = 6, top = 6) {
  return Array.from({ length: count }, (_, index) => ({
    top: top + Math.floor(index / columns) * (rowHeight + gap),
    height: rowHeight,
  }));
}

test("Top 30, 50, 100 and 200 expose two complete rows with six to eight desktop images", () => {
  for (const count of [30, 50, 100, 200]) {
    for (const columns of [3, 4]) {
      assert.equal(galleryRowWindowHeight(gridCards(count, columns, 140), 2, 6, 6), 298);
    }
  }
  assert.equal(galleryRowWindowHeight(gridCards(30, 4, 140)), 286);
});

test("a narrow panel and expanded attribute or feedback rows adapt to actual dimensions", () => {
  assert.equal(galleryRowWindowHeight(gridCards(50, 2, 206, 9), 4, 6, 6), 863);
  assert.equal(galleryRowWindowHeight(gridCards(50, 1, 250, 12), 4, 8, 8), 1052);
  assert.equal(galleryRowWindowHeight(gridCards(50, 4, 108), 4, 6, 6), 462);
});

test("fewer than two rows and a partially filled last row retain natural short heights", () => {
  assert.equal(galleryRowWindowHeight(gridCards(1, 4, 140), 2, 6, 6), 152);
  assert.equal(galleryRowWindowHeight(gridCards(5, 4, 140), 2, 6, 6), 298);
  assert.equal(galleryRowWindowHeight(gridCards(12, 4, 140), 2, 6, 6), 298);
  assert.equal(galleryRowWindowHeight([], 2, 6, 6), null);
});

test("row grouping uses the tallest actual card and preserves nonuniform row gaps", () => {
  const cards = [
    { top: 340, height: 120 }, { top: 6, height: 100 },
    { top: 221, height: 90 }, { top: 6, height: 120 },
    { top: 132, height: 80 }, { top: 340, height: 90 },
    { top: 472, height: 500 },
  ];
  const original = structuredClone(cards);
  assert.equal(galleryRowWindowHeight(cards, 4, 6, 6), 466);
  assert.deepEqual(cards, original);
});

test("padding cannot expose part of the third row and hidden cards cannot create rows", () => {
  assert.equal(galleryRowWindowHeight(gridCards(30, 4, 100, 2), 2, 6, 20), 210);
  assert.equal(galleryRowWindowHeight([{ top: 0, height: 0 }, { top: NaN, height: 100 }]), null);
});

test("gallery measurement observes content and coalesces idempotent writes without ranking changes", async () => {
  const [gallery, hook, css] = await Promise.all([
    readFile(new URL("../app/components/TopGallery.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/useGalleryRowWindow.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(gallery, /useGalleryRowWindow\(galleryScrollRef, galleryGridRef, rankedItems\)/);
  assert.match(gallery, /ref=\{galleryGridRef\} className="gallery-grid"/);
  assert.match(gallery, /galleryScrollRef\.current\.scrollTop = 0/);
  assert.match(hook, /card\.offsetTop[\s\S]*?card\.offsetHeight/);
  assert.match(hook, /cards\.map\([^\n]+\n\s*2,/);
  assert.match(hook, /observer\?\.observe\(grid\)/);
  assert.match(hook, /observer\?\.observe\(card\)/);
  assert.doesNotMatch(hook, /observer\?\.observe\(viewport\)|getBoundingClientRect|setState/);
  assert.match(hook, /frame === null[\s\S]*?requestAnimationFrame\(measure\)/);
  assert.match(hook, /getPropertyValue\(HEIGHT_PROPERTY\) !== nextHeight/);
  assert.match(hook, /observer\?\.disconnect\(\)/);
  assert.match(hook, /cancelAnimationFrame\(frame\)/);
  assert.match(css, /max-height:\s*var\(--gallery-row-window-height,\s*none\)/);
  assert.match(gallery, /title=\{`Ranked by \$\{learner.label\}`\}>Top \{safeLimit\}/);
});
