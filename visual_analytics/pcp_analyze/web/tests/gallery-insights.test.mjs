import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = async (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("original VQA supervision is fetched independently of personal sessions and never from GT", async () => {
  const [api, hook] = await Promise.all([
    source("app/lib/tuningApi.ts"),
    source("app/lib/useOriginalVqaSupervision.ts"),
  ]);
  assert.match(
    api,
    /fetchOriginalVqaSupervision[\s\S]*?\/tasks\/\$\{encodeURIComponent\(taskId\)\}\/targets\/\$\{encodeURIComponent\(targetId\)\}\/original-supervision/,
  );
  assert.match(api, /credentials: "same-origin"/);
  assert.match(hook, /new Map<number, OriginalVqaSupervisionItem>/);
  assert.match(hook, /item\.label !== 0 && item\.label !== 1/);
  assert.match(hook, /records\.has\(item\.rowIndex\)/);
  assert.match(hook, /AbortController/);
  assert.doesNotMatch(hook, /groundTruth|ground-truth/i);
  const start = api.indexOf("export function fetchOriginalVqaSupervision(");
  const end = api.indexOf("export function ", start + 1);
  assert.ok(start >= 0 && end > start);
  assert.doesNotMatch(api.slice(start, end), /groundTruth|ground-truth/i);
});

test("Top cards and the lightbox share VQA provenance and attribute profiles", async () => {
  const [gallery, lightbox, profile, css] = await Promise.all([
    source("app/components/TopGallery.tsx"),
    source("app/components/GalleryLightbox.tsx"),
    source("app/components/AttributeStrengthProfile.tsx"),
    source("app/globals.css"),
  ]);

  const rankingStart = gallery.indexOf("const { ranked, candidateCount } = useMemo");
  const rankingEnd = gallery.indexOf("const truePositiveCount", rankingStart);
  assert.ok(rankingStart >= 0 && rankingEnd > rankingStart);
  assert.doesNotMatch(
    gallery.slice(rankingStart, rankingEnd),
    /getOriginalVqaLabel|getAttributeStrengths|VQA/,
    "provenance and explanations must not affect Top ranking",
  );

  assert.match(gallery, /const originalVqaLabel = getOriginalVqaLabel\?\.\(item\) \?\? null/);
  assert.match(
    gallery,
    /originalVqaLabel !== null[\s\S]*?originalVqaLabel === 1 \? "✓" : "✕"/,
    "label zero must render as an explicit VQA negative instead of disappearing",
  );
  assert.match(gallery, /gallery-card-badge-stack[\s\S]*?gallery-original-vqa-badge/);
  assert.match(gallery, /<AttributeStrengthProfile points=\{attributeStrengths\} variant="compact"/);
  assert.match(
    gallery,
    /<GalleryLightbox[\s\S]*?getOriginalVqaLabel=\{getOriginalVqaLabel\}[\s\S]*?getAttributeStrengths=\{getAttributeStrengths\}/,
  );

  assert.match(lightbox, /const originalVqaLabel = getOriginalVqaLabel\?\.\(item\) \?\? null/);
  assert.match(lightbox, /gallery-lightbox-vqa-badge/);
  assert.match(lightbox, /variant="expanded"/);
  assert.match(profile, /role="img"/);
  assert.match(profile, /attribute-strength-profile-line/);
  assert.match(profile, /clampUnit\(point\.value\)/);
  assert.match(css, /\.gallery-card-badge-stack\s*\{/);
  assert.match(css, /\.attribute-strength-profile-compact\s*\{/);
  assert.match(css, /\.gallery-lightbox-insights\s*\{/);
});

test("compact attribute profiles sit below the image and show aligned attribute labels", async () => {
  const [gallery, lightbox, profile] = await Promise.all([
    source("app/components/TopGallery.tsx"),
    source("app/components/GalleryLightbox.tsx"),
    source("app/components/AttributeStrengthProfile.tsx"),
  ]);
  const imageButtonStart = gallery.indexOf('className="gallery-card-main"');
  const imageButtonEnd = gallery.indexOf("</button>", imageButtonStart);
  assert.ok(imageButtonStart >= 0 && imageButtonEnd > imageButtonStart);
  const imageButton = gallery.slice(imageButtonStart, imageButtonEnd);
  assert.doesNotMatch(imageButton, /AttributeStrengthProfile|gallery-card-insights/);
  assert.match(imageButton, /GalleryThumbnail/);
  assert.match(imageButton, /gallery-card-badge-stack/);
  assert.match(imageButton, /onItemSelect\?\.\(item\)/);
  const belowImage = gallery.slice(imageButtonEnd, gallery.indexOf('className="gallery-meta"', imageButtonEnd));
  assert.match(belowImage, /gallery-card-insights[\s\S]*?<AttributeStrengthProfile points=\{attributeStrengths\} variant="compact"/);
  assert.match(profile, /const rowStep = compact \? 16 : 30/);
  assert.match(profile, /attribute-strength-profile-compact-labels[\s\S]*?finitePoints\.map[\s\S]*?\{point\.label\}/);
  assert.match(profile, /style=\{compact \? \{ height \} : undefined\}/);
  const lightboxStage = lightbox.indexOf('className="gallery-lightbox-stage"');
  const lightboxInsights = lightbox.indexOf('className="gallery-lightbox-insights"');
  assert.ok(lightboxInsights > lightboxStage);
  assert.match(lightbox.slice(lightboxStage, lightboxInsights), /<LightboxVisual[\s\S]*?<\/div>/);
});
