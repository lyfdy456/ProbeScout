import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = async (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("Top Gallery opens the lightbox and reuses the card feedback source", async () => {
  const [gallery, lightbox] = await Promise.all([
    source("app/components/TopGallery.tsx"),
    source("app/components/GalleryLightbox.tsx"),
  ]);
  assert.match(gallery, /className="gallery-card-main"[\s\S]*?aria-haspopup="dialog"/);
  assert.match(gallery, /onItemSelect\?\.\(item\);[\s\S]*?setPreviewId\(item\.id\)/);
  assert.match(
    gallery,
    /const renderPreferenceControls = \([\s\S]*?item: GalleryItem,[\s\S]*?onChange: PreferenceChangeHandler \| undefined = onPreferenceChange[\s\S]*?value=\{getPreference\(preferences, item\.id\)\}[\s\S]*?canAnnotate=\{canAnnotate\?\.\(item\) \?\? Boolean\(onPreferenceChange\)\}[\s\S]*?onChange=\{onChange\}/,
  );
  assert.match(gallery, /\{renderPreferenceControls\(item\)\}/);
  assert.match(
    gallery,
    /<GalleryLightbox[\s\S]*?onClose=\{handlePreviewClose\}[\s\S]*?renderFeedback=\{\(item\) => renderPreferenceControls\([\s\S]*?item,[\s\S]*?true,[\s\S]*?handleLightboxPreferenceChange/,
  );
  assert.match(lightbox, /\{renderFeedback && \([\s\S]*?\{renderFeedback\(item\)\}/);
  assert.doesNotMatch(
    gallery.match(/const renderPreferenceControls[\s\S]*?\n  \);/)?.[0] ?? "",
    /onClose|setPreviewId/,
    "feedback mutation must not close the lightbox",
  );
});

test("lightbox navigation synchronizes the outer selected item and stale previews are cleared", async () => {
  const gallery = await source("app/components/TopGallery.tsx");
  assert.match(
    gallery,
    /useEffect\(\(\) => \{[\s\S]*?previewId === null \|\| previewIndex >= 0[\s\S]*?queueMicrotask\([\s\S]*?previewIdRef\.current !== stalePreviewId[\s\S]*?previewIdRef\.current = null[\s\S]*?setPreviewId\(null\)[\s\S]*?\}, \[previewId, previewIndex\]\)/,
    "an item leaving the ranked Top must retire its preview state",
  );
  assert.match(
    gallery,
    /const handlePreviewIndexChange = useCallback\([\s\S]*?const nextItem = rankedItems\[index\][\s\S]*?setPreviewId\(nextItem\.id\)[\s\S]*?onItemSelect\?\.\(nextItem\)/,
    "keyboard and button navigation must select the newly active image outside the modal",
  );
  assert.match(
    gallery,
    /<GalleryLightbox[\s\S]*?onActiveIndexChange=\{handlePreviewIndexChange\}/,
  );
});

test("enlarged Development feedback advances only after a saved new label", async () => {
  const [dashboard, gallery, lightbox] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TopGallery.tsx"),
    source("app/components/GalleryLightbox.tsx"),
  ]);
  assert.match(
    dashboard,
    /const updatePreference = useCallback\(async[\s\S]*?if \(!canAnnotate\(item\)\) return false[\s\S]*?await tuning\.updateAnnotation[\s\S]*?return true[\s\S]*?catch[\s\S]*?return false/,
    "the gallery must expose whether the persisted annotation succeeded",
  );
  const handlerStart = gallery.indexOf("const handleLightboxPreferenceChange");
  const handlerEnd = gallery.indexOf("const handlePreviewClose", handlerStart);
  const handler = gallery.slice(handlerStart, handlerEnd);
  assert.ok(handlerStart >= 0 && handlerEnd > handlerStart, "auto-advance handler should be locatable");
  assert.match(handler, /lightboxFeedbackSavingRef\.current/);
  assert.match(handler, /await onPreferenceChange\(item, preference\)/);
  assert.match(handler, /committed === false[\s\S]*?return false/);
  assert.match(handler, /previewIdRef\.current !== item\.id[\s\S]*?return committed/);
  assert.match(handler, /previewNavigationRevisionRef\.current !== navigationRevision/);
  assert.match(handler, /preference === "unmarked"[\s\S]*?return committed/);
  assert.match(handler, /rankedItems\[currentIndex \+ 1\]/);
  assert.match(handler, /if \(!nextItem\)[\s\S]*?return committed/);
  assert.match(handler, /setPreviewId\(nextItem\.id\)[\s\S]*?onItemSelect\?\.\(nextItem\)/);
  assert.doesNotMatch(handler, /%\s*rankedItems\.length/, "automatic review must stop instead of wrapping");
  assert.match(gallery, /disabled=\{!onChange \|\| pending\}/);
  assert.match(gallery, /feedbackBusy=\{lightboxFeedbackSaving\}/);
  assert.match(lightbox, /aria-busy=\{feedbackBusy\}/);
  assert.match(lightbox, /gallery-lightbox-feedback-status[\s\S]*?role="status"/);
});

test("full images are separate from atlas thumbnails and requested only by the open preview", async () => {
  const [dashboard, gallery, lightbox] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TopGallery.tsx"),
    source("app/components/GalleryLightbox.tsx"),
  ]);
  assert.match(dashboard, /fullImageSrc: `\/api\/tuning\/images\/\$\{encodeURIComponent\(selectedTaskId\)\}\/\$\{rowIndex\}`/);
  assert.doesNotMatch(gallery.match(/function GalleryThumbnail[\s\S]*?\n\}/)?.[0] ?? "", /fullImageSrc/);
  assert.match(lightbox, /const directUrl = item\.fullImageSrc \|\| item\.imageSrc \|\| null/);
  assert.match(lightbox, /The full image is requested only after the lightbox opens/);
  assert.match(lightbox, /onError=\{\(\) => \{[\s\S]*?setFailed\(true\)/);
  assert.match(lightbox, /showAtlas[\s\S]*?gallery-lightbox-atlas/);
});

test("lightbox supports modal focus, closing, keyboard navigation, and scroll locking", async () => {
  const lightbox = await source("app/components/GalleryLightbox.tsx");
  assert.match(lightbox, /createPortal\(/);
  assert.match(lightbox, /role="dialog"[\s\S]*?aria-modal="true"/);
  assert.match(lightbox, /event\.key === "Escape"[\s\S]*?onClose\(\)/);
  assert.match(
    lightbox,
    /const feedbackOwnsArrowKey =[\s\S]*?closest\("\[data-gallery-lightbox-feedback\]"\)[\s\S]*?if \(feedbackOwnsArrowKey\) return;[\s\S]*?event\.key === "ArrowLeft"/,
    "feedback controls must own their arrow keys instead of changing the active image",
  );
  assert.match(lightbox, /event\.key === "ArrowLeft"[\s\S]*?move\(-1\)/);
  assert.match(lightbox, /event\.key === "ArrowRight"[\s\S]*?move\(1\)/);
  assert.match(lightbox, /document\.body\.style\.overflow = "hidden"/);
  assert.match(lightbox, /document\.body\.style\.overflow = previousOverflow/);
  assert.match(lightbox, /previousFocus\?\.focus\(\)/);
  assert.match(lightbox, /event\.target === event\.currentTarget/);
});

test("lightbox is viewport-bound and has visible keyboard focus", async () => {
  const css = await source("app/globals.css");
  assert.match(css, /\.gallery-lightbox-backdrop\s*\{[\s\S]*?position: fixed/);
  assert.match(css, /\.gallery-lightbox-dialog\s*\{[\s\S]*?max-height: calc\(100dvh/);
  assert.match(css, /\.gallery-lightbox-image\s*\{[\s\S]*?object-fit: contain/);
  assert.match(css, /\.gallery-lightbox-close:focus-visible/);
  assert.match(css, /\.gallery-preference-button:focus-visible/);
  assert.match(css, /\.gallery-lightbox-feedback\s*\{[\s\S]*?display: flex/);
  assert.match(css, /\.gallery-lightbox-feedback \.gallery-preference-button\s*\{[\s\S]*?height: 34px/);
  assert.match(css, /@media \(max-width: 760px\)[\s\S]*?\.gallery-lightbox-dialog\s*\{[\s\S]*?height: 100dvh/);
  assert.match(css, /@media \(max-width: 760px\)[\s\S]*?\.gallery-lightbox-feedback\s*\{[\s\S]*?flex-direction: column/);
});

test("fixed Query images reuse the accessible lightbox and direct exported assets", async () => {
  const [dashboard, lightbox, css] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/GalleryLightbox.tsx"),
    source("app/globals.css"),
  ]);
  assert.match(dashboard, /const queryPreviewItems = useMemo<GalleryItem\[\]>/);
  assert.match(dashboard, /imageSrc: source,[\s\S]*?fullImageSrc: source/);
  assert.match(dashboard, /className="fixed-query-image-button"[\s\S]*?aria-haspopup="dialog"[\s\S]*?setQueryPreviewIndex\(index\)/);
  assert.match(dashboard, /<GalleryLightbox[\s\S]*?items=\{queryPreviewItems\}[\s\S]*?fixed Query images/);
  const queryLightbox = dashboard.match(/<GalleryLightbox\s+[\s\S]*?items=\{queryPreviewItems\}[\s\S]*?\/>/)?.[0] ?? "";
  assert.doesNotMatch(queryLightbox, /renderFeedback/);
  assert.match(lightbox, /navigationHint = "Esc closes/);
  assert.match(css, /\.fixed-query-image-button\s*\{[\s\S]*?cursor: zoom-in/);
  assert.match(css, /\.fixed-query-image-button:focus-visible/);
});

test("Frozen Test previews expose the read-only reason without a feedback mutation", async () => {
  const [dashboard, gallery] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TopGallery.tsx"),
  ]);
  assert.match(dashboard, /Frozen Test images are evaluation-only\./);
  assert.match(
    gallery,
    /if \(!canAnnotate\) \{[\s\S]*?gallery-feedback-locked[\s\S]*?expanded && disabledReason[\s\S]*?Feedback unavailable[\s\S]*?\}/,
  );
  assert.match(
    gallery,
    /renderFeedback=\{\(item\) => renderPreferenceControls\([\s\S]*?item,[\s\S]*?true,[\s\S]*?handleLightboxPreferenceChange/,
    "the modal must reuse the same row-level guard as each Top card",
  );
});

test("Frozen Test GT positives are visibly marked after ranking and in the active preview", async () => {
  const [dashboard, gallery, lightbox, css] = await Promise.all([
    source("app/Dashboard.tsx"),
    source("app/components/TopGallery.tsx"),
    source("app/components/GalleryLightbox.tsx"),
    source("app/globals.css"),
  ]);

  assert.match(
    dashboard,
    /const getGroundTruth = useCallback\([\s\S]*?currentTarget = targetIndex\.get\(retrievalTarget\)[\s\S]*?isGroundTruthPositive\([\s\S]*?dataset\.groundTruth,[\s\S]*?item\.rowIndex,[\s\S]*?currentTarget,[\s\S]*?dataset\.manifest\.targetCount[\s\S]*?\[dataset, retrievalTarget, targetIndex\]/,
    "the audit marker must follow the current Retrieval target",
  );
  assert.match(
    dashboard,
    /evaluation=\{resultScope === "test" \? \{[\s\S]*?label: "Frozen Test",[\s\S]*?isPositive: getGroundTruth,[\s\S]*?\} : undefined\}/,
    "per-image GT status must only be exposed in Frozen Test",
  );

  const rankingStart = gallery.indexOf("const { ranked, candidateCount } = useMemo");
  const rankingEnd = gallery.indexOf("const truePositiveCount", rankingStart);
  assert.ok(rankingStart >= 0 && rankingEnd > rankingStart, "ranking block should be locatable");
  assert.doesNotMatch(
    gallery.slice(rankingStart, rankingEnd),
    /evaluation|isPositive|groundTruth/,
    "GT status must be computed only after ranking and filtering",
  );

  const markerStart = gallery.indexOf("const evaluationPositive = Boolean(evaluation?.isPositive(item))");
  const markerEnd = gallery.indexOf("</article>", markerStart);
  const markerBlock = gallery.slice(markerStart, markerEnd);
  assert.ok(markerStart >= 0 && markerEnd > markerStart, "positive marker block should be locatable");
  assert.match(markerBlock, /gallery-card-evaluation-positive/);
  assert.match(markerBlock, /gallery-evaluation-positive-badge[\s\S]*?GT positive/);
  assert.doesNotMatch(
    markerBlock,
    /preferences|getPreference|FeedbackLabel|strong-positive/,
    "Frozen Test GT must not be inferred from personal feedback",
  );
  assert.match(gallery, /<GalleryLightbox[\s\S]*?evaluation=\{evaluation\}/);
  assert.match(lightbox, /const evaluationPositive = Boolean\(evaluation\?\.isPositive\(item\)\)/);
  assert.match(lightbox, /gallery-lightbox-dialog-evaluation-positive/);
  assert.match(lightbox, /gallery-lightbox-evaluation-badge[\s\S]*?GT positive/);
  assert.match(css, /\.gallery-card-evaluation-positive\s*\{[\s\S]*?border-color: var\(--green\)/);
  assert.match(css, /\.gallery-evaluation-positive-badge\s*\{[\s\S]*?pointer-events: none/);
  assert.match(css, /\.gallery-lightbox-dialog-evaluation-positive\s*\{[\s\S]*?border-color: var\(--green\)/);
});
