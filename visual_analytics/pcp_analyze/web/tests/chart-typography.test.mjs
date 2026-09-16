import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CHART_UI_FONT,
  CHART_UI_FONT_FALLBACK,
  chartCanvasFont,
} from "../app/lib/chartTypography.ts";

test("SVG chart typography follows the shared modern UI font", () => {
  assert.match(CHART_UI_FONT, /^var\(--font-ui,/);
  assert.match(CHART_UI_FONT_FALLBACK, /Inter.*sans-serif/);
  assert.doesNotMatch(CHART_UI_FONT, /ui-monospace|Georgia|Times New Roman/);
});

test("Canvas receives the computed font family, never an unresolved CSS variable", () => {
  const canvas = {
    ownerDocument: {
      defaultView: {
        getComputedStyle(element) {
          assert.equal(element, canvas);
          return { fontFamily: '"Inter Local", sans-serif' };
        },
      },
    },
  };
  assert.equal(chartCanvasFont(canvas, 11), '11px "Inter Local", sans-serif');
  assert.equal(chartCanvasFont(canvas, 10), '10px "Inter Local", sans-serif');
});

test("Canvas uses the sans-serif fallback without browser styles", () => {
  assert.equal(chartCanvasFont(null), `11px ${CHART_UI_FONT_FALLBACK}`);
  const canvas = {
    ownerDocument: { defaultView: { getComputedStyle: () => ({ fontFamily: " " }) } },
  };
  assert.equal(chartCanvasFont(canvas), `11px ${CHART_UI_FONT_FALLBACK}`);
});

test("all large chart renderers share typography without hard-coded monospace faces", async () => {
  for (const name of ["ParallelCoordinates", "ClusterSummaryParallelCoordinates", "ProjectionScatter"]) {
    const source = await readFile(new URL(`../app/components/${name}.tsx`, import.meta.url), "utf8");
    assert.match(source, /CHART_UI_FONT/, name);
    assert.doesNotMatch(source, /ui-monospace|SFMono-Regular|Menlo|Georgia|Times New Roman/, name);
  }
});

test("scatter redraws Canvas after font loading and keeps 11px coordinate labels", async () => {
  const source = await readFile(new URL("../app/components/ProjectionScatter.tsx", import.meta.url), "utf8");
  assert.match(source, /context\.font = chartCanvasFont\(canvas, 11\)/);
  assert.match(source, /document\.fonts\.ready\.then\(redrawWithLoadedFont\)/);
  assert.match(source, /addEventListener\("loadingdone", redrawWithLoadedFont\)/);
  assert.match(source, /removeEventListener\("loadingdone", redrawWithLoadedFont\)/);
  assert.match(source, /\[candidateMask, fontRevision, highlightCount/);
});

test("hierarchy rows expose distinct visual depths without changing axis row geometry", async () => {
  const source = await readFile(new URL("../app/components/HierarchicalPcpRail.tsx", import.meta.url), "utf8");
  assert.match(source, /hierarchical-pcp-row--depth-\$\{axis\.depth\}/);
  assert.match(source, /data-depth=\{axis\.depth\}/);
  assert.match(source, /data-node-role=\{nodeRole\}/);
  assert.match(source, /axis\.kind === "attribute-evidence" \|\| axis\.kind === "holistic" \? "evidence"/);
  assert.match(source, /const ROW_HEIGHT = 32;/);
  assert.match(source, /HIERARCHICAL_PCP_TOP \+ index \* rowGap/);
  assert.match(source, /centerY - ROW_HEIGHT \/ 2/);
});
