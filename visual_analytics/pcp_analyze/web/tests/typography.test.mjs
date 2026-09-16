import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function source(relativePath) {
  return readFile(new URL(`../${relativePath}`, import.meta.url), "utf8");
}

test("interface typography uses local Inter with 10–11px notes and 12px controls", async () => {
  const [css, clusterSummary, projection] = await Promise.all([
    source("app/globals.css"),
    source("app/components/ClusterSummaryParallelCoordinates.tsx"),
    source("app/components/ProjectionScatter.tsx"),
  ]);

  const fixedCssSizes = [...css.matchAll(/font-size:\s*(\d+)px/g)]
    .map((match) => Number(match[1]));
  assert.equal(fixedCssSizes.filter((size) => size < 10).length, 0);
  assert.match(css, /--text-body:\s*12px/);
  assert.match(css, /--text-meta:\s*11px/);
  assert.match(css, /--text-micro:\s*10px/);
  assert.match(css, /--font-ui:\s*"Inter"/);
  assert.match(css, /url\("\/fonts\/inter\/inter-latin\.woff2"\)/);
  assert.doesNotMatch(css, /font-family:\s*ui-monospace/);
  assert.doesNotMatch(clusterSummary, /fontSize(?:=\{|:\s*)10\b/);
  assert.match(projection, /chartCanvasFont/);
});

test("common 1280 and 1366px viewports use the readable two-column layout", async () => {
  const css = await source("app/globals.css");
  assert.match(
    css,
    /@media \(max-width: 1399px\) \{[\s\S]*?\.workspace-grid \{[\s\S]*?grid-template-columns: minmax\(360px, 0\.9fr\) minmax\(0, 1\.1fr\)/,
  );
  assert.match(
    css,
    /@media \(max-width: 1399px\) \{[\s\S]*?\.optimization-panel \{[\s\S]*?grid-column: 1 \/ -1/,
  );
});

test("requested column and module titles use larger shared sizes without enlarging body text", async () => {
  const [css, clusterSummary] = await Promise.all([
    source("app/globals.css"), source("app/components/ClusterSummaryParallelCoordinates.tsx"),
  ]);
  assert.match(css, /--text-column-title:\s*20px/);
  assert.match(css, /--text-module-title:\s*18px/);
  assert.match(css, /#dashboard-heading,\s*#visualization-heading,\s*#optimization-heading,\s*#test-audit-heading\s*\{\s*font-size: var\(--text-column-title\)/);
  assert.match(css, /\.section-label span\s*\{[^}]*font-size: var\(--text-module-title\)/);
  assert.match(css, /\.projection-title,\s*\.gallery-heading\s*\{[^}]*font-size: var\(--text-module-title\)/);
  assert.doesNotMatch(css, /\.projection-title,\s*\.gallery-heading\s*\{[^}]*font-size: \d+px/);
  assert.match(css, /\.pcp-summary-title\s*\{[^}]*font-size: var\(--text-module-title\)/);
  assert.match(clusterSummary, /<strong className="pcp-summary-title">\{title\}<\/strong>/);
  assert.match(css, /--text-body:\s*12px/);
  assert.match(css, /--text-meta:\s*11px/);
});

test("dense legends stay two rows tall and long brush counts never wrap", async () => {
  const [css, dashboard] = await Promise.all([
    source("app/globals.css"), source("app/Dashboard.tsx"),
  ]);
  assert.match(css, /\.cluster-legend--dense\s*\{[^}]*grid-auto-rows: 26px;[^}]*height: 63px;[^}]*overflow-y: auto/s);
  assert.match(css, /\.cluster-legend--dense \.legend-item small\s*\{[^}]*text-overflow: ellipsis;[^}]*white-space: nowrap/s);
  assert.match(dashboard, /title=\{`\$\{summary\.label_zh\} · \$\{activePcpHighlightMask/);
});

test("overlap accessibility table is clipped by a wrapper, not intrinsic table width", async () => {
  const overlap = await source("app/components/SelectionOverlapPanel.tsx");
  assert.match(overlap, /<div className="sr-only">\s*<table>/);
  assert.doesNotMatch(overlap, /<table className="sr-only">/);
});

test("mobile dataset controls keep intrinsic height in the column header", async () => {
  const css = await source("app/globals.css");
  assert.match(
    css,
    /@media \(max-width: 760px\) \{[\s\S]*?\.dataset-task-switcher \{[\s\S]*?flex: 0 0 auto/,
  );
  assert.match(
    css,
    /@media \(max-width: 440px\) \{[\s\S]*?\.dataset-task-switcher label:nth-child\(2\),[\s\S]*?\.dataset-task-switcher \.data-purpose-switcher \{[\s\S]*?flex: 0 0 auto/,
  );
});
