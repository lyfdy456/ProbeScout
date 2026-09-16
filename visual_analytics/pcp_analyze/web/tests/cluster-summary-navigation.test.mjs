import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  buildClusterSummaryNavigation,
  clusterSummaryDirectionForKey,
  finiteClusterProfileMean,
  moveClusterSummarySelection,
} from "../app/lib/clusterSummaryNavigation.ts";

const profile = (id, means) => ({ id, means });

test("finite PCP mean ignores non-finite axes and reports an empty profile", () => {
  assert.equal(
    finiteClusterProfileMean([0.1, Number.NaN, 0.5, Number.POSITIVE_INFINITY], [0, 1, 2, 3]),
    0.3,
  );
  assert.equal(finiteClusterProfileMean([Number.NaN, Number.NEGATIVE_INFINITY], [0, 1]), null);
  assert.equal(finiteClusterProfileMean([0.25], []), null);
});

test("navigation order follows visible-axis mean, numeric IDs, then stable source order", () => {
  const navigation = buildClusterSummaryNavigation([
    profile("beta", [0.4, 0.2]),
    profile("10", [0.2, 0.2]),
    profile(2, [0.1, 0.3]),
    profile("alpha", [0.2, 0.2]),
    profile("empty-b", [Number.NaN, Number.NaN]),
    profile("empty-a", [Number.NaN, Number.NaN]),
    profile(7, [0.9, 0.7]),
  ], [0, 1]);

  assert.deepEqual(
    navigation.entries.map((entry) => entry.profile.id),
    [2, "10", "alpha", "beta", 7, "empty-b", "empty-a"],
  );
  assert.deepEqual(navigation.entries.slice(0, 3).map((entry) => entry.mean), [0.2, 0.2, 0.2]);
  assert.ok(Math.abs(navigation.entries[3].mean - 0.3) < 1e-12);
  assert.equal(navigation.entries[4].mean, 0.8);
  assert.deepEqual(navigation.entries.slice(5).map((entry) => entry.mean), [null, null]);
  assert.equal(navigation.indexByKey.get("7"), 4);
});

test("no selection starts at a directional endpoint and a missing selection resets there", () => {
  const navigation = buildClusterSummaryNavigation([
    profile(4, [0.4]),
    profile(1, [0.1]),
    profile(3, [0.3]),
  ], [0]);

  assert.equal(moveClusterSummarySelection(navigation, null, 1), 1);
  assert.equal(moveClusterSummarySelection(navigation, undefined, -1), 4);
  assert.equal(moveClusterSummarySelection(navigation, "missing", 1), 1);
  assert.equal(moveClusterSummarySelection(navigation, "missing", -1), 4);
});

test("movement selects adjacent clusters and stops without wrapping at either edge", () => {
  const navigation = buildClusterSummaryNavigation([
    profile(1, [0.1]),
    profile(2, [0.2]),
    profile(3, [0.3]),
  ], [0]);

  assert.equal(moveClusterSummarySelection(navigation, 2, -1), 1);
  assert.equal(moveClusterSummarySelection(navigation, 2, 1), 3);
  assert.equal(moveClusterSummarySelection(navigation, 1, -1), null);
  assert.equal(moveClusterSummarySelection(navigation, 3, 1), null);
  assert.equal(moveClusterSummarySelection(buildClusterSummaryNavigation([], [0]), null, 1), null);
});

test("only ArrowLeft, ArrowRight, A and D map to cluster navigation", () => {
  assert.equal(clusterSummaryDirectionForKey("ArrowLeft"), -1);
  assert.equal(clusterSummaryDirectionForKey("a"), -1);
  assert.equal(clusterSummaryDirectionForKey("A"), -1);
  assert.equal(clusterSummaryDirectionForKey("ArrowRight"), 1);
  assert.equal(clusterSummaryDirectionForKey("d"), 1);
  assert.equal(clusterSummaryDirectionForKey("D"), 1);
  for (const key of ["Left", "Right", "ArrowUp", "ArrowDown", "w", "s", " ", ""]) {
    assert.equal(clusterSummaryDirectionForKey(key), null);
  }
});

test("cluster PCP wires guarded keyboard navigation, accessible metadata, and select semantics", async () => {
  const [component, dashboard] = await Promise.all([
    readFile(new URL("../app/components/ClusterSummaryParallelCoordinates.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(component, /window\.addEventListener\("keydown", handleKeyDown\)/);
  assert.match(component, /dialog\[open\],[\s\S]*?aria-modal/);
  assert.match(component, /event\.isComposing[\s\S]*?event\.ctrlKey[\s\S]*?event\.metaKey[\s\S]*?event\.altKey/);
  assert.match(component, /"input",[\s\S]*?"select",[\s\S]*?"textarea"/);
  assert.match(component, /aria-keyshortcuts="ArrowLeft ArrowRight A D"/);
  assert.doesNotMatch(component, /Current PCP mean: low → high|← \/ A|→ \/ D/);
  assert.match(component, /data-cluster-selected=\{selected \? "true" : undefined\}/);
  assert.match(dashboard, /const selectRankClusterSummary = useCallback[\s\S]*?setRankClusterFilter\(String\(cluster\)\)/);
  assert.match(dashboard, /onClusterSelect=\{selectRankClusterSummary\}/);
});

test("recognized cluster navigation keys prevent scrolling even at an endpoint", async () => {
  const component = await readFile(
    new URL("../app/components/ClusterSummaryParallelCoordinates.tsx", import.meta.url),
    "utf8",
  );
  const direction = component.indexOf("const direction = clusterSummaryDirectionForKey(event.key)");
  const ignoredKeyReturn = component.indexOf("if (direction === null) return", direction);
  const preventDefault = component.indexOf("event.preventDefault()", ignoredKeyReturn);
  const move = component.indexOf("const nextCluster = moveClusterSummarySelection", ignoredKeyReturn);
  const endpointReturn = component.indexOf("if (nextCluster === null) return", move);

  assert.notEqual(direction, -1);
  assert.ok(ignoredKeyReturn > direction);
  assert.ok(preventDefault > ignoredKeyReturn);
  assert.ok(move > preventDefault, "recognized keys must be canceled before an endpoint is resolved");
  assert.ok(endpointReturn > move);
});

test("cluster PCP exposes double-click drill-down on legend buttons and centroid paths", async () => {
  const component = await readFile(
    new URL("../app/components/ClusterSummaryParallelCoordinates.tsx", import.meta.url),
    "utf8",
  );

  assert.match(component, /onClusterDoubleClick\?: \(cluster: ClusterSummaryId\) => void/);
  assert.doesNotMatch(component, /double-click a curve to inspect its images/);
  assert.equal(
    component.match(/onDoubleClick=/g)?.length,
    2,
    "both the legend control and the drawn centroid must open drill-down",
  );
  assert.match(component, /if \(event\.detail > 1\) return/);
  assert.match(component, /onClusterDoubleClick\?\.\(profile\.id\)/);
});
