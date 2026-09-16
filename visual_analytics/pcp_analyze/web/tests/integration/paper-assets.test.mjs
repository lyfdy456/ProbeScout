import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve, relative, isAbsolute } from "node:path";

const sourceWeb = new URL("../../", import.meta.url);
const root = new URL("../../../", sourceWeb);
const web = process.env.PROBESCOUT_ASSET_WEB
  ? pathToFileURL(resolve(process.env.PROBESCOUT_ASSET_WEB) + "/")
  : sourceWeb;
const json = async (url) => JSON.parse(await readFile(url, "utf8"));

test("Main17 assets preserve image IDs, aligned binary files and fixed query targets", async () => {
  const scope = await json(new URL("manifests/paper_main17.json", root));
  const catalog = await json(new URL("public/data/catalog.json", web));
  const tasks = new Map(catalog.datasets.flatMap((d) => d.tasks.map((t) => [t.id, t])));
  assert.equal(scope.tasks.length, 17);
  for (const expected of scope.tasks) {
    const entry = tasks.get(expected.task_id);
    assert.ok(entry, `Missing paper task: ${expected.task_id}`);
    const folder = new URL(`public/${entry.dataRoot.replace(/^\//, "").replace(/\/?$/, "/")}`, web);
    const manifest = await json(new URL("manifest.json", folder));
    const base = fileURLToPath(folder);
    for (const [key, spec] of Object.entries(manifest.files)) {
      if (!spec.path || !spec.sha256 || typeof spec.bytes !== "number") continue;
      const target = resolve(base, spec.path);
      const rel = relative(base, target);
      assert.ok(!rel.startsWith("..") && !isAbsolute(rel), `${key}: path escapes task`);
      const bytes = await readFile(target);
      assert.equal(bytes.length, spec.bytes, `${expected.task_id}/${key}: bytes`);
      assert.equal(createHash("sha256").update(bytes).digest("hex"), spec.sha256, `${key}: checksum`);
    }
    const ids = await json(new URL(manifest.files.imageIds.path, folder));
    assert.equal(ids.length, manifest.rowCount);
    assert.equal(new Set(ids).size, ids.length);
    assert.deepEqual(manifest.retrievalTargets.filter((t) => t.kind === "attribute").map((t) => t.id), expected.attribute_ids);
    assert.ok(manifest.retrievalTargets.some((target) => target.id === manifest.defaultRetrievalTarget));
    assert.ok(manifest.query.text.trim());
    for (const query of manifest.query.images) {
      assert.equal(ids[query.imageIndex], query.imageId, `${expected.task_id}: query row identity`);
    }
  }
});
