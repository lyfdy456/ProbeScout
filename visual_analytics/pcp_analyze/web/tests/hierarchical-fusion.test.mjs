import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register(
  new URL("./support/extensionless-typescript-loader.mjs", import.meta.url),
  import.meta.url,
);

const {
  HIERARCHICAL_FUSION_ALGORITHM,
  HIERARCHICAL_FUSION_API_PATH,
  HIERARCHICAL_FUSION_LAYOUT,
  requestHierarchicalFusion,
} = await import("../app/lib/hierarchicalFusion.ts");
const {
  HIERARCHICAL_PCP_METHODS,
  createDefaultHierarchicalPcpConfig,
} = await import("../app/lib/hierarchicalPcp.ts");

const attributes = ["cat", "hugging"];
const dimensions = {
  taskId: "hierarchical-fusion-task",
  rowCount: 2,
  targetCount: 3,
  attributeIds: attributes,
};

function installFetch(implementation) {
  const previous = globalThis.fetch;
  globalThis.fetch = implementation;
  return () => {
    globalThis.fetch = previous;
  };
}

function responseFor(body, headers = {}) {
  return new Response(body, {
    headers: {
      "Content-Type": "application/octet-stream",
      "X-PCP-Algorithm": HIERARCHICAL_FUSION_ALGORITHM,
      "X-PCP-Layout": HIERARCHICAL_FUSION_LAYOUT.join(","),
      "X-PCP-Row-Count": String(dimensions.rowCount),
      "X-PCP-Target-Count": String(dimensions.targetCount),
      ...headers,
    },
  });
}

function weightedConfig() {
  const config = createDefaultHierarchicalPcpConfig(attributes);
  config.attributeWeights = { cat: 2, hugging: 6 };
  config.methodWeightsByAttribute.cat = Object.fromEntries(
    HIERARCHICAL_PCP_METHODS.map((method, index) => [method, index + 1]),
  );
  config.methodWeightsByAttribute.hugging = Object.fromEntries(
    HIERARCHICAL_PCP_METHODS.map((method, index) => [
      method,
      HIERARCHICAL_PCP_METHODS.length - index,
    ]),
  );
  return config;
}

test("rank hierarchy request posts A x 13 plus A normalized weights and parses three arrays", async () => {
  assert.equal(HIERARCHICAL_FUSION_ALGORITHM, "hierarchical-rank-fusion-13-v1");
  assert.deepEqual(HIERARCHICAL_FUSION_LAYOUT, ["raw", "calibrated", "rank"]);

  const count = dimensions.rowCount * dimensions.targetCount;
  const values = Float32Array.from(
    { length: count * HIERARCHICAL_FUSION_LAYOUT.length },
    (_, index) => index + 1,
  );
  const calls = [];
  const restore = installFetch(async (url, init) => {
    calls.push({ url: String(url), init });
    return responseFor(values.buffer);
  });
  try {
    const result = await requestHierarchicalFusion({
      ...dimensions,
      config: weightedConfig(),
    });

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, HIERARCHICAL_FUSION_API_PATH);
    assert.equal(calls[0].init.method, "POST");
    const payload = JSON.parse(calls[0].init.body);
    assert.equal(payload.taskId, dimensions.taskId);
    assert.deepEqual(payload.attributeWeights, { cat: 0.25, hugging: 0.75 });
    assert.equal(Object.keys(payload.methodWeightsByAttribute.cat).length, 13);
    assert.equal(Object.keys(payload.methodWeightsByAttribute.hugging).length, 13);
    for (const [index, method] of HIERARCHICAL_PCP_METHODS.entries()) {
      assert.equal(payload.methodWeightsByAttribute.cat[method], (index + 1) / 91);
      assert.equal(
        payload.methodWeightsByAttribute.hugging[method],
        (HIERARCHICAL_PCP_METHODS.length - index) / 91,
      );
    }

    assert.deepEqual([...result.rawScores], [...values.slice(0, count)]);
    assert.deepEqual([...result.calibratedScores], [...values.slice(count, count * 2)]);
    assert.deepEqual([...result.ranks], [...values.slice(count * 2, count * 3)]);
    assert.deepEqual(result.config.attributeWeights, payload.attributeWeights);
    assert.deepEqual(
      result.config.methodWeightsByAttribute,
      payload.methodWeightsByAttribute,
    );
  } finally {
    restore();
  }
});

test("rank hierarchy request rejects a payload whose three-array byte length is wrong", async () => {
  const count = dimensions.rowCount * dimensions.targetCount;
  const truncated = new Float32Array(
    count * HIERARCHICAL_FUSION_LAYOUT.length - 1,
  );
  const restore = installFetch(async () => responseFor(truncated.buffer));
  try {
    await assert.rejects(
      requestHierarchicalFusion({ ...dimensions, config: weightedConfig() }),
      /payload has \d+ bytes; expected \d+/,
    );
  } finally {
    restore();
  }
});
