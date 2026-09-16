import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

register(
  new URL("./support/extensionless-typescript-loader.mjs", import.meta.url),
  import.meta.url,
);

const {
  RUNTIME_RANK_PROFILE_CLUSTER_ALGORITHM,
  RUNTIME_RANK_PROFILE_CLUSTER_API_PATH,
  RUNTIME_RANK_PROFILE_CLUSTER_COUNTS,
  RUNTIME_RANK_PROFILE_CLUSTER_FEATURE_BASIS,
  clearRuntimeRankProfileClustersMemoryCache,
  requestRuntimeRankProfileClusters,
  runtimeRankProfileClusterKey,
} = await import("../app/lib/runtimeRankProfileClusters.ts");
const {
  HIERARCHICAL_PCP_METHODS,
  createDefaultHierarchicalPcpConfig,
  updateHierarchicalAttributeWeight,
  updateHierarchicalMethodWeight,
} = await import("../app/lib/hierarchicalPcp.ts");

const attributes = ["cat", "hugging"];

function input(overrides = {}) {
  return {
    taskId: "runtime-cluster-task",
    rowCount: 120,
    targetCount: 3,
    attributeIds: attributes,
    config: createDefaultHierarchicalPcpConfig(attributes),
    scheme: "fine30",
    ...overrides,
  };
}

function labels(rowCount, clusters) {
  return Uint8Array.from({ length: rowCount }, (_, index) => index % clusters);
}

function responseFor(
  requestInput,
  overrides = {},
) {
  const clusterCount = RUNTIME_RANK_PROFILE_CLUSTER_COUNTS[requestInput.scheme];
  const body = overrides.body ?? labels(requestInput.rowCount, clusterCount);
  const headers = new Headers({
    "Content-Type": "application/octet-stream",
    "X-PCP-Algorithm": RUNTIME_RANK_PROFILE_CLUSTER_ALGORITHM,
    "X-PCP-Layout": "labels",
    "X-PCP-Row-Count": String(requestInput.rowCount),
    "X-PCP-Target-Count": String(requestInput.targetCount),
    "X-PCP-Cluster-Scheme": requestInput.scheme,
    "X-PCP-Cluster-Count": String(clusterCount),
    "X-PCP-Feature-Count": String(
      1 + attributes.length * (1 + HIERARCHICAL_PCP_METHODS.length),
    ),
    "X-PCP-Feature-Basis": RUNTIME_RANK_PROFILE_CLUSTER_FEATURE_BASIS,
    "X-PCP-Fit-Scope": "development",
    "X-PCP-Fit-Row-Count": "80",
    ...overrides.headers,
  });
  return new Response(body, {
    status: overrides.status ?? 200,
    headers,
  });
}

function installFetch(implementation) {
  const previous = globalThis.fetch;
  globalThis.fetch = implementation;
  return () => {
    globalThis.fetch = previous;
  };
}

test("runtime cluster schemes expose the audited K contract and semantic cache key", () => {
  assert.equal(
    RUNTIME_RANK_PROFILE_CLUSTER_ALGORITHM,
    "dynamic-attribute-rank-fusion-clusters-v1",
  );
  assert.equal(
    RUNTIME_RANK_PROFILE_CLUSTER_FEATURE_BASIS,
    "attribute-hierarchy-13-rank-v1",
  );
  assert.deepEqual(RUNTIME_RANK_PROFILE_CLUSTER_COUNTS, {
    fine30: 30,
    fine50: 50,
    fine100: 100,
    absolute: 4,
    shape: 3,
  });

  const baseline = input();
  const scaled = input({ config: createDefaultHierarchicalPcpConfig(attributes, 7) });
  assert.equal(runtimeRankProfileClusterKey(baseline), runtimeRankProfileClusterKey(scaled));
  assert.notEqual(
    runtimeRankProfileClusterKey(baseline),
    runtimeRankProfileClusterKey(input({ scheme: "shape" })),
  );
  assert.notEqual(
    runtimeRankProfileClusterKey(baseline),
    runtimeRankProfileClusterKey(input({ targetCount: 4 })),
  );

  const changed = updateHierarchicalMethodWeight(
    baseline.config,
    "cat",
    HIERARCHICAL_PCP_METHODS[0],
    4,
  );
  assert.notEqual(
    runtimeRankProfileClusterKey(baseline),
    runtimeRankProfileClusterKey(input({ config: changed })),
  );

  const changedGlobal = updateHierarchicalAttributeWeight(baseline.config, "cat", 4);
  assert.notEqual(
    runtimeRankProfileClusterKey(baseline),
    runtimeRankProfileClusterKey(input({ config: changedGlobal })),
  );
});

test("request parses the binary contract, posts normalized weights, and fills the shared cache", async () => {
  clearRuntimeRankProfileClustersMemoryCache();
  const requestInput = input();
  const calls = [];
  const restore = installFetch(async (url, init) => {
    calls.push({ url: String(url), init });
    return responseFor(requestInput);
  });
  try {
    const first = await requestRuntimeRankProfileClusters(requestInput);
    const equivalent = await requestRuntimeRankProfileClusters(input({
      config: createDefaultHierarchicalPcpConfig(attributes, 9),
    }));
    assert.strictEqual(first, equivalent);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, RUNTIME_RANK_PROFILE_CLUSTER_API_PATH);
    assert.equal(calls[0].init.method, "POST");
    const payload = JSON.parse(calls[0].init.body);
    assert.equal(payload.taskId, requestInput.taskId);
    assert.equal(payload.scheme, "fine30");
    assert.deepEqual(payload.attributeWeights, { cat: 0.5, hugging: 0.5 });
    for (const attribute of attributes) {
      assert.deepEqual(
        payload.methodWeightsByAttribute[attribute],
        first.config.methodWeightsByAttribute[attribute],
      );
      for (const method of HIERARCHICAL_PCP_METHODS) {
        assert.equal(payload.methodWeightsByAttribute[attribute][method], 1 / 13);
      }
    }
    assert.equal(first.clusterCount, 30);
    assert.equal(first.labels.length, 120);
    assert.equal(first.featureCount, 29);
    assert.equal(first.fitRowCount, 80);
    assert.equal(typeof first.configFingerprint, "string");
    assert.ok(first.configFingerprint.length > 0);
  } finally {
    restore();
  }
});

test("concurrent callers share one in-flight computation", async () => {
  clearRuntimeRankProfileClustersMemoryCache();
  const requestInput = input({ scheme: "fine50" });
  let calls = 0;
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const restore = installFetch(async () => {
    calls += 1;
    await gate;
    return responseFor(requestInput);
  });
  try {
    const first = requestRuntimeRankProfileClusters(requestInput);
    const second = requestRuntimeRankProfileClusters(requestInput);
    assert.equal(calls, 1);
    release();
    const [left, right] = await Promise.all([first, second]);
    assert.strictEqual(left, right);
    assert.equal(calls, 1);
  } finally {
    restore();
  }
});

test("one aborted subscriber does not cancel or poison the shared computation", async () => {
  clearRuntimeRankProfileClustersMemoryCache();
  const requestInput = input({ scheme: "absolute" });
  let calls = 0;
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const restore = installFetch(async () => {
    calls += 1;
    await gate;
    return responseFor(requestInput);
  });
  try {
    const controller = new AbortController();
    const aborted = requestRuntimeRankProfileClusters(requestInput, {
      signal: controller.signal,
    });
    controller.abort();
    await assert.rejects(aborted, (error) => error?.name === "AbortError");
    release();
    const completed = await requestRuntimeRankProfileClusters(requestInput);
    assert.equal(completed.clusterCount, 4);
    assert.equal(calls, 1);
  } finally {
    restore();
  }
});

test("request rejects mismatched metadata, malformed labels, and invalid optional audit headers", async (t) => {
  const requestInput = input();
  const cases = [
    {
      name: "algorithm",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Algorithm": "other" },
      }),
      pattern: /Unsupported runtime rank-profile clustering algorithm/,
    },
    {
      name: "layout",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Layout": "row-major" },
      }),
      pattern: /response layout/,
    },
    {
      name: "scheme",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Cluster-Scheme": "fine50" },
      }),
      pattern: /scheme fine50 does not match fine30/,
    },
    {
      name: "truncated body",
      response: () => responseFor(requestInput, { body: new Uint8Array(119) }),
      pattern: /119 bytes; expected 120/,
    },
    {
      name: "out-of-range label",
      response: () => {
        const body = labels(120, 30);
        body[0] = 30;
        return responseFor(requestInput, { body });
      },
      pattern: /out-of-range label 30/,
    },
    {
      name: "missing cluster",
      response: () => responseFor(requestInput, {
        body: Uint8Array.from({ length: 120 }, (_, index) => index % 29),
      }),
      pattern: /did not assign cluster 29/,
    },
    {
      name: "invalid feature count",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Feature-Count": "0" },
      }),
      pattern: /invalid X-PCP-Feature-Count/,
    },
    {
      name: "wrong feature basis",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Feature-Basis": "joint-only" },
      }),
      pattern: /feature basis/,
    },
    {
      name: "wrong fit scope",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Fit-Scope": "all" },
      }),
      pattern: /not fitted on Development/,
    },
    {
      name: "fit count exceeds gallery",
      response: () => responseFor(requestInput, {
        headers: { "X-PCP-Fit-Row-Count": "121" },
      }),
      pattern: /fit row count exceeds/,
    },
  ];

  for (const current of cases) {
    await t.test(current.name, async () => {
      clearRuntimeRankProfileClustersMemoryCache();
      const restore = installFetch(async () => current.response());
      try {
        await assert.rejects(
          requestRuntimeRankProfileClusters(requestInput),
          current.pattern,
        );
      } finally {
        restore();
      }
    });
  }
});

test("runtime hook source contains request-generation and abort guards", async () => {
  const source = await readFile(
    new URL("../app/lib/runtimeRankProfileClusters.ts", import.meta.url),
    "utf8",
  );
  assert.match(source, /const generation = useRef\(0\)/);
  assert.match(source, /requestGeneration !== generation\.current/);
  assert.match(source, /const controller = useRef<AbortController \| null>/);
  assert.match(source, /controller\.current\?\.abort\(\)/);
  assert.match(source, /const prefetched = disabled \? null : cachedResult\(sourceKey\)/);
});
