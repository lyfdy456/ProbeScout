import { methodDisplayLabel } from "./methodDisplay.js";

export const HIERARCHICAL_PCP_LEARNERS = [
  "MLP",
  "K-Fold",
  "Triplet Loss",
  "Attention Pooling",
  "Attribute-conditioned Attention",
  "nnPU",
  "DC-PU",
  "Ours-PURA",
] as const;

export const HIERARCHICAL_PCP_EMBEDDING_BASELINES = [
  "Image Prototype",
  "Query MaxSim",
  "Image--Text Fusion",
  "Text Prompt Ensemble",
  "Z-score Image--Text Fusion",
] as const;

export const REFINEMENT_PCP_DEFAULT_EMBEDDING_METHODS = [
  "Query MaxSim", "Text Prompt Ensemble",
] as const;

/**
 * Canonical order for the thirteen equal-level members fused inside every
 * attribute aggregate. Keep this aligned with the tensor method manifest.
 */
export const HIERARCHICAL_PCP_METHODS = [
  ...HIERARCHICAL_PCP_EMBEDDING_BASELINES,
  ...HIERARCHICAL_PCP_LEARNERS,
] as const;

export type HierarchicalPcpLearner = typeof HIERARCHICAL_PCP_LEARNERS[number];
export type HierarchicalPcpEmbeddingBaseline =
  typeof HIERARCHICAL_PCP_EMBEDDING_BASELINES[number];
export type HierarchicalPcpMethod = typeof HIERARCHICAL_PCP_METHODS[number];
export type HierarchicalPcpMethodWeights = Record<HierarchicalPcpMethod, number>;

export const HIERARCHICAL_PCP_TOP = 54;
export const HIERARCHICAL_PCP_BOTTOM_PADDING = 34;

export type HierarchicalPcpAxisKind =
  | "full"
  | "attribute-evidence"
  | "attribute"
  | "holistic"
  | "baseline"
  | "learner";

export interface HierarchicalPcpAttribute {
  id: string;
  label: string;
}

export interface HierarchicalPcpAxis {
  id: string;
  kind: HierarchicalPcpAxisKind;
  depth: 0 | 1 | 2 | 3;
  label: string;
  description?: string;
  targetId: string;
  attributeId?: string;
  methodId?: HierarchicalPcpMethod;
}

export interface BuildHierarchicalPcpAxesInput {
  attributes: readonly HierarchicalPcpAttribute[];
  jointTargetId: string;
  fullExpanded: boolean;
  expandedAttributeIds: ReadonlySet<string> | readonly string[];
  fullLabel?: string;
}

export interface BuildRefinementPcpAxesInput extends BuildHierarchicalPcpAxesInput {
  attributeEvidenceExpanded?: boolean;
  holisticExpanded: boolean;
  embeddingMethods?: readonly HierarchicalPcpEmbeddingBaseline[];
}

export interface HierarchicalPcpConfig {
  attributeWeights: Record<string, number>;
  methodWeightsByAttribute: Record<string, HierarchicalPcpMethodWeights>;
}

const WEIGHT_EPSILON = 1e-10;

function encoded(value: string) {
  return encodeURIComponent(value);
}

function owns(record: object, key: string) {
  return Object.prototype.hasOwnProperty.call(record, key);
}

export function hierarchicalFullAxisId(jointTargetId: string) {
  return `hierarchical::full::${encoded(jointTargetId)}`;
}

export function hierarchicalAttributeAxisId(attributeId: string) {
  return `hierarchical::attribute::${encoded(attributeId)}`;
}

export function hierarchicalMethodAxisId(
  attributeId: string,
  methodId: HierarchicalPcpMethod,
) {
  return `hierarchical::method::${encoded(attributeId)}::${encoded(methodId)}`;
}

export function refinementHolisticAxisId(jointTargetId: string) {
  return `refinement::holistic::${encoded(jointTargetId)}`;
}

export function refinementAttributeEvidenceAxisId(jointTargetId: string) {
  return `refinement::attribute-evidence::${encoded(jointTargetId)}`;
}

export function refinementEmbeddingAxisId(
  jointTargetId: string,
  methodId: HierarchicalPcpEmbeddingBaseline,
) {
  return `refinement::embedding::${encoded(jointTargetId)}::${encoded(methodId)}`;
}

function assertAttributeIds(attributeIds: readonly string[]) {
  if (attributeIds.length === 0) {
    throw new Error("Hierarchical PCP requires at least one attribute.");
  }
  const unique = new Set(attributeIds);
  if (unique.size !== attributeIds.length || attributeIds.some((id) => !id.trim())) {
    throw new Error("Hierarchical PCP attribute IDs must be unique and non-empty.");
  }
}

function assertExactKeys(
  record: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  label: string,
) {
  const expectedSet = new Set(expected);
  const missing = expected.filter((key) => !owns(record, key));
  if (missing.length > 0) {
    throw new Error(`${label} are missing: ${missing.join(", ")}.`);
  }
  const unknown = Object.keys(record).filter((key) => !expectedSet.has(key));
  if (unknown.length > 0) {
    throw new Error(`${label} contain unknown entries: ${unknown.join(", ")}.`);
  }
}

function assertWeight(value: number, label: string) {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a finite non-negative number.`);
  }
  return value;
}

function normalizedFingerprintValue(value: number) {
  return Number(value.toPrecision(12));
}

function isEmbeddingBaseline(
  methodId: HierarchicalPcpMethod,
): methodId is HierarchicalPcpEmbeddingBaseline {
  return (HIERARCHICAL_PCP_EMBEDDING_BASELINES as readonly string[]).includes(methodId);
}

export function buildHierarchicalPcpAxes({
  attributes,
  jointTargetId,
  fullExpanded,
  expandedAttributeIds,
  fullLabel = "Weighted Fusion overall",
}: BuildHierarchicalPcpAxesInput): HierarchicalPcpAxis[] {
  assertAttributeIds(attributes.map((attribute) => attribute.id));
  if (!jointTargetId.trim()) throw new Error("Hierarchical PCP requires a Joint target ID.");

  const expanded = new Set(expandedAttributeIds);
  const axes: HierarchicalPcpAxis[] = [{
    id: hierarchicalFullAxisId(jointTargetId),
    kind: "full",
    depth: 0,
    label: fullLabel,
    targetId: jointTargetId,
  }];
  if (!fullExpanded) return axes;

  for (const attribute of attributes) {
    axes.push({
      id: hierarchicalAttributeAxisId(attribute.id),
      kind: "attribute",
      depth: 1,
      label: attribute.label,
      targetId: attribute.id,
      attributeId: attribute.id,
    });
    if (!expanded.has(attribute.id)) continue;

    for (const method of HIERARCHICAL_PCP_METHODS) {
      axes.push({
        id: hierarchicalMethodAxisId(attribute.id, method),
        kind: isEmbeddingBaseline(method) ? "baseline" : "learner",
        depth: 2,
        label: methodDisplayLabel(method),
        // Embedding baselines have one query-image ranking for the whole task;
        // their copy under each attribute intentionally reads canonical Joint.
        targetId: isEmbeddingBaseline(method) ? jointTargetId : attribute.id,
        attributeId: attribute.id,
        methodId: method,
      });
    }
  }
  return axes;
}

/**
 * Exact visible hierarchy for the score-level refinement model.  Embedding
 * retrieval heads are one global Holistic branch; they must never be copied
 * into every attribute as if they were members of the per-attribute beta.
 */
export function buildRefinementPcpAxes({
  attributes,
  jointTargetId,
  fullExpanded,
  expandedAttributeIds,
  holisticExpanded,
  attributeEvidenceExpanded = true,
  embeddingMethods = REFINEMENT_PCP_DEFAULT_EMBEDDING_METHODS,
  fullLabel = "Overall",
}: BuildRefinementPcpAxesInput): HierarchicalPcpAxis[] {
  assertAttributeIds(attributes.map((attribute) => attribute.id));
  if (!jointTargetId.trim()) throw new Error("Refinement PCP requires a Joint target ID.");
  if (!embeddingMethods.length || new Set(embeddingMethods).size !== embeddingMethods.length
    || embeddingMethods.some((method) => !isEmbeddingBaseline(method))) {
    throw new Error("Refinement embedding methods must be unique supported methods.");
  }
  const expanded = new Set(expandedAttributeIds);
  const axes: HierarchicalPcpAxis[] = [{
    id: hierarchicalFullAxisId(jointTargetId),
    kind: "full",
    depth: 0,
    label: fullLabel,
    description: "Overall relevance F = C × [(1 − λ) + λH].",
    targetId: jointTargetId,
  }];
  if (!fullExpanded) return axes;

  axes.push({
    id: refinementAttributeEvidenceAxisId(jointTargetId),
    kind: "attribute-evidence",
    depth: 1,
    label: "Attribute evidence",
    description: "Attribute evidence C = ∏ gₐ^γₐ; conjunctive support from all required attributes.",
    targetId: jointTargetId,
  });
  if (attributeEvidenceExpanded) {
    for (const attribute of attributes) {
      axes.push({
        id: hierarchicalAttributeAxisId(attribute.id),
        kind: "attribute",
        depth: 2,
        label: attribute.label,
        description: `${attribute.label}: SoftGate attribute satisfaction gₐ; γₐ controls its conjunction strength.`,
        targetId: attribute.id,
        attributeId: attribute.id,
      });
      if (!expanded.has(attribute.id)) continue;
      for (const method of HIERARCHICAL_PCP_LEARNERS) {
        axes.push({
          id: hierarchicalMethodAxisId(attribute.id, method),
          kind: "learner",
          depth: 3,
          label: methodDisplayLabel(method),
          description: `${methodDisplayLabel(method)}: normalized probe score z for ${attribute.label}; β is its within-attribute weight.`,
          targetId: attribute.id,
          attributeId: attribute.id,
          methodId: method,
        });
      }
    }
  }

  axes.push({
    id: refinementHolisticAxisId(jointTargetId),
    kind: "holistic",
    depth: 1,
    label: "Embedding evidence",
    description: "Embedding evidence H = ∑ ηᵣeᵣ; whole-query similarity, controlled by λ at the Overall layer.",
    targetId: jointTargetId,
  });
  if (holisticExpanded) {
    for (const method of embeddingMethods) {
      axes.push({
        id: refinementEmbeddingAxisId(jointTargetId, method),
        kind: "baseline",
        depth: 2,
        label: method,
        description: `${method}: normalized whole-query embedding score e; η is its weight in H.`,
        targetId: jointTargetId,
        methodId: method,
      });
    }
  }
  return axes;
}

export function refinementPcpComponentIds(
  attributeIds: readonly string[],
  jointTargetId: string,
  embeddingMethods: readonly HierarchicalPcpEmbeddingBaseline[] = REFINEMENT_PCP_DEFAULT_EMBEDDING_METHODS,
): string[] {
  const attributes = attributeIds.map((id) => ({ id, label: id }));
  return buildRefinementPcpAxes({
    attributes,
    jointTargetId,
    fullExpanded: true,
    attributeEvidenceExpanded: true,
    expandedAttributeIds: attributeIds,
    holisticExpanded: true,
    embeddingMethods,
  }).map((axis) => axis.id);
}

export function createDefaultHierarchicalPcpConfig(
  attributeIds: readonly string[],
  value = 1,
): HierarchicalPcpConfig {
  assertAttributeIds(attributeIds);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error("Default hierarchical weights must be finite and greater than zero.");
  }
  return {
    attributeWeights: Object.fromEntries(
      attributeIds.map((attributeId) => [attributeId, value]),
    ),
    methodWeightsByAttribute: Object.fromEntries(
      attributeIds.map((attributeId) => [
        attributeId,
        Object.fromEntries(
          HIERARCHICAL_PCP_METHODS.map((method) => [method, value]),
        ) as HierarchicalPcpMethodWeights,
      ]),
    ),
  };
}

/**
 * Converts a Tune run into the exact configuration used by the PCP and fusion
 * endpoint. Every row must contain all thirteen methods: an older eight-learner
 * matrix cannot be silently promoted by inventing five baseline weights.
 */
export function hierarchicalPcpConfigFromTuningWeights(
  methodWeightsByAttribute: Readonly<
    Record<string, Readonly<Record<string, number>>>
  >,
  attributeIds: readonly string[],
  attributeWeights?: Readonly<Record<string, number>>,
): HierarchicalPcpConfig {
  assertAttributeIds(attributeIds);
  assertExactKeys(methodWeightsByAttribute, attributeIds, "Tuning attribute rows");
  if (attributeWeights) {
    assertExactKeys(attributeWeights, attributeIds, "Tuning attribute weights");
  }

  const config: HierarchicalPcpConfig = {
    attributeWeights: Object.fromEntries(
      attributeIds.map((attributeId) => [
        attributeId,
        attributeWeights
          ? assertWeight(
              attributeWeights[attributeId],
              `${attributeId} tuning attribute weight`,
            )
          : 1,
      ]),
    ),
    methodWeightsByAttribute: Object.fromEntries(
      attributeIds.map((attributeId) => {
        const source = methodWeightsByAttribute[attributeId];
        assertExactKeys(source, HIERARCHICAL_PCP_METHODS, `${attributeId} tuning method weights`);
        return [
          attributeId,
          Object.fromEntries(
            HIERARCHICAL_PCP_METHODS.map((method) => [
              method,
              assertWeight(source[method], `${attributeId} / ${method} tuning weight`),
            ]),
          ) as HierarchicalPcpMethodWeights,
        ];
      }),
    ),
  };
  return normalizeHierarchicalPcpConfig(config, attributeIds);
}

export function cloneHierarchicalPcpConfig(
  config: Readonly<HierarchicalPcpConfig>,
  attributeIds: readonly string[] = Object.keys(config.attributeWeights),
): HierarchicalPcpConfig {
  assertAttributeIds(attributeIds);
  return {
    attributeWeights: Object.fromEntries(
      attributeIds.map((attributeId) => [
        attributeId,
        config.attributeWeights[attributeId],
      ]),
    ),
    methodWeightsByAttribute: Object.fromEntries(
      attributeIds.map((attributeId) => [
        attributeId,
        Object.fromEntries(
          HIERARCHICAL_PCP_METHODS.map((method) => [
            method,
            config.methodWeightsByAttribute[attributeId]?.[method],
          ]),
        ) as HierarchicalPcpMethodWeights,
      ]),
    ),
  };
}

export function normalizeHierarchicalPcpConfig(
  config: Readonly<HierarchicalPcpConfig>,
  attributeIds: readonly string[] = Object.keys(config.attributeWeights),
): HierarchicalPcpConfig {
  assertAttributeIds(attributeIds);
  assertExactKeys(config.attributeWeights, attributeIds, "Attribute weights");
  assertExactKeys(config.methodWeightsByAttribute, attributeIds, "Method-weight rows");

  const attributeValues = attributeIds.map((attributeId) => (
    assertWeight(config.attributeWeights[attributeId], `${attributeId} attribute weight`)
  ));
  const attributeTotal = attributeValues.reduce((sum, value) => sum + value, 0);
  if (attributeTotal <= 0) {
    throw new Error("At least one attribute weight must be greater than zero.");
  }

  const methodWeightsByAttribute = Object.fromEntries(
    attributeIds.map((attributeId) => {
      const source = config.methodWeightsByAttribute[attributeId];
      assertExactKeys(source, HIERARCHICAL_PCP_METHODS, `${attributeId} method weights`);
      const values = HIERARCHICAL_PCP_METHODS.map((method) => (
        assertWeight(source[method], `${attributeId} / ${method} weight`)
      ));
      const total = values.reduce((sum, value) => sum + value, 0);
      if (total <= 0) {
        throw new Error(`At least one method weight for ${attributeId} must be greater than zero.`);
      }
      return [
        attributeId,
        Object.fromEntries(
          HIERARCHICAL_PCP_METHODS.map((method, index) => [method, values[index] / total]),
        ) as HierarchicalPcpMethodWeights,
      ];
    }),
  );

  return {
    attributeWeights: Object.fromEntries(
      attributeIds.map((attributeId, index) => [
        attributeId,
        attributeValues[index] / attributeTotal,
      ]),
    ),
    methodWeightsByAttribute,
  };
}

export function hierarchicalPcpConfigFingerprint(
  config: Readonly<HierarchicalPcpConfig>,
  attributeIds: readonly string[] = Object.keys(config.attributeWeights),
) {
  const normalized = normalizeHierarchicalPcpConfig(config, attributeIds);
  return JSON.stringify({
    attributes: attributeIds.map((attributeId) => [
      attributeId,
      normalizedFingerprintValue(normalized.attributeWeights[attributeId]),
      HIERARCHICAL_PCP_METHODS.map((method) => (
        normalizedFingerprintValue(normalized.methodWeightsByAttribute[attributeId][method])
      )),
    ]),
  });
}

export function areHierarchicalPcpConfigsEquivalent(
  left: Readonly<HierarchicalPcpConfig>,
  right: Readonly<HierarchicalPcpConfig>,
  attributeIds: readonly string[] = Object.keys(left.attributeWeights),
) {
  try {
    const normalizedLeft = normalizeHierarchicalPcpConfig(left, attributeIds);
    const normalizedRight = normalizeHierarchicalPcpConfig(right, attributeIds);
    return attributeIds.every((attributeId) => (
      Math.abs(
        normalizedLeft.attributeWeights[attributeId]
        - normalizedRight.attributeWeights[attributeId],
      ) <= WEIGHT_EPSILON
      && HIERARCHICAL_PCP_METHODS.every((method) => (
        Math.abs(
          normalizedLeft.methodWeightsByAttribute[attributeId][method]
          - normalizedRight.methodWeightsByAttribute[attributeId][method],
        ) <= WEIGHT_EPSILON
      ))
    ));
  } catch {
    return false;
  }
}

export function isDefaultHierarchicalPcpConfig(
  config: Readonly<HierarchicalPcpConfig>,
  attributeIds: readonly string[] = Object.keys(config.attributeWeights),
) {
  try {
    return areHierarchicalPcpConfigsEquivalent(
      config,
      createDefaultHierarchicalPcpConfig(attributeIds),
      attributeIds,
    );
  } catch {
    return false;
  }
}

export function updateHierarchicalAttributeWeight(
  config: Readonly<HierarchicalPcpConfig>,
  attributeId: string,
  weight: number,
): HierarchicalPcpConfig {
  if (!owns(config.attributeWeights, attributeId)) {
    throw new Error(`Unknown hierarchical PCP attribute: ${attributeId}`);
  }
  assertWeight(weight, `${attributeId} attribute weight`);
  return {
    attributeWeights: { ...config.attributeWeights, [attributeId]: weight },
    methodWeightsByAttribute: config.methodWeightsByAttribute,
  };
}

export function updateHierarchicalMethodWeight(
  config: Readonly<HierarchicalPcpConfig>,
  attributeId: string,
  methodId: HierarchicalPcpMethod,
  weight: number,
): HierarchicalPcpConfig {
  const current = config.methodWeightsByAttribute[attributeId];
  if (!current) throw new Error(`Unknown hierarchical PCP attribute: ${attributeId}`);
  if (!(HIERARCHICAL_PCP_METHODS as readonly string[]).includes(methodId)) {
    throw new Error(`Unknown hierarchical PCP method: ${methodId}`);
  }
  assertWeight(weight, `${attributeId} / ${methodId} weight`);
  return {
    attributeWeights: config.attributeWeights,
    methodWeightsByAttribute: {
      ...config.methodWeightsByAttribute,
      [attributeId]: { ...current, [methodId]: weight },
    },
  };
}

export function coerceHierarchicalPcpWeight(value: string | number, maximum = 100) {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(maximum, Math.max(0, parsed));
}

export function pruneHierarchicalPcpBrushes<T>(
  brushes: Readonly<Record<string, T | undefined>>,
  visibleAxes: readonly Pick<HierarchicalPcpAxis, "id">[],
): Record<string, T> {
  const visible = new Set(visibleAxes.map((axis) => axis.id));
  return Object.fromEntries(
    Object.entries(brushes).filter(
      (entry): entry is [string, T] => visible.has(entry[0]) && entry[1] !== undefined,
    ),
  );
}
