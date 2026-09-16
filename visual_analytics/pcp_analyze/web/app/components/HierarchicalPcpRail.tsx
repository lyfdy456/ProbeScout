"use client";

import {
  HIERARCHICAL_PCP_BOTTOM_PADDING,
  HIERARCHICAL_PCP_TOP,
  coerceHierarchicalPcpWeight,
  type HierarchicalPcpAxis,
  type HierarchicalPcpConfig,
  type HierarchicalPcpMethod,
} from "../lib/hierarchicalPcp";

export type HierarchicalPcpRailVariant = "aligned" | "summary";

export interface RefinementPcpWeights {
  beta: Readonly<Record<string, Readonly<Record<string, number>>>>;
  gamma: Readonly<Record<string, number>>;
  embeddingWeights: Readonly<Record<string, number>>;
  embeddingFusionStrength: number;
}

export interface HierarchicalPcpRailProps {
  axes: readonly HierarchicalPcpAxis[];
  config: Readonly<HierarchicalPcpConfig>;
  fullExpanded: boolean;
  attributeEvidenceExpanded?: boolean;
  holisticExpanded?: boolean;
  expandedAttributeIds: ReadonlySet<string> | readonly string[];
  height: number;
  /**
   * `aligned` overlays the sample PCP and shares its row geometry. `summary`
   * is a compact, in-flow control panel for cluster centroids, whose own axis
   * labels remain visible in the chart.
   */
  variant?: HierarchicalPcpRailVariant;
  disabled?: boolean;
  weightsReadOnly?: boolean;
  refinementWeights?: RefinementPcpWeights | null;
  className?: string;
  onFullExpandedChange: (expanded: boolean) => void;
  onAttributeEvidenceExpandedChange?: (expanded: boolean) => void;
  onHolisticExpandedChange?: (expanded: boolean) => void;
  onAttributeExpandedChange: (attributeId: string, expanded: boolean) => void;
  onAttributeWeightChange: (attributeId: string, weight: number) => void;
  onMethodWeightChange: (
    attributeId: string,
    methodId: HierarchicalPcpMethod,
    weight: number,
  ) => void;
}

const ROW_HEIGHT = 32;

function weightTitle(axis: HierarchicalPcpAxis, refinement: boolean) {
  if (axis.kind === "attribute") {
    return refinement
      ? `${axis.label}: independent conjunction strength gamma`
      : `${axis.label}: outer attribute contribution weight in Weighted Fusion`;
  }
  if (axis.kind === "holistic") {
    return "Embedding evidence: late-fusion strength λ in F = C × [(1 − λ) + λH]";
  }
  if (axis.kind === "baseline") {
    return refinement
      ? `${axis.label}: global holistic weight eta`
      : `${axis.label}: inner rank weight for ${axis.attributeId}; uses the canonical Joint ranking`;
  }
  return refinement
    ? `${axis.label}: within-attribute learner weight beta for ${axis.attributeId}`
    : `${axis.label}: inner rank weight for ${axis.attributeId}`;
}

export function HierarchicalPcpRail({
  axes,
  config,
  fullExpanded,
  attributeEvidenceExpanded = true,
  holisticExpanded = false,
  expandedAttributeIds,
  height,
  variant = "aligned",
  disabled = false,
  weightsReadOnly = false,
  refinementWeights = null,
  className = "",
  onFullExpandedChange,
  onAttributeEvidenceExpandedChange,
  onHolisticExpandedChange,
  onAttributeExpandedChange,
  onAttributeWeightChange,
  onMethodWeightChange,
}: HierarchicalPcpRailProps) {
  const expanded = new Set(expandedAttributeIds);
  const bottom = Math.max(HIERARCHICAL_PCP_TOP + 20, height - HIERARCHICAL_PCP_BOTTOM_PADDING);
  const rowGap = axes.length > 1
    ? (bottom - HIERARCHICAL_PCP_TOP) / (axes.length - 1)
    : 0;
  const summaryVariant = variant === "summary";
  const rootClassName = [
    "hierarchical-pcp-rail",
    summaryVariant ? "hierarchical-pcp-rail--summary" : "",
    className,
  ].filter(Boolean).join(" ");

  return (
    <div
      className={rootClassName}
      role="group"
      aria-label={summaryVariant
        ? "Cluster summary hierarchy controls"
        : refinementWeights
          ? "Tuned conjunction hierarchy, learners, and holistic retrieval heads"
          : "Weighted Fusion, attributes, and thirteen rank methods per attribute"}
      data-variant={variant}
      data-refinement={Boolean(refinementWeights)}
      style={summaryVariant ? undefined : { height }}
    >
      {axes.map((axis, index) => {
        const centerY = axes.length > 1
          ? HIERARCHICAL_PCP_TOP + index * rowGap
          : (HIERARCHICAL_PCP_TOP + bottom) / 2;
        const attributeId = axis.attributeId;
        const methodId = axis.methodId;
        const expandedAxis = axis.kind === "full"
          ? fullExpanded
          : axis.kind === "attribute-evidence"
            ? attributeEvidenceExpanded
          : axis.kind === "holistic"
            ? holisticExpanded
          : axis.kind === "attribute" && attributeId
            ? expanded.has(attributeId)
            : undefined;
        const weight = refinementWeights
          ? axis.kind === "attribute" && attributeId
            ? refinementWeights.gamma[attributeId]
            : axis.kind === "learner" && attributeId && methodId
              ? refinementWeights.beta[attributeId]?.[methodId]
              : axis.kind === "holistic"
                ? refinementWeights.embeddingFusionStrength
                : axis.kind === "baseline" && methodId
                  ? refinementWeights.embeddingWeights[methodId]
                  : undefined
          : axis.kind === "attribute" && attributeId
            ? config.attributeWeights[attributeId]
            : (axis.kind === "baseline" || axis.kind === "learner")
                && attributeId
                && methodId
              ? config.methodWeightsByAttribute[attributeId]?.[methodId]
              : undefined;
        const rowClassName = [
          "hierarchical-pcp-row",
          `hierarchical-pcp-row--${axis.kind}`,
          `hierarchical-pcp-row--depth-${axis.depth}`,
        ].join(" ");
        const nodeRole = axis.kind === "full" ? "root"
          : axis.kind === "attribute-evidence" || axis.kind === "holistic" ? "evidence"
          : axis.kind === "attribute" ? "attribute"
          : axis.kind === "baseline" ? "embedding"
          : "probe";

        return (
          <div
            className={rowClassName}
            data-axis-id={axis.id}
            data-depth={axis.depth}
            data-node-role={nodeRole}
            key={axis.id}
            style={summaryVariant
              ? undefined
              : { top: Math.max(0, centerY - ROW_HEIGHT / 2) }}
          >
            {axis.kind === "full" || axis.kind === "attribute-evidence" || axis.kind === "attribute" || axis.kind === "holistic" ? (
              <button
                type="button"
                className="hierarchical-pcp-disclosure"
                aria-expanded={Boolean(expandedAxis)}
                aria-label={`${expandedAxis ? "Collapse" : "Expand"} ${axis.label}`}
                disabled={disabled}
                onClick={() => {
                  if (axis.kind === "full") {
                    onFullExpandedChange(!fullExpanded);
                  } else if (axis.kind === "attribute-evidence") {
                    onAttributeEvidenceExpandedChange?.(!attributeEvidenceExpanded);
                  } else if (axis.kind === "holistic") {
                    onHolisticExpandedChange?.(!holisticExpanded);
                  } else if (attributeId) {
                    onAttributeExpandedChange(attributeId, !expanded.has(attributeId));
                  }
                }}
              >
                <span aria-hidden="true">{expandedAxis ? "▾" : "▸"}</span>
              </button>
            ) : (
              <span className="hierarchical-pcp-branch" aria-hidden="true">─</span>
            )}

            <span className="hierarchical-pcp-label" title={axis.description ?? axis.label}>
              {axis.label}
            </span>

            {axis.kind === "full" ? (
              <span className="hierarchical-pcp-overall">{refinementWeights ? "F" : "overall"}</span>
            ) : axis.kind === "attribute-evidence" ? (
              <span className="hierarchical-pcp-evidence-badge" title={axis.description}>C</span>
            ) : refinementWeights ? (
              <output
                className="hierarchical-pcp-weight-readout"
                aria-label={weightTitle(axis, true)}
                title={weightTitle(axis, true)}
              >
                <span className="hierarchical-pcp-weight-symbol" aria-hidden="true">
                  {axis.kind === "attribute" ? "γ" : axis.kind === "holistic" ? "λ" : axis.kind === "learner" ? "β" : "η"}
                </span>
                {Number.isFinite(weight)
                  ? axis.kind === "attribute" || axis.kind === "holistic"
                    ? weight!.toFixed(3)
                    : `${(weight! * 100).toFixed(1)}%`
                  : "—"}
              </output>
            ) : (
              <input
                className="hierarchical-pcp-weight"
                type="number"
                min={0}
                max={100}
                step={0.05}
                value={Number.isFinite(weight) ? weight : 0}
                aria-label={weightTitle(axis, Boolean(refinementWeights))}
                title={weightTitle(axis, Boolean(refinementWeights))}
                disabled={disabled || weightsReadOnly}
                onChange={(event) => {
                  const nextWeight = coerceHierarchicalPcpWeight(event.target.value);
                  if (axis.kind === "attribute" && attributeId) {
                    onAttributeWeightChange(attributeId, nextWeight);
                  } else if (attributeId && methodId) {
                    onMethodWeightChange(attributeId, methodId, nextWeight);
                  }
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

export default HierarchicalPcpRail;
