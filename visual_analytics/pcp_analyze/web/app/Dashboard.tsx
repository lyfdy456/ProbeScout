"use client";

import {
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { hcl } from "d3";
import {
  ParallelCoordinates,
  type PcpBrushMap,
  type PcpColorMode,
} from "./components/ParallelCoordinates";
import {
  ClusterSummaryParallelCoordinates,
  type ClusterSummaryId,
} from "./components/ClusterSummaryParallelCoordinates";
import {
  ProjectionScatter,
  type ProjectionKind,
  type ProjectionPoint,
  type ProjectionSelection,
} from "./components/ProjectionScatter";
import {
  TopGallery,
  type GalleryItem,
  type PreferenceState,
  type ThumbnailAtlasDescriptor,
} from "./components/TopGallery";
import { GalleryLightbox } from "./components/GalleryLightbox";
import { AnnotationReviewGallery } from "./components/AnnotationReviewGallery";
import {
  HierarchicalPcpRail,
  type RefinementPcpWeights,
} from "./components/HierarchicalPcpRail";
import type { AttributeStrengthPoint } from "./components/AttributeStrengthProfile";
import { SelectionOverlapPanel } from "./components/SelectionOverlapPanel";
import { SelectionRuleSummaryPanel } from "./components/SelectionRuleSummaryPanel";
import { SmartFilterPanel } from "./components/SmartFilterPanel";
import { TaskLoadingEstimate } from "./components/TaskLoadingEstimate";
import { TuningFunctionActions } from "./components/TuningFunctionActions";
import { TuningPanel } from "./components/TuningPanel";
import { FrozenTestEvaluation } from "./components/FrozenTestEvaluation";
import { ProbeSourceControls } from "./components/ProbeSourceControls";
import { UserSessionControls } from "./components/UserSessionControls";
import { usePcpPanelAlignment } from "./lib/usePcpPanelAlignment";
import { methodDisplayLabel } from "./lib/methodDisplay.js";
import {
  ValidationComparison,
  type ValidationComparisonRow,
} from "./components/ValidationComparison";
import {
  initialDownloadBytesFromCatalog,
  initialDownloadBytesFromManifest,
  resolveInitialDownloadBytes,
  type InitialDownloadResolution,
} from "./lib/taskLoadingEstimate";
import {
  analysisScopeMask,
  applyResultScope,
  filterAnalysisRows,
  isGroundTruthPositive,
  type ScopeMasks,
  type ResultScope,
} from "./lib/resultScope";
import {
  isStaticValidationRankingCurrent,
  manualHighlightScope,
  scopeActivitySummary,
} from "./lib/dashboardState";
import { evaluateRankedScope } from "./lib/validationMetrics";
import {
  buildSmartFilterMask,
  SMART_FILTER_LEARNER_METHODS,
  type SmartFilterKind,
} from "./smartFilter";
import { useTuningSession } from "./lib/useTuningSession";
import { useInitialBaseline } from "./lib/useInitialBaseline";
import { initialBaselineTargetColumn, LEGACY_OURS_FULL_OPTION } from "./lib/initialBaseline";
import { buildSnapshotDiagnosticValues } from "./lib/snapshotDiagnostics";
import { useOriginalVqaSupervision } from "./lib/useOriginalVqaSupervision";
import { useFixedVqaValidation } from "./lib/useFixedVqaValidation";
import { developmentWithoutFixedVal, fixedValAllowsFeedback } from "./lib/fixedVqaValidation";
import { summarizeSelectionRules } from "./lib/selectionRuleSummary";
import {
  effectivePcpSelectionRanges,
  summarizeSelectionOverlap,
  type SelectionOverlapConditionInput,
  type SelectionOverlapStageInput,
} from "./lib/selectionOverlap";
import {
  applyPcpBrushZoomMask,
  createPcpBrushZoomState,
  drillIntoPcpBrushZoom,
  hasPcpBrushZoom,
  pcpBrushZoomDepth,
  resetPcpBrushZoom,
  stepBackPcpBrushZoom,
} from "./lib/pcpBrushZoom";
import {
  isHierarchicalRankFusionRun,
  isWeightRefinementRun,
  REFINEMENT_EMBEDDING_METHODS,
  refinementEmbeddingMethods,
  MAX_BULK_TUNING_ANNOTATIONS,
  tuningRankingLabel,
} from "./lib/tuningApi";
import {
  buildHierarchicalPcpAxes,
  buildRefinementPcpAxes,
  createDefaultHierarchicalPcpConfig,
  HIERARCHICAL_PCP_LEARNERS,
  hierarchicalAttributeAxisId,
  hierarchicalPcpConfigFromTuningWeights,
  hierarchicalPcpConfigFingerprint,
  updateHierarchicalAttributeWeight,
  updateHierarchicalMethodWeight,
  refinementPcpComponentIds,
  type HierarchicalPcpConfig,
} from "./lib/hierarchicalPcp";
import {
  HIERARCHICAL_FUSION_METHOD_ID,
  HIERARCHICAL_FUSION_METHOD_LABEL,
  useHierarchicalFusion,
} from "./lib/hierarchicalFusion";
import {
  requestRuntimeRankProfileClusters,
  useRuntimeRankProfileClusters,
  type RuntimeRankProfileClusterScheme,
} from "./lib/runtimeRankProfileClusters";
import {
  requestRefinementClusters,
  useRefinementClusters,
} from "./lib/refinementVisualization";
import {
  useVisualEmbeddingAnalysis,
  type VisualClusterCount,
  type VisualClusterScheme,
  type VisualEmbeddingManifest,
  type VisualEmbeddingSource,
} from "./lib/visualEmbeddingAnalysis";

type ClusterScheme = "fine30" | "fine50" | "fine100" | "absolute" | "shape";
type ClusterFamily = "fine" | "structural";
type RetrievalTarget = string;
type PcpValueKind = "rank" | "calibrated";
type PcpLineMode = "samples" | "clusters";
type TopLimit = 30 | 50 | 100 | 200;

interface ClusterDrilldown {
  scheme: ClusterScheme;
  scope: ResultScope;
  clusterId: string;
}

interface AppliedFusionTune {
  taskId: string;
  runId: string;
  sessionId: string;
  sourceTargetId: string;
  baseMethod: string;
  label: string;
  configFingerprint: string;
}

interface EvaluationContract {
  defaultResultScope: ResultScope;
  scopes: Array<{
    id: ResultScope;
    label: string;
    rowCount: number;
  }>;
  development: {
    maskFileKey: "developmentMask";
    queryExcluded: boolean;
    rowCount: number;
    positiveCounts: Record<RetrievalTarget, number>;
    groundTruthUsage: string;
  };
  validation: {
    maskFileKey: "validationMask";
    splitSeed: number;
    fractionOfEligibleRemainder: number;
    queryExcluded: boolean;
    stratifiedBy: string[];
    rowCount: number;
    positiveCounts: Record<RetrievalTarget, number>;
    groundTruthUsage: string;
  };
  frozenTest: {
    maskFileKey: "testMask";
    splitSeed: number;
    fraction: number;
    queryExcluded: boolean;
    stratifiedBy: string[];
    rowCount: number;
    positiveCounts: Record<RetrievalTarget, number>;
    groundTruthUsage: string;
  };
}

interface RetrievalTargetDefinition {
  id: RetrievalTarget;
  label: string;
  kind: "attribute" | "derived";
  members?: string[];
  rule?: string;
}

interface FileSpec {
  path: string;
  shape: number[];
  columns?: string[];
  bytes?: number;
  dtype?: string;
  byteOrder?: string;
  layout?: string;
  sha256?: string;
}

interface ClusterSchemeDefinition {
  id: ClusterScheme;
  label: string;
  family: ClusterFamily;
  clusters: number;
  labelsFileKey: string;
}

interface FixedQueryImage {
  path: string;
  imageId: string;
  imageIndex: number;
  mimeType: string;
  bytes: number;
  sha256: string;
}

interface FixedQueryDefinition {
  mode: "fixed";
  text: string;
  images: FixedQueryImage[];
}

interface TaskCatalogEntry {
  id: string;
  label: string;
  dataRoot: string;
  description?: string;
  defaultRetrievalTarget?: RetrievalTarget;
  initialDownloadBytes?: number;
  bundleBytes?: number;
  thumbnailBytes?: number;
}

interface DatasetCatalogEntry {
  id: string;
  label: string;
  tasks: TaskCatalogEntry[];
}

interface DatasetCatalog {
  schemaVersion: number;
  defaultDataset: string;
  defaultTask: string;
  datasets: DatasetCatalogEntry[];
}

interface DashboardManifest {
  schemaVersion: number;
  rowCount: number;
  methodCount: number;
  targetCount: number;
  clusterMethodCount: number;
  methods: string[];
  retrievalTargets: RetrievalTargetDefinition[];
  defaultRetrievalTarget: RetrievalTarget;
  clusterMethods: string[];
  defaultRankMethod: string;
  files: {
    [key: string]: FileSpec | undefined;
    imageIds: FileSpec;
    metadata: FileSpec;
    rawScores: FileSpec;
    calibratedScores: FileSpec;
    ranks: FileSpec;
    fine30Labels: FileSpec;
    fine50Labels: FileSpec;
    fine100Labels: FileSpec;
    absoluteLabels: FileSpec;
    shapeLabels: FileSpec;
    pca2d: FileSpec;
    umap2d?: FileSpec;
    metrics: FileSpec;
    groundTruth: FileSpec;
    developmentMask: FileSpec;
    validationMask: FileSpec;
    testMask: FileSpec;
    drawOrder?: FileSpec;
  };
  projections: {
    basisTarget: RetrievalTarget;
    pca: {
      available: boolean;
      explainedVarianceRatio?: number[];
      fitScope?: string;
      fitMaskFileKey?: string | null;
    };
    umap: {
      available: boolean;
      reason?: string;
      fitScope?: string;
      fitMaskFileKey?: string | null;
    };
  };
  clusters: {
    basisTarget: RetrievalTarget;
    defaultScheme: ClusterScheme;
    note?: string;
    schemes: ClusterSchemeDefinition[];
  };
  visualEmbedding: VisualEmbeddingManifest;
  thumbnails: ThumbnailMetadata;
  query?: FixedQueryDefinition;
  evaluation: EvaluationContract;
}

interface ThumbnailMetadata {
  available: boolean;
  directory: string;
  columns: number;
  rows: number;
  itemsPerAtlas: number;
  atlasCount: number;
  tileWidth: number;
  tileHeight: number;
}

interface ClusterSummary {
  scheme: ClusterScheme;
  cluster_id: number;
  label: string;
  label_zh: string;
  size: number;
  fraction: number;
  mean_rank: number;
  learned_minus_fixed: number;
}

interface DashboardMetadata {
  task: {
    dataset: string;
    task: string;
    taskName: string;
    split: string;
    defaultRetrievalTarget: RetrievalTarget;
    baseAttributes: string[];
    retrievalTargets: RetrievalTargetDefinition[];
    rawScoreDefinition?: string;
    calibratedScoreDefinition?: string;
    rankDefinition?: string;
  };
  clusterSummary: ClusterSummary[];
  thumbnails: ThumbnailMetadata;
  evaluation: EvaluationContract;
}

interface DashboardDataset {
  dataRoot: string;
  manifest: DashboardManifest;
  metadata: DashboardMetadata;
  imageIds: string[];
  rawScores: Float32Array;
  calibratedScores: Float32Array;
  ranks: Float32Array;
  clusterLabels: ReadonlyMap<ClusterScheme, Uint8Array>;
  pca2d: Float32Array;
  umap2d: Float32Array | null;
  metrics: Float32Array;
  groundTruth: Uint8Array;
  developmentMask: Uint8Array;
  validationMask: Uint8Array;
  testMask: Uint8Array;
}

const CLUSTER_COLORS = ["#ee6a4b", "#4169c1", "#2d9a74", "#8b62b5"];
const CLUSTER_SCHEME_LABELS: Record<ClusterScheme, string> = {
  fine30: "Fine-grained",
  fine50: "Fine-grained",
  fine100: "Fine-grained",
  absolute: "Absolute",
  shape: "Shape",
};
const CATALOG_PATH = "/data/catalog.json";
const FALLBACK_DATA_ROOT = "/data";
const FALLBACK_CATALOG: DatasetCatalog = {
  schemaVersion: 1,
  defaultDataset: "hico",
  defaultTask: "032_hico_task_hico_hugging_cat",
  datasets: [{
    id: "hico",
    label: "HICO-DET",
    tasks: [{
      id: "032_hico_task_hico_hugging_cat",
      label: "Hugging a cat",
      dataRoot: FALLBACK_DATA_ROOT,
      defaultRetrievalTarget: "joint",
    }],
  }],
};

function scoreOffset(
  rowIndex: number,
  methodIndex: number,
  targetIndex: number,
  methodCount: number,
  targetCount: number,
) {
  return ((rowIndex * methodCount + methodIndex) * targetCount) + targetIndex;
}

function createEmptyHierarchicalPcpConfig(): HierarchicalPcpConfig {
  return {
    attributeWeights: {},
    methodWeightsByAttribute: {},
  };
}

function normalizeDataRoot(dataRoot: string) {
  const rooted = dataRoot.startsWith("/") ? dataRoot : `/${dataRoot}`;
  return rooted.replace(/\/$/, "");
}

function dataPath(dataRoot: string, path: string) {
  return `${normalizeDataRoot(dataRoot)}/${path}`;
}

async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status})`);
  return (await response.json()) as T;
}

async function fetchBuffer(path: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status})`);
  return response.arrayBuffer();
}

async function loadDashboardDataset(
  dataRoot: string,
  signal?: AbortSignal,
  onManifest?: (manifest: DashboardManifest) => void,
): Promise<DashboardDataset> {
  const root = normalizeDataRoot(dataRoot);
  const manifest = await fetchJson<DashboardManifest>(dataPath(root, "manifest.json"), signal);
  onManifest?.(manifest);
  const files = manifest.files;
  const clusterSchemes = manifest.clusters.schemes;
  if (!Array.isArray(clusterSchemes) || clusterSchemes.length === 0) {
    throw new Error("The task manifest has no cluster schemes.");
  }
  const supportedClusterSchemes = new Set<ClusterScheme>([
    "fine30",
    "fine50",
    "fine100",
    "absolute",
    "shape",
  ]);
  if (
    new Set(clusterSchemes.map((scheme) => scheme.id)).size !== clusterSchemes.length
    || clusterSchemes.some((scheme) => (
      !supportedClusterSchemes.has(scheme.id)
      || !Number.isInteger(scheme.clusters)
      || scheme.clusters < 2
      || scheme.clusters > 255
    ))
    || !clusterSchemes.some((scheme) => scheme.id === manifest.clusters.defaultScheme)
  ) {
    throw new Error("The task manifest has an invalid cluster scheme contract.");
  }
  const clusterSpecs = clusterSchemes.map((scheme) => {
    const spec = (files as Record<string, FileSpec | undefined>)[scheme.labelsFileKey];
    if (!spec) throw new Error(`Cluster scheme ${scheme.id} has no label file.`);
    return { scheme, spec };
  });
  const [
    metadata,
    imageIds,
    rawScores,
    calibratedScores,
    ranks,
    clusterBuffers,
    pca,
    metrics,
    groundTruth,
    developmentMask,
    validationMask,
    testMask,
    umap,
  ] =
    await Promise.all([
      fetchJson<DashboardMetadata>(dataPath(root, files.metadata.path), signal),
      fetchJson<string[]>(dataPath(root, files.imageIds.path), signal),
      fetchBuffer(dataPath(root, files.rawScores.path), signal),
      fetchBuffer(dataPath(root, files.calibratedScores.path), signal),
      fetchBuffer(dataPath(root, files.ranks.path), signal),
      Promise.all(
        clusterSpecs.map(({ spec }) => fetchBuffer(dataPath(root, spec.path), signal)),
      ),
      fetchBuffer(dataPath(root, files.pca2d.path), signal),
      fetchBuffer(dataPath(root, files.metrics.path), signal),
      fetchBuffer(dataPath(root, files.groundTruth.path), signal),
      fetchBuffer(dataPath(root, files.developmentMask.path), signal),
      fetchBuffer(dataPath(root, files.validationMask.path), signal),
      fetchBuffer(dataPath(root, files.testMask.path), signal),
      files.umap2d && manifest.projections.umap.available
        ? fetchBuffer(dataPath(root, files.umap2d.path), signal)
        : Promise.resolve(null),
    ]);

  const dataset: DashboardDataset = {
    dataRoot: root,
    manifest,
    metadata,
    imageIds,
    rawScores: new Float32Array(rawScores),
    calibratedScores: new Float32Array(calibratedScores),
    ranks: new Float32Array(ranks),
    clusterLabels: new Map(
      clusterSpecs.map(({ scheme }, index) => [
        scheme.id,
        new Uint8Array(clusterBuffers[index]),
      ]),
    ),
    pca2d: new Float32Array(pca),
    umap2d: umap ? new Float32Array(umap) : null,
    metrics: new Float32Array(metrics),
    groundTruth: new Uint8Array(groundTruth),
    developmentMask: new Uint8Array(developmentMask),
    validationMask: new Uint8Array(validationMask),
    testMask: new Uint8Array(testMask),
  };

  if (
    dataset.imageIds.length !== manifest.rowCount ||
    manifest.retrievalTargets.length !== manifest.targetCount ||
    dataset.rawScores.length !== manifest.rowCount * manifest.methodCount * manifest.targetCount ||
    dataset.calibratedScores.length !== manifest.rowCount * manifest.methodCount * manifest.targetCount ||
    dataset.ranks.length !== manifest.rowCount * manifest.methodCount * manifest.targetCount ||
    dataset.clusterLabels.size !== clusterSchemes.length ||
    [...dataset.clusterLabels.values()].some((clusterLabels) => (
      clusterLabels.length !== manifest.rowCount
    )) ||
    dataset.pca2d.length !== manifest.rowCount * 2 ||
    dataset.metrics.length !== manifest.files.metrics.shape.reduce((size, value) => size * value, 1) ||
    dataset.groundTruth.length !== manifest.rowCount * manifest.targetCount ||
    dataset.developmentMask.length !== manifest.rowCount ||
    dataset.validationMask.length !== manifest.rowCount ||
    dataset.testMask.length !== manifest.rowCount
  ) {
    throw new Error("The exported browser data does not match its manifest.");
  }
  for (const scheme of clusterSchemes) {
    const schemeLabels = dataset.clusterLabels.get(scheme.id);
    const observed = new Uint8Array(scheme.clusters);
    if (!schemeLabels) throw new Error(`Cluster labels are missing for ${scheme.id}.`);
    for (const clusterId of schemeLabels) {
      if (clusterId >= scheme.clusters) {
        throw new Error(`Cluster labels exceed K${scheme.clusters} for ${scheme.id}.`);
      }
      observed[clusterId] = 1;
    }
    if (observed.some((value) => value === 0)) {
      throw new Error(`Cluster labels are incomplete for ${scheme.id}.`);
    }
  }
  const scopeRows = { development: 0, validation: 0, test: 0 };
  const queryIndices = new Set(manifest.query?.images.map((image) => image.imageIndex) ?? []);
  for (let rowIndex = 0; rowIndex < manifest.rowCount; rowIndex += 1) {
    const values = [
      dataset.developmentMask[rowIndex],
      dataset.validationMask[rowIndex],
      dataset.testMask[rowIndex],
    ];
    if (values.some((value) => value !== 0 && value !== 1)) {
      throw new Error("Development, Validation and Frozen Test masks must be binary.");
    }
    const memberships = values[0] + values[1] + values[2];
    if (memberships > 1 || (queryIndices.has(rowIndex) ? memberships !== 0 : memberships !== 1)) {
      throw new Error("The three data splits overlap or do not cover the non-Query gallery.");
    }
    scopeRows.development += values[0];
    scopeRows.validation += values[1];
    scopeRows.test += values[2];
  }
  if (
    dataset.manifest.evaluation.development.maskFileKey !== "developmentMask"
    || dataset.manifest.evaluation.validation.maskFileKey !== "validationMask"
    || dataset.manifest.evaluation.frozenTest.maskFileKey !== "testMask"
    || scopeRows.development !== dataset.manifest.evaluation.development.rowCount
    || scopeRows.validation !== dataset.manifest.evaluation.validation.rowCount
    || scopeRows.test !== dataset.manifest.evaluation.frozenTest.rowCount
  ) {
    throw new Error("The data split masks do not match their evaluation contract.");
  }
  if (manifest.query) {
    if (manifest.query.mode !== "fixed" || !Array.isArray(manifest.query.images)) {
      throw new Error("The task manifest has an invalid fixed query contract.");
    }
    const invalidQueryImage = manifest.query.images.find((image) => {
      const pathParts = image.path.replaceAll("\\", "/").split("/");
      return image.path.startsWith("/")
        || pathParts.includes("..")
        || !Number.isInteger(image.imageIndex)
        || image.imageIndex < 0
        || image.imageIndex >= dataset.imageIds.length
        || dataset.imageIds[image.imageIndex] !== image.imageId;
    });
    if (invalidQueryImage) {
      throw new Error("The fixed query images do not match the task gallery.");
    }
  }
  return dataset;
}

function countMask(mask: Uint8Array) {
  let count = 0;
  for (const value of mask) count += value ? 1 : 0;
  return count;
}

function intersectMasks(left: Uint8Array, right: Uint8Array | null) {
  if (!right) return left;
  const length = Math.min(left.length, right.length);
  const intersection = new Uint8Array(left.length);
  for (let index = 0; index < length; index += 1) {
    intersection[index] = left[index] && right[index] ? 1 : 0;
  }
  return intersection;
}

function fileLabel(imageId: string) {
  return imageId.split("/").at(-1)?.replace(/\.jpg$/i, "") ?? imageId;
}

function clusterColor(cluster: string | number, scheme: ClusterScheme) {
  const number = typeof cluster === "number" ? cluster : Number(cluster);
  const clusterId = Math.abs(Number.isFinite(number) ? Math.trunc(number) : 0);
  if (!isFineClusterScheme(scheme)) return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
  // The golden-angle step spreads adjacent, mean-rank-ordered cluster IDs
  // around perceptual HCL space instead of recycling the four overview colors.
  const hue = (12 + clusterId * 137.50776405) % 360;
  return hcl(hue, 52, 58).formatHex();
}

function isFineClusterScheme(scheme: ClusterScheme) {
  return scheme.startsWith("fine");
}

function thumbnailFor(
  rowIndex: number,
  thumbnails: ThumbnailMetadata,
  dataRoot: string,
): ThumbnailAtlasDescriptor | null {
  if (!thumbnails.available) return null;
  const atlas = Math.floor(rowIndex / thumbnails.itemsPerAtlas);
  const cell = rowIndex % thumbnails.itemsPerAtlas;
  return {
    atlasUrl: `${normalizeDataRoot(dataRoot)}/${thumbnails.directory}/atlas-${String(atlas).padStart(4, "0")}.webp`,
    row: Math.floor(cell / thumbnails.columns),
    col: cell % thumbnails.columns,
    cols: thumbnails.columns,
    rows: thumbnails.rows,
  };
}

function LoadingScreen({
  error,
  estimate,
  onRetry,
}: {
  error?: string;
  estimate?: InitialDownloadResolution | null;
  onRetry?: () => void;
}) {
  return (
    <main className="loading-screen">
      <div className="loading-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p className="eyebrow">ProbeScout: Interactive Retrieval Refinement</p>
      <h1>{error ? "数据载入失败" : "正在准备任务分析空间"}</h1>
      <p>{error ?? "正在加载模型分数、聚类标签和二维投影。"}</p>
      {!error && (
        <TaskLoadingEstimate
          className="loading-estimate"
          resolution={estimate}
        />
      )}
      {error && onRetry && (
        <button type="button" className="button-primary" onClick={onRetry}>Retry loading</button>
      )}
    </main>
  );
}

export function Dashboard() {
  const [catalog, setCatalog] = useState<DatasetCatalog | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [dataset, setDataset] = useState<DashboardDataset | null>(null);
  const [loadError, setLoadError] = useState<string>();
  const [reloadNonce, setReloadNonce] = useState(0);
  const [colorMode, setColorMode] = useState<PcpColorMode>("cluster");
  const [colorLearner, setColorLearner] = useState("");
  const [rankMethod, setRankMethod] = useState("");
  const [legacyBaselineSelected, setLegacyBaselineSelected] = useState(false);
  const [rankClusterScheme, setRankClusterScheme] = useState<ClusterScheme>("fine50");
  const [rankClusterFilter, setRankClusterFilter] = useState("all");
  const [visualClusterScheme, setVisualClusterScheme] =
    useState<VisualClusterScheme>("fine50");
  const [visualClusterFilter, setVisualClusterFilter] = useState("all");
  const [clusterDrilldown, setClusterDrilldown] = useState<ClusterDrilldown | null>(null);
  const [retrievalTarget, setRetrievalTarget] = useState<RetrievalTarget>("");
  const [pcpFullExpanded, setPcpFullExpanded] = useState(true);
  const [pcpAttributeEvidenceExpanded, setPcpAttributeEvidenceExpanded] = useState(true);
  const [pcpHolisticExpanded, setPcpHolisticExpanded] = useState(false);
  const [expandedPcpAttributes, setExpandedPcpAttributes] = useState<Set<string>>(new Set());
  const [hierarchicalDraft, setHierarchicalDraft] = useState<HierarchicalPcpConfig>(
    createEmptyHierarchicalPcpConfig,
  );
  const [appliedFusionTune, setAppliedFusionTune] = useState<AppliedFusionTune | null>(null);
  const [fusionTuneApplyError, setFusionTuneApplyError] = useState<string | null>(null);
  const [rankClusterApplyError, setRankClusterApplyError] = useState<string | null>(null);
  const [rankClusterRequestBusy, setRankClusterRequestBusy] = useState(false);
  const [pcpValueKind, setPcpValueKind] = useState<PcpValueKind>("rank");
  const [pcpLineMode, setPcpLineMode] = useState<PcpLineMode>("samples");
  const [projection, setProjection] = useState<ProjectionKind>("pca");
  const [brushes, setBrushes] = useState<PcpBrushMap>({});
  const [pcpBrushZoom, setPcpBrushZoom] = useState(createPcpBrushZoomState);
  const [projectionSelection, setProjectionSelection] = useState<ProjectionSelection | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showManualHighlights, setShowManualHighlights] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [queryPreviewIndex, setQueryPreviewIndex] = useState<number | null>(null);
  const [topLimit, setTopLimit] = useState<TopLimit>(30);
  const [resultScope, setResultScope] = useState<ResultScope>("development");
  const [generatedOnly, setGeneratedOnly] = useState(false);
  const [smartFilterKind, setSmartFilterKind] = useState<SmartFilterKind | null>(null);
  const [smartFilterLearner, setSmartFilterLearner] = useState<string>(
    SMART_FILTER_LEARNER_METHODS[0],
  );
  const [taskLoadEstimate, setTaskLoadEstimate] = useState<{
    taskKey: string;
    resolution: InitialDownloadResolution;
  } | null>(null);
  const pcpFrameRef = useRef<HTMLDivElement>(null);
  const explorationContentRef = useRef<HTMLDivElement>(null);
  const fusionTuneApplyGenerationRef = useRef(0);
  const hierarchicalApplyGenerationRef = useRef(0);
  const rankClusterSchemeGenerationRef = useRef(0);
  const rankClusterInteractionSourceRef = useRef<string | null>(null);
  const tuningApplyScopeRef = useRef<{
    taskId: string;
    runId: string | null;
    sessionId: string | null;
  }>({ taskId: "", runId: null, sessionId: null });

  const selectedDatasetEntry = useMemo(
    () => catalog?.datasets.find((entry) => entry.id === selectedDatasetId) ?? null,
    [catalog, selectedDatasetId],
  );
  const selectedTaskEntry = useMemo(
    () => selectedDatasetEntry?.tasks.find((entry) => entry.id === selectedTaskId) ?? null,
    [selectedDatasetEntry, selectedTaskId],
  );
  const catalogLoadEstimate = initialDownloadBytesFromCatalog(
    selectedTaskEntry?.initialDownloadBytes,
  );
  const selectedTaskLoadKey = selectedTaskEntry
    ? `${selectedDatasetId}|${selectedTaskEntry.id}|${normalizeDataRoot(selectedTaskEntry.dataRoot)}`
    : "";
  const currentLoadEstimate = taskLoadEstimate?.taskKey === selectedTaskLoadKey
    ? taskLoadEstimate.resolution
    : catalogLoadEstimate;
  const hierarchicalFusionSelected = rankMethod === HIERARCHICAL_FUSION_METHOD_ID;
  const developmentMode = resultScope === "development";
  const validationMode = resultScope === "validation";
  const testMode = resultScope === "test";
  const fixedValidation = useFixedVqaValidation({
    enabled: Boolean(dataset),
    taskId: selectedTaskId,
    rowCount: dataset?.manifest.rowCount ?? 0,
    reloadKey: reloadNonce,
  });
  const visibleDevelopmentCount = useMemo(() => dataset && fixedValidation.audit
    ? dataset.manifest.evaluation.development.rowCount
      - fixedValidation.audit.rowIndices.filter((row) => dataset.developmentMask[row] === 1).length
    : null, [dataset, fixedValidation.audit]);
  const feedbackItemAllowed = useCallback((item: GalleryItem) => Boolean(
    dataset
    && developmentMode
    && !hierarchicalFusionSelected
    && fixedValAllowsFeedback(item, dataset.imageIds, dataset.developmentMask, fixedValidation.mask)
    && !dataset.manifest.query?.images.some((image) => image.imageIndex === item.rowIndex)
  ), [dataset, developmentMode, hierarchicalFusionSelected, fixedValidation.mask]);
  const feedbackRemovalAllowed = useCallback((item: GalleryItem) => Boolean(
    dataset
    && developmentMode
    && !hierarchicalFusionSelected
    && fixedValidation.status === "ready"
    && Number.isInteger(item.rowIndex)
    && item.rowIndex >= 0 && item.rowIndex < dataset.imageIds.length
    && dataset.imageIds[item.rowIndex] === item.id
    && (dataset.developmentMask[item.rowIndex] === 1 || fixedValidation.mask?.[item.rowIndex] === 1)
    && !dataset.manifest.query?.images.some((image) => image.imageIndex === item.rowIndex)
  ), [dataset, developmentMode, hierarchicalFusionSelected, fixedValidation.status, fixedValidation.mask]);
  const tuning = useTuningSession({
    // Keep an already-applied run available while changing data purpose so the
    // final read-only Test view can audit that exact ranking. Mutation controls
    // remain Development-only and the backend revalidates every training row.
    enabled: Boolean(dataset) && !hierarchicalFusionSelected,
    taskId: selectedTaskId,
    targetId: retrievalTarget,
    baseMethod: rankMethod,
    rowCount: dataset?.manifest.rowCount ?? 0,
    canWriteFeedback: feedbackItemAllowed,
    canRemoveFeedback: feedbackRemovalAllowed,
    validationReady: fixedValidation.status === "ready",
  });
  tuningApplyScopeRef.current = {
    taskId: selectedTaskId,
    runId: tuning.run?.id ?? null,
    sessionId: tuning.session?.id ?? null,
  };
  const originalVqa = useOriginalVqaSupervision({
    enabled: Boolean(dataset),
    taskId: selectedTaskId,
    targetId: retrievalTarget,
    rowCount: dataset?.manifest.rowCount ?? 0,
  });
  const visualClusterCount: VisualClusterCount = visualClusterScheme === "fine30"
    ? 30
    : visualClusterScheme === "fine100"
      ? 100
      : 50;
  const visualEmbeddingSource = useMemo<VisualEmbeddingSource | null>(
    () => dataset ? {
      taskId: selectedTaskId,
      dataRoot: dataset.dataRoot,
      rowCount: dataset.manifest.rowCount,
      manifest: dataset.manifest.visualEmbedding,
      files: dataset.manifest.files,
    } : null,
    [dataset, selectedTaskId],
  );
  const visualAnalysis = useVisualEmbeddingAnalysis(
    visualEmbeddingSource,
    visualClusterCount,
    Boolean(dataset) && !validationMode,
  );

  useEffect(() => {
    let active = true;
    fetchJson<DatasetCatalog>(CATALOG_PATH)
      .catch(() => FALLBACK_CATALOG)
      .then((loadedCatalog) => {
        if (!active) return;
        const firstDataset = loadedCatalog.datasets.find(
          (entry) => entry.id === loadedCatalog.defaultDataset,
        ) ?? loadedCatalog.datasets[0];
        const firstTask = firstDataset?.tasks.find(
          (entry) => entry.id === loadedCatalog.defaultTask,
        ) ?? firstDataset?.tasks[0];
        if (!firstDataset || !firstTask) throw new Error("No exported dataset tasks are available.");
        setCatalog(loadedCatalog);
        setSelectedDatasetId(firstDataset.id);
        setSelectedTaskId(firstTask.id);
      })
      .catch((error: unknown) => {
        if (active) setLoadError(error instanceof Error ? error.message : String(error));
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedTaskEntry) return;
    fusionTuneApplyGenerationRef.current += 1;
    hierarchicalApplyGenerationRef.current += 1;
    rankClusterSchemeGenerationRef.current += 1;
    const controller = new AbortController();
    let active = true;
    const taskKey = selectedTaskLoadKey;
    const catalogResolution = initialDownloadBytesFromCatalog(
      selectedTaskEntry.initialDownloadBytes,
    );
    const onManifest = (manifest: DashboardManifest) => {
      if (!active) return;
      const immediate = initialDownloadBytesFromManifest(manifest);
      // The catalog total also includes manifest.json itself, so keep it as
      // the authoritative estimate when available instead of replacing it
      // with the slightly smaller files-only manifest sum.
      const visibleResolution = catalogResolution ?? immediate;
      setTaskLoadEstimate((current) => {
        // A reload of the same task should not briefly replace a complete
        // estimate with an incomplete manifest lower bound.
        if (
          current?.taskKey === taskKey
          && current.resolution.complete
          && !visibleResolution.complete
        ) return current;
        return { taskKey, resolution: visibleResolution };
      });
      if (immediate.complete) return;
      void resolveInitialDownloadBytes(manifest, {
        dataRoot: selectedTaskEntry.dataRoot,
        catalogInitialDownloadBytes: selectedTaskEntry.initialDownloadBytes,
        signal: controller.signal,
      }).then((resolution) => {
        if (active) {
          setTaskLoadEstimate({ taskKey, resolution });
        }
      }).catch(() => {
        // The lower-bound estimate is still useful when HEAD is unsupported.
      });
    };
    loadDashboardDataset(selectedTaskEntry.dataRoot, controller.signal, onManifest)
      .then((loaded) => {
        if (!active) return;
        const targetIds = new Set(loaded.manifest.retrievalTargets.map((target) => target.id));
        const requestedDefault = selectedTaskEntry.defaultRetrievalTarget;
        const nextTarget = requestedDefault && targetIds.has(requestedDefault)
          ? requestedDefault
          : loaded.manifest.defaultRetrievalTarget;
        const baseAttributes = loaded.manifest.retrievalTargets
          .filter((target) => target.kind === "attribute")
          .map((target) => target.id);

        setDataset(loaded);
        setColorMode("cluster");
        setColorLearner(loaded.manifest.defaultRankMethod);
        setRankMethod(loaded.manifest.defaultRankMethod);
        setLegacyBaselineSelected(false);
        setRankClusterScheme(
          loaded.manifest.clusters.schemes.some((scheme) => (
            scheme.id === loaded.manifest.clusters.defaultScheme
          ))
            ? loaded.manifest.clusters.defaultScheme
            : loaded.manifest.clusters.schemes[0].id,
        );
        setRankClusterFilter("all");
        setVisualClusterScheme("fine50");
        setVisualClusterFilter("all");
        setClusterDrilldown(null);
        setRetrievalTarget(nextTarget);
        setPcpFullExpanded(true);
        setPcpAttributeEvidenceExpanded(true);
        setPcpHolisticExpanded(false);
        setExpandedPcpAttributes(new Set());
        setHierarchicalDraft(
          baseAttributes.length > 0
            ? createDefaultHierarchicalPcpConfig(baseAttributes)
            : createEmptyHierarchicalPcpConfig(),
        );
        setPcpValueKind("rank");
        setPcpLineMode("samples");
        setResultScope(loaded.manifest.evaluation.defaultResultScope === "test" ? "test" : "development");
        setGeneratedOnly(false);
        setSmartFilterKind(null);
        setSmartFilterLearner(
          SMART_FILTER_LEARNER_METHODS.find((method) => loaded.manifest.methods.includes(method))
            ?? loaded.manifest.methods[0],
        );
        setBrushes({});
        setPcpBrushZoom(createPcpBrushZoomState());
        setProjectionSelection(null);
        setSelectedId(null);
        setShowManualHighlights(false);
        setHoveredId(null);
        setQueryPreviewIndex(null);
        setProjection("umap");
      })
      .catch((error: unknown) => {
        if (!active || error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(error instanceof Error ? error.message : String(error));
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [reloadNonce, selectedTaskEntry, selectedTaskLoadKey]);

  const activeVisualAnalysis = visualAnalysis.data
    && dataset
    && visualAnalysis.data.taskId === selectedTaskId
    && visualAnalysis.data.rowCount === dataset.manifest.rowCount
    && visualAnalysis.data.clusterCount === visualClusterCount
    ? visualAnalysis.data
    : null;
  const visualClusterLabels = useMemo(
    () => activeVisualAnalysis?.labels ?? new Uint8Array(dataset?.manifest.rowCount ?? 0),
    [activeVisualAnalysis, dataset],
  );

  const activeRankClusterScheme = useMemo(
    () => dataset?.manifest.clusters.schemes.find((scheme) => (
      scheme.id === rankClusterScheme
    )) ?? null,
    [dataset, rankClusterScheme],
  );

  const activeVisualClusterScheme = useMemo(
    () => dataset?.manifest.visualEmbedding.clusterings.find((scheme) => (
      scheme.id === visualClusterScheme
    )) ?? null,
    [dataset, visualClusterScheme],
  );

  const resolveRankClusterColor = useCallback(
    (cluster: string | number) => clusterColor(cluster, rankClusterScheme),
    [rankClusterScheme],
  );

  const resolveVisualClusterColor = useCallback(
    (cluster: string | number) => clusterColor(cluster, visualClusterScheme),
    [visualClusterScheme],
  );

  const visualClusterSummaries = useMemo<ClusterSummary[]>(() => {
    if (!dataset) return [];
    if (!activeVisualAnalysis) return [];
    return Array.from({ length: visualClusterCount }, (_, clusterId) => ({
      scheme: visualClusterScheme,
      cluster_id: clusterId,
      label: `Visual ${String(clusterId + 1).padStart(2, "0")}`,
      label_zh: `Visual ${String(clusterId + 1).padStart(2, "0")}`,
      size: 0,
      fraction: 0,
      mean_rank: 0,
      learned_minus_fixed: 0,
    }));
  }, [activeVisualAnalysis, dataset, visualClusterCount, visualClusterScheme]);

  const isRobustCatTask = selectedTaskId === "059_hico_task_hico_hugging_cat_robust_test";
  const analysisMask = useMemo(
    () => {
      if (!dataset) return new Uint8Array();
      const mask = analysisScopeMask(dataset.manifest.rowCount, {
          developmentMask: developmentWithoutFixedVal(dataset.developmentMask, fixedValidation.mask),
          validationMask: dataset.validationMask,
          testMask: dataset.testMask,
        }, resultScope);
      if (isRobustCatTask && generatedOnly) {
        for (let row = 0; row < mask.length; row += 1) {
          if (!dataset.imageIds[row].startsWith("robust/")) mask[row] = 0;
        }
      }
      return mask;
    },
    [dataset, resultScope, fixedValidation.mask, isRobustCatTask, generatedOnly],
  );
  const scopeMasks = useMemo<ScopeMasks | null>(() => dataset ? ({
    developmentMask: dataset.developmentMask,
    validationMask: dataset.validationMask,
    testMask: dataset.testMask,
  }) : null, [dataset]);

  const analysisVisualClusterCounts = useMemo(() => {
    const counts = new Map<string, number>();
    let total = 0;
    if (!dataset || !activeVisualAnalysis) return { counts, total };
    for (let index = 0; index < dataset.manifest.rowCount; index += 1) {
      if (!analysisMask[index]) continue;
      const key = String(visualClusterLabels[index]);
      counts.set(key, (counts.get(key) ?? 0) + 1);
      total += 1;
    }
    return { counts, total };
  }, [activeVisualAnalysis, analysisMask, dataset, visualClusterLabels]);

  const targetIndex = useMemo(
    () => new Map(
      dataset?.manifest.retrievalTargets.map((target, index) => [target.id, index]) ?? [],
    ),
    [dataset],
  );

  const methodIndex = useMemo(
    () => new Map(dataset?.manifest.methods.map((method, index) => [method, index]) ?? []),
    [dataset],
  );

  const hierarchicalAttributeIds = useMemo(
    () => dataset?.manifest.retrievalTargets
      .filter((target) => target.kind === "attribute")
      .map((target) => target.id) ?? [],
    [dataset],
  );
  const initialBaseline = useInitialBaseline({
    taskId: selectedTaskId,
    rowCount: dataset?.manifest.rowCount ?? 0,
    attributeIds: hierarchicalAttributeIds,
    scheme: rankClusterScheme,
    enabled: Boolean(dataset),
    reloadKey: reloadNonce,
  });
  const hierarchicalFusion = useHierarchicalFusion({
    taskId: selectedTaskId,
    rowCount: dataset?.manifest.rowCount ?? 0,
    targetCount: dataset?.manifest.targetCount ?? 0,
    attributeIds: hierarchicalAttributeIds,
    disabled: !dataset || hierarchicalAttributeIds.length === 0,
  });
  // Personal Fusion Tune owns a separate request/result channel from the
  // manual Ours-Hierarchical editor. This prevents either Apply path, Revert,
  // or a late response from overwriting the other path's committed result.
  const fusionTuneHierarchy = useHierarchicalFusion({
    taskId: selectedTaskId,
    rowCount: dataset?.manifest.rowCount ?? 0,
    targetCount: dataset?.manifest.targetCount ?? 0,
    attributeIds: hierarchicalAttributeIds,
    disabled: !dataset || hierarchicalAttributeIds.length === 0,
  });

  const activeHierarchicalFusionResult = hierarchicalFusion.result
    && dataset
    && hierarchicalFusion.result.taskId === selectedTaskId
    && hierarchicalFusion.result.rowCount === dataset.manifest.rowCount
    && hierarchicalFusion.result.targetCount === dataset.manifest.targetCount
    ? hierarchicalFusion.result
    : null;
  const availableFusionTuneResult = fusionTuneHierarchy.result
    && dataset
    && fusionTuneHierarchy.result.taskId === selectedTaskId
    && fusionTuneHierarchy.result.rowCount === dataset.manifest.rowCount
    && fusionTuneHierarchy.result.targetCount === dataset.manifest.targetCount
    ? fusionTuneHierarchy.result
    : null;
  const displayMethods = useMemo(
    () => dataset
      ? [
          ...dataset.manifest.methods,
          ...(activeHierarchicalFusionResult ? [HIERARCHICAL_FUSION_METHOD_ID] : []),
        ]
      : [],
    [activeHierarchicalFusionResult, dataset],
  );

  const smartFilterLearners = useMemo(
    () => SMART_FILTER_LEARNER_METHODS.filter((method) => methodIndex.has(method)),
    [methodIndex],
  );

  const targetLabels = useMemo(
    () => new Map(
      dataset?.manifest.retrievalTargets.map((target) => [target.id, target.label]) ?? [],
    ),
    [dataset],
  );

  const attributeTargets = useMemo(
    () => dataset?.manifest.retrievalTargets.filter((target) => target.kind === "attribute") ?? [],
    [dataset],
  );

  const jointTarget = useMemo(() => {
    if (!dataset) return null;
    const direct = dataset.manifest.retrievalTargets.find((target) => target.id === "joint");
    if (direct) return direct;
    const attributeIds = new Set(attributeTargets.map((target) => target.id));
    return dataset.manifest.retrievalTargets.find(
      (target) => target.kind === "derived"
        && target.members?.length === attributeIds.size
        && target.members.every((member) => attributeIds.has(member)),
    ) ?? dataset.manifest.retrievalTargets.find((target) => target.kind === "derived") ?? null;
  }, [attributeTargets, dataset]);

  const defaultHierarchicalConfig = useMemo<HierarchicalPcpConfig>(
    () => hierarchicalAttributeIds.length > 0
      ? createDefaultHierarchicalPcpConfig(hierarchicalAttributeIds)
      : createEmptyHierarchicalPcpConfig(),
    [hierarchicalAttributeIds],
  );
  const activeFusionTuneResult = useMemo(() => {
    if (
      !appliedFusionTune
      || !availableFusionTuneResult
      || appliedFusionTune.taskId !== selectedTaskId
      || hierarchicalAttributeIds.length === 0
    ) return null;
    try {
      return hierarchicalPcpConfigFingerprint(
        availableFusionTuneResult.config,
        hierarchicalAttributeIds,
      ) === appliedFusionTune.configFingerprint
        ? availableFusionTuneResult
        : null;
    } catch {
      return null;
    }
  }, [
    appliedFusionTune,
    availableFusionTuneResult,
    hierarchicalAttributeIds,
    selectedTaskId,
  ]);
  const activeRuntimeFusionResult = activeFusionTuneResult
    ?? (hierarchicalFusionSelected ? activeHierarchicalFusionResult : null);
  const activeHierarchicalConfig = activeRuntimeFusionResult?.config
    ?? defaultHierarchicalConfig;

  const activeTunedOutputMatches = Boolean(
    tuning.appliedRanks
    && tuning.appliedScores
    && tuning.run?.status === "succeeded"
    && tuning.appliedRunId === tuning.run.id
    && tuning.session?.taskId === selectedTaskId
    && tuning.run.sessionId === tuning.session.id
    && tuning.session.targetId === retrievalTarget
    && tuning.session.baseMethod === rankMethod,
  );
  const activeTunedRanks = activeTunedOutputMatches ? tuning.appliedRanks : null;
  const activeTunedScores = activeTunedOutputMatches ? tuning.appliedScores : null;
  const activeTunedCalibratedScores = activeTunedScores
    && tuning.run
    && (
      tuning.run.mode === "fusion-weight"
      || tuning.run.mode === "residual"
      || tuning.run.mode === "weight_staged"
      || tuning.run.mode === "weight_joint"
      || tuning.run.mode === "probe_staged"
      || tuning.run.mode === "probe_joint"
    )
    ? activeTunedScores
    : null;
  const activeInitialBaseline = !legacyBaselineSelected && rankMethod === "Ours-Full"
    && !activeTunedOutputMatches && !activeRuntimeFusionResult
    ? initialBaseline.data : null;
  const personalRefinementEmbeddingMethods = useMemo(
    () => refinementEmbeddingMethods(tuning.run),
    [tuning.run],
  );
  const activeRefinementEmbeddingMethods = activeInitialBaseline
    ? REFINEMENT_EMBEDDING_METHODS : personalRefinementEmbeddingMethods;
  const refinementComponentIds = useMemo(
    () => jointTarget && activeRefinementEmbeddingMethods
      ? refinementPcpComponentIds(hierarchicalAttributeIds, jointTarget.id, activeRefinementEmbeddingMethods)
      : [],
    [activeRefinementEmbeddingMethods, hierarchicalAttributeIds, jointTarget],
  );
  const activePersonalRefinementVisualization = activeTunedOutputMatches
    && dataset
    && tuning.run
    && isWeightRefinementRun(tuning.run)
    && tuning.appliedVisualization?.runId === tuning.run.id
    && tuning.appliedVisualization.rowCount === dataset.manifest.rowCount
    && tuning.appliedVisualization.componentCount === refinementComponentIds.length
    && tuning.appliedVisualization.attributeIds.length === hierarchicalAttributeIds.length
    && tuning.appliedVisualization.attributeIds.every((id, index) => id === hierarchicalAttributeIds[index])
    ? tuning.appliedVisualization
    : null;
  const activeRefinementVisualization = activePersonalRefinementVisualization
    ?? activeInitialBaseline?.snapshot ?? null;
  const activeRefinementWeights = useMemo<RefinementPcpWeights | null>(() => {
    const summary = activePersonalRefinementVisualization
      ? tuning.run?.after?.modelSummary : activeInitialBaseline?.manifest.modelSummary;
    if (
      !summary?.beta
      || !summary.gamma
      || !summary.embeddingWeights
      || !Number.isFinite(summary.embeddingFusionStrength)
    ) return null;
    const complete = hierarchicalAttributeIds.every((attributeId) => (
      Number.isFinite(summary.gamma?.[attributeId])
      && HIERARCHICAL_PCP_LEARNERS.every((method) => (
        Number.isFinite(summary.beta?.[attributeId]?.[method])
      ))
    )) && activeRefinementEmbeddingMethods?.every((method) => (
      Number.isFinite(summary.embeddingWeights?.[method])
    ));
    if (!complete) return null;
    return {
      beta: summary.beta,
      gamma: summary.gamma,
      embeddingWeights: summary.embeddingWeights,
      embeddingFusionStrength: summary.embeddingFusionStrength!,
    };
  }, [activeRefinementEmbeddingMethods, activePersonalRefinementVisualization, activeInitialBaseline, hierarchicalAttributeIds, tuning.run]);

  const runtimeRankClusters = useRuntimeRankProfileClusters({
    taskId: selectedTaskId,
    rowCount: dataset?.manifest.rowCount ?? 0,
    targetCount: dataset?.manifest.targetCount ?? 0,
    attributeIds: hierarchicalAttributeIds,
    config: activeHierarchicalConfig,
    scheme: rankClusterScheme as RuntimeRankProfileClusterScheme,
    disabled: !dataset || !activeRuntimeFusionResult || hierarchicalAttributeIds.length === 0,
  });
  const activeRuntimeConfigFingerprint = activeRuntimeFusionResult
    ? hierarchicalPcpConfigFingerprint(activeRuntimeFusionResult.config, hierarchicalAttributeIds)
    : null;
  const activeRuntimeRankClusters = runtimeRankClusters.data
    && activeRuntimeConfigFingerprint
    && runtimeRankClusters.data.taskId === selectedTaskId
    && runtimeRankClusters.data.scheme === rankClusterScheme
    && runtimeRankClusters.data.configFingerprint === activeRuntimeConfigFingerprint
    ? runtimeRankClusters.data
    : null;
  const refinementRankClusters = useRefinementClusters({
    run: activePersonalRefinementVisualization ? tuning.run : null,
    snapshot: activePersonalRefinementVisualization,
    scheme: rankClusterScheme,
    disabled: !activePersonalRefinementVisualization,
  });
  const personalRefinementRankClusters = refinementRankClusters.data
    && activePersonalRefinementVisualization
    && tuning.run
    && refinementRankClusters.data.runId === tuning.run.id
    && refinementRankClusters.data.scheme === rankClusterScheme
    && refinementRankClusters.data.sourceFingerprint
      === activePersonalRefinementVisualization.sourceFingerprint
    ? refinementRankClusters.data
    : null;
  const activeRefinementRankClusters = activeInitialBaseline
    ? activeInitialBaseline.clusters.scheme === rankClusterScheme ? activeInitialBaseline.clusters : null
    : personalRefinementRankClusters;
  const rankClustersReady = activeRefinementVisualization
    ? Boolean(activeRefinementRankClusters)
    : !activeRuntimeFusionResult || Boolean(activeRuntimeRankClusters);
  const rankClusterLabels = useMemo(() => {
    if (!dataset) return new Uint8Array();
    if (activeRefinementVisualization) {
      return activeRefinementRankClusters?.labels ?? new Uint8Array();
    }
    if (activeRuntimeFusionResult) {
      return activeRuntimeRankClusters?.labels ?? new Uint8Array();
    }
    return dataset.clusterLabels.get(rankClusterScheme) ?? new Uint8Array();
  }, [
    activeRefinementRankClusters,
    activeRefinementVisualization,
    activeRuntimeFusionResult,
    activeRuntimeRankClusters,
    dataset,
    rankClusterScheme,
  ]);
  const rankClusterSourceKey = activeInitialBaseline
    ? initialBaseline.sourceKey
    : activePersonalRefinementVisualization ? refinementRankClusters.sourceKey
    : activeRuntimeFusionResult
      ? runtimeRankClusters.sourceKey
      : [
        "static",
        selectedTaskId,
        rankClusterScheme,
        activeRankClusterScheme
          ? dataset?.manifest.files[activeRankClusterScheme.labelsFileKey]?.sha256 ?? "no-sha"
          : "no-scheme",
      ].join("|");
  const rankClusterSummaries = useMemo<ClusterSummary[]>(() => {
    if (!dataset) return [];
    if (!activeRefinementVisualization && !activeRuntimeFusionResult) {
      return dataset.metadata.clusterSummary.filter((row) => row.scheme === rankClusterScheme);
    }
    const clusterCount = activeRefinementVisualization
      ? activeRefinementRankClusters?.clusterCount
      : activeRuntimeRankClusters?.clusterCount;
    if (!clusterCount) return [];
    return Array.from({ length: clusterCount }, (_, clusterId) => ({
      scheme: rankClusterScheme,
      cluster_id: clusterId,
      label: `Current PCP ${String(clusterId + 1).padStart(2, "0")}`,
      label_zh: `Current PCP ${String(clusterId + 1).padStart(2, "0")}`,
      size: 0,
      fraction: 0,
      mean_rank: 0,
      learned_minus_fixed: 0,
    }));
  }, [
    activeRefinementRankClusters,
    activeRefinementVisualization,
    activeRuntimeFusionResult,
    activeRuntimeRankClusters,
    dataset,
    rankClusterScheme,
  ]);
  const hierarchicalWeightsReadOnly = testMode || Boolean(activeRefinementVisualization);
  const displayedHierarchicalConfig = hierarchicalWeightsReadOnly
    ? activeHierarchicalConfig
    : hierarchicalDraft;

  const preferences = useMemo(
    () => hierarchicalFusionSelected
      ? new Map<string, PreferenceState>()
      : tuning.preferences,
    [hierarchicalFusionSelected, tuning.preferences],
  );
  const queryRowIndices = useMemo(
    () => new Set(dataset?.manifest.query?.images.map((image) => image.imageIndex) ?? []),
    [dataset],
  );
  const canAnnotate = useCallback((item: GalleryItem) => (
    dataset !== null
    && developmentMode
    && !hierarchicalFusionSelected
    && tuning.serviceState === "ready"
    && tuning.session?.taskId === selectedTaskId
    && tuning.session.targetId === retrievalTarget
    && tuning.session.baseMethod === rankMethod
    && dataset.developmentMask[item.rowIndex] === 1
    && feedbackItemAllowed(item)
    && !queryRowIndices.has(item.rowIndex)
  ), [dataset, developmentMode, hierarchicalFusionSelected, queryRowIndices, tuning.serviceState,
    tuning.session, selectedTaskId, retrievalTarget, rankMethod, feedbackItemAllowed]);
  const annotationDisabledReason = useCallback((item: GalleryItem) => {
    if (!developmentMode) {
      return validationMode
        ? "Validation is metrics-only and cannot be annotated."
        : "Frozen Test images are evaluation-only.";
    }
    if (hierarchicalFusionSelected) {
      return "Custom Fusion is an exploratory ranking; switch to an exported method before saving feedback.";
    }
    if (!dataset || tuning.serviceState !== "ready") return "The private tuning service is not ready.";
    if (fixedValidation.status !== "ready") return fixedValidation.error ?? "Loading fixed Val membership before feedback is enabled.";
    if (fixedValidation.mask?.[item.rowIndex] === 1) return "Fixed Val images cannot be annotated or used for tuning feedback.";
    if (!dataset.developmentMask[item.rowIndex]) return "Only Development Gallery images can become training feedback.";
    if (queryRowIndices.has(item.rowIndex)) return "Fixed Query images cannot be training feedback.";
    return "Feedback is unavailable for this image.";
  }, [dataset, developmentMode, hierarchicalFusionSelected, queryRowIndices, tuning.serviceState, validationMode,
    fixedValidation.status, fixedValidation.error, fixedValidation.mask]);
  const canRemoveAnnotation = useCallback((item: GalleryItem) => Boolean(
    feedbackRemovalAllowed(item)
    && tuning.serviceState === "ready"
    && tuning.session?.taskId === selectedTaskId
    && tuning.session.targetId === retrievalTarget
    && tuning.session.baseMethod === rankMethod
    && tuning.annotations.has(item.id)
  ), [feedbackRemovalAllowed, tuning.serviceState, tuning.session, tuning.annotations,
    selectedTaskId, retrievalTarget, rankMethod]);
  const isValidationRow = useCallback((item: GalleryItem) => fixedValidation.mask?.[item.rowIndex] === 1,
    [fixedValidation.mask]);

  const rankStructuralMask = useMemo(() => {
    if (!dataset) return new Uint8Array();
    const mask = new Uint8Array(dataset.manifest.rowCount);
    for (let index = 0; index < mask.length; index += 1) {
      if (!analysisMask[index]) continue;
      if (
        rankClustersReady
        &&
        rankClusterFilter !== "all"
        && rankClusterLabels[index] !== Number(rankClusterFilter)
      ) continue;
      mask[index] = 1;
    }
    return mask;
  }, [analysisMask, dataset, rankClusterFilter, rankClusterLabels, rankClustersReady]);

  const visualStructuralMask = useMemo(() => {
    if (!dataset) return new Uint8Array();
    const mask = new Uint8Array(dataset.manifest.rowCount);
    for (let index = 0; index < mask.length; index += 1) {
      if (!analysisMask[index]) continue;
      if (
        activeVisualAnalysis
        &&
        visualClusterFilter !== "all"
        && visualClusterLabels[index] !== Number(visualClusterFilter)
      ) continue;
      mask[index] = 1;
    }
    return mask;
  }, [activeVisualAnalysis, analysisMask, dataset, visualClusterFilter, visualClusterLabels]);

  const pcpAxes = useMemo(() => {
    if (!dataset || !jointTarget || attributeTargets.length === 0) return [];
    if (activeRefinementVisualization) {
      return buildRefinementPcpAxes({
        attributes: attributeTargets.map((target) => ({ id: target.id, label: target.label })),
        jointTargetId: jointTarget.id,
        fullExpanded: pcpFullExpanded,
        attributeEvidenceExpanded: pcpAttributeEvidenceExpanded,
        expandedAttributeIds: expandedPcpAttributes,
        holisticExpanded: pcpHolisticExpanded,
        embeddingMethods: activeRefinementEmbeddingMethods!,
        fullLabel: "Overall",
      });
    }
    return buildHierarchicalPcpAxes({
      attributes: attributeTargets.map((target) => ({ id: target.id, label: target.label })),
      jointTargetId: jointTarget.id,
      fullExpanded: pcpFullExpanded,
      expandedAttributeIds: expandedPcpAttributes,
      fullLabel: activeFusionTuneResult
        ? `${appliedFusionTune?.label ?? "Fusion Weight Tune"} overall`
        : activeRuntimeFusionResult
          ? "Weighted Fusion overall"
          : initialBaseline.data ? "Legacy Ours-Full overall" : "Ours-Full overall",
    });
  }, [
    activeRefinementVisualization,
    activeRefinementEmbeddingMethods,
    activeFusionTuneResult,
    appliedFusionTune?.label,
    activeRuntimeFusionResult,
    attributeTargets,
    dataset,
    expandedPcpAttributes,
    jointTarget,
    initialBaseline.data,
    pcpFullExpanded,
    pcpAttributeEvidenceExpanded,
    pcpHolisticExpanded,
  ]);

  const pcpAxisIds = useMemo(() => pcpAxes.map((axis) => axis.id), [pcpAxes]);
  const pcpAxisLabels = useMemo(
    () => Object.fromEntries(pcpAxes.map((axis) => [
      axis.id,
      (axis.kind === "learner" || axis.kind === "baseline") && axis.attributeId
        ? `${targetLabels.get(axis.attributeId) ?? axis.attributeId} · ${axis.label}`
        : axis.label,
    ])),
    [pcpAxes, targetLabels],
  );
  const enabledAxisIds = useMemo(
    () => new Set(pcpAxes.map((axis) => axis.id)),
    [pcpAxes],
  );
  const refinementComponentIndex = useMemo(
    () => new Map(refinementComponentIds.map((componentId, index) => [componentId, index])),
    [refinementComponentIds],
  );
  const pcpValues = useMemo(() => {
    if (!dataset || pcpAxes.length === 0) return new Float32Array();
    if (activeRefinementVisualization) {
      const source = pcpValueKind === "rank"
        ? activeRefinementVisualization.ranks
        : activeRefinementVisualization.scores;
      const values = new Float32Array(dataset.manifest.rowCount * pcpAxes.length);
      for (let rowIndex = 0; rowIndex < dataset.manifest.rowCount; rowIndex += 1) {
        for (let axisIndex = 0; axisIndex < pcpAxes.length; axisIndex += 1) {
          const componentIndex = refinementComponentIndex.get(pcpAxes[axisIndex].id);
          values[rowIndex * pcpAxes.length + axisIndex] = componentIndex === undefined
            ? Number.NaN
            : source[
                rowIndex * activeRefinementVisualization.componentCount + componentIndex
              ];
        }
      }
      return values;
    }
    const source = pcpValueKind === "rank" ? dataset.ranks : dataset.calibratedScores;
    const finalSource = activeRuntimeFusionResult
      ? (pcpValueKind === "rank"
          ? activeRuntimeFusionResult.ranks
          : activeRuntimeFusionResult.calibratedScores)
      : null;
    const oursFullIndex = methodIndex.get("Ours-Full");
    const values = new Float32Array(dataset.manifest.rowCount * pcpAxes.length);
    for (let rowIndex = 0; rowIndex < dataset.manifest.rowCount; rowIndex += 1) {
      for (let axisIndex = 0; axisIndex < pcpAxes.length; axisIndex += 1) {
        const axis = pcpAxes[axisIndex];
        const currentTargetIndex = targetIndex.get(axis.targetId);
        const runtimeSource = axis.kind === "full" || axis.kind === "attribute"
          ? finalSource
          : null;
        const currentMethodIndex = axis.methodId
          ? methodIndex.get(axis.methodId)
          : oursFullIndex;
        if (currentTargetIndex === undefined || currentMethodIndex === undefined) {
          values[rowIndex * pcpAxes.length + axisIndex] = Number.NaN;
          continue;
        }
        values[rowIndex * pcpAxes.length + axisIndex] = runtimeSource
          ? runtimeSource[rowIndex * dataset.manifest.targetCount + currentTargetIndex]
          : source[scoreOffset(
              rowIndex,
              currentMethodIndex,
              currentTargetIndex,
              dataset.manifest.methodCount,
              dataset.manifest.targetCount,
            )];
      }
    }
    return values;
  }, [
    activeRefinementVisualization,
    activeRuntimeFusionResult,
    dataset,
    methodIndex,
    pcpAxes,
    pcpValueKind,
    refinementComponentIndex,
    targetIndex,
  ]);
  const pcpZoomActive = hasPcpBrushZoom(pcpBrushZoom);
  const currentPcpZoomDepth = pcpBrushZoomDepth(pcpBrushZoom);
  const pcpCandidateMask = useMemo(() => (
    pcpZoomActive
      ? applyPcpBrushZoomMask({
          values: pcpValues,
          axisIds: pcpAxisIds,
          candidateMask: rankStructuralMask,
          ranges: pcpBrushZoom.ranges,
        })
      : rankStructuralMask
  ), [pcpAxisIds, pcpBrushZoom.ranges, pcpValues, pcpZoomActive, rankStructuralMask]);
  const pcpColorAxis = pcpAxisIds.includes(colorLearner)
    ? colorLearner
    : (pcpAxisIds[0] ?? "");
  const pcpPanel = usePcpPanelAlignment({
    frameRef: pcpFrameRef,
    explorationRef: explorationContentRef,
    layoutKey: `${selectedTaskId}:${resultScope}:${pcpLineMode}:${Boolean(clusterDrilldown)}`,
    minimumPlotHeight: Math.max(260, pcpAxes.length * 34 + 90),
    enabled: Boolean(dataset) && !loadError && !validationMode,
  });
  const pcpHeight = pcpPanel.plotHeight;
  const summaryPcpHeight = pcpPanel.plotHeight;

  const getOriginalVqaLabel = useCallback((item: GalleryItem) => (
    originalVqa.recordsByRow.get(item.rowIndex)?.label ?? null
  ), [originalVqa.recordsByRow]);

  const getAttributeStrengths = useCallback((item: GalleryItem): readonly AttributeStrengthPoint[] => {
    if (!dataset) return [];
    const staticMethodIndex = methodIndex.get(rankMethod);
    return attributeTargets.map((target) => {
      const currentTargetIndex = targetIndex.get(target.id);
      let value = Number.NaN;
      if (currentTargetIndex !== undefined) {
        const virtualOffset = item.rowIndex * dataset.manifest.targetCount + currentTargetIndex;
        const refinementIndex = activeRefinementVisualization
          ? refinementComponentIndex.get(hierarchicalAttributeAxisId(target.id))
          : undefined;
        if (activeRefinementVisualization && refinementIndex !== undefined) {
          value = activeRefinementVisualization.scores[
            item.rowIndex * activeRefinementVisualization.componentCount + refinementIndex
          ];
        } else if (activeRuntimeFusionResult) {
          value = activeRuntimeFusionResult.calibratedScores[virtualOffset];
        } else if (
          activeTunedCalibratedScores
          && target.id === retrievalTarget
          && target.kind === "attribute"
        ) {
          value = activeTunedCalibratedScores[item.rowIndex];
        } else if (staticMethodIndex !== undefined) {
          value = dataset.calibratedScores[scoreOffset(
            item.rowIndex,
            staticMethodIndex,
            currentTargetIndex,
            dataset.manifest.methodCount,
            dataset.manifest.targetCount,
          )];
        }
      }
      return {
        id: target.id,
        label: target.label,
        value,
        active: target.id === retrievalTarget,
      };
    });
  }, [
    activeRefinementVisualization,
    activeRuntimeFusionResult,
    activeTunedCalibratedScores,
    attributeTargets,
    dataset,
    methodIndex,
    rankMethod,
    refinementComponentIndex,
    retrievalTarget,
    targetIndex,
  ]);

  const hasActivePcpBrush = Object.keys(brushes).length > 0;
  const effectivePcpRanges = useMemo(
    () => effectivePcpSelectionRanges(pcpAxisIds, pcpBrushZoom.ranges, brushes),
    [brushes, pcpAxisIds, pcpBrushZoom.ranges],
  );
  const effectivePcpRangeMap = useMemo(
    () => Object.fromEntries(effectivePcpRanges.map(({ axisId, range }) => [axisId, range])),
    [effectivePcpRanges],
  );
  const hasPcpRestriction = effectivePcpRanges.length > 0;
  const pcpIndependentMask = useMemo(() => {
    return applyPcpBrushZoomMask({
      values: pcpValues,
      axisIds: pcpAxisIds,
      candidateMask: analysisMask,
      ranges: effectivePcpRangeMap,
    });
  }, [
    analysisMask,
    effectivePcpRangeMap,
    pcpAxisIds,
    pcpValues,
  ]);
  const pcpEligibleMask = useMemo(
    () => intersectMasks(rankStructuralMask, pcpIndependentMask),
    [pcpIndependentMask, rankStructuralMask],
  );
  const activePcpHighlightMask = hasPcpRestriction ? pcpEligibleMask : null;
  const projectionEligibleMask = useMemo(
    () => intersectMasks(pcpEligibleMask, visualStructuralMask),
    [pcpEligibleMask, visualStructuralMask],
  );
  const projectionMask = useMemo(() => {
    if (!dataset || !activeVisualAnalysis || !projectionSelection) return null;
    const coordinates = projectionSelection.projection === "umap"
      ? activeVisualAnalysis.umap2d
      : activeVisualAnalysis.pca2d;
    if (!coordinates) return null;
    const [xLower, xUpper] = projectionSelection.xDomain;
    const [yLower, yUpper] = projectionSelection.yDomain;
    const mask = new Uint8Array(dataset.manifest.rowCount);
    for (let rowIndex = 0; rowIndex < dataset.manifest.rowCount; rowIndex += 1) {
      if (!analysisMask[rowIndex]) continue;
      const x = coordinates[rowIndex * 2];
      const y = coordinates[rowIndex * 2 + 1];
      if (
        Number.isFinite(x)
        && Number.isFinite(y)
        && x >= xLower
        && x <= xUpper
        && y >= yLower
        && y <= yUpper
      ) {
        mask[rowIndex] = 1;
      }
    }
    return mask;
  }, [activeVisualAnalysis, analysisMask, dataset, projectionSelection]);

  const pcpSelectedVisualClusterCounts = useMemo(() => {
    const counts = new Map<string, number>();
    if (!dataset || !activeVisualAnalysis || !activePcpHighlightMask) return counts;
    for (let index = 0; index < dataset.manifest.rowCount; index += 1) {
      if (!analysisMask[index] || !activePcpHighlightMask[index]) continue;
      const key = String(visualClusterLabels[index]);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
  }, [activePcpHighlightMask, activeVisualAnalysis, analysisMask, dataset, visualClusterLabels]);
  const preRuleMask = useMemo(
    () => intersectMasks(projectionEligibleMask, projectionMask),
    [projectionEligibleMask, projectionMask],
  );
  // Rule is now a read-only description of this selection. It must never feed
  // back into the mask that it summarizes.
  const finalMask = preRuleMask;
  const scopedResultMask = useMemo(
    () => dataset && scopeMasks
      ? applyResultScope(finalMask, scopeMasks, resultScope)
      : new Uint8Array(),
    [dataset, finalMask, resultScope, scopeMasks],
  );
  const diagnosticActive = smartFilterKind !== null;
  const diagnosticInitialExpected = !legacyBaselineSelected && rankMethod === "Ours-Full"
    && !activeTunedOutputMatches && !activeRuntimeFusionResult;
  const diagnosticSnapshotRequired = Boolean(activeRefinementVisualization
    || diagnosticInitialExpected
    || (activeTunedOutputMatches && tuning.run && isWeightRefinementRun(tuning.run)));
  const diagnosticSource = useMemo(() => {
    if (!activeRefinementVisualization) {
      return {
        values: null,
        error: diagnosticSnapshotRequired ? "Waiting for a valid current model snapshot." : null,
      };
    }
    if (!diagnosticActive) return { values: null, error: null };
    try {
      const values = buildSnapshotDiagnosticValues(
        activeRefinementVisualization, retrievalTarget, smartFilterLearners,
      );
      // Use the same committed target rank as Top; never the pending training result.
      if (activeTunedRanks) {
        values.fusionRanks = activeTunedRanks;
        values.comparisonRanks = activeTunedRanks;
      }
      return { values, error: null };
    } catch (error) {
      return {
        values: null,
        error: error instanceof Error ? error.message : "The current diagnostic source is invalid.",
      };
    }
  }, [
    activeRefinementVisualization,
    activeTunedRanks,
    diagnosticActive,
    diagnosticSnapshotRequired,
    retrievalTarget,
    smartFilterLearners,
  ]);
  const unavailableDiagnosticMask = useMemo(
    () => new Uint8Array(dataset?.manifest.rowCount ?? 0),
    [dataset],
  );
  const smartFilterResult = useMemo(() => {
    if (!dataset || !smartFilterKind) return null;
    if (diagnosticSource.error) return null;
    const currentTargetIndex = targetIndex.get(retrievalTarget);
    const fusionMethodIndex = methodIndex.get("Ours-Full");
    const prototypeMethodIndex = methodIndex.get("Image Prototype");
    const selectedLearnerIndex = methodIndex.get(smartFilterLearner);
    const comparisonMethodIndex = methodIndex.get(rankMethod) ?? fusionMethodIndex;
    const currentTarget = dataset.manifest.retrievalTargets.find(
      (target) => target.id === retrievalTarget,
    );
    const memberIds = currentTarget?.kind === "derived" && currentTarget.members?.length
      ? currentTarget.members
      : [retrievalTarget];
    const targetMemberIndices = memberIds.flatMap((target) => {
      const index = targetIndex.get(target);
      return index === undefined ? [] : [index];
    });
    const learnerMethodIndices = smartFilterLearners.flatMap((method) => {
      const index = methodIndex.get(method);
      return index === undefined ? [] : [index];
    });
    if (
      currentTargetIndex === undefined
      || fusionMethodIndex === undefined
      || prototypeMethodIndex === undefined
      || selectedLearnerIndex === undefined
      || comparisonMethodIndex === undefined
      || learnerMethodIndices.length !== SMART_FILTER_LEARNER_METHODS.length
    ) return null;
    return buildSmartFilterMask(smartFilterKind, {
      rowCount: dataset.manifest.rowCount,
      methodCount: dataset.manifest.methodCount,
      targetCount: dataset.manifest.targetCount,
      targetIndex: currentTargetIndex,
      targetMemberIndices,
      learnerMethodIndices,
      fusionMethodIndex,
      prototypeMethodIndex,
      selectedLearnerIndex,
      comparisonMethodIndex,
      rawScores: dataset.rawScores,
      ranks: dataset.ranks,
      liveValues: diagnosticSource.values ?? undefined,
      candidateMask: scopedResultMask,
    });
  }, [
    dataset,
    diagnosticSource,
    methodIndex,
    rankMethod,
    retrievalTarget,
    scopedResultMask,
    smartFilterKind,
    smartFilterLearner,
    smartFilterLearners,
    targetIndex,
  ]);
  const resultMask = smartFilterResult?.mask
    ?? (smartFilterKind && diagnosticSource.error ? unavailableDiagnosticMask : scopedResultMask);
  const resultCandidateCount = useMemo(() => countMask(resultMask), [resultMask]);
  const selectionRuleActive = rankClusterFilter !== "all"
    || visualClusterFilter !== "all"
    || hasPcpRestriction
    || projectionMask !== null
    || smartFilterKind !== null;
  const selectionOverlapConditions = useMemo(() => {
    const conditions: Array<Omit<SelectionOverlapConditionInput, "ruleResult">> = [];
    if (rankClusterFilter !== "all" && rankClustersReady) {
      const clusterNumber = String(Number(rankClusterFilter) + 1).padStart(2, "0");
      conditions.push({
        id: "rank-cluster",
        label: `PCP cluster ${clusterNumber}`,
        rule: {
          kind: "category",
          field: "rank-cluster",
          value: `${activeRankClusterScheme?.label ?? CLUSTER_SCHEME_LABELS[rankClusterScheme]}${activeRankClusterScheme ? ` · K${activeRankClusterScheme.clusters}` : ""} · ${clusterNumber}`,
        },
        mask: rankStructuralMask,
      });
    }
    for (const { axisId, range, source } of effectivePcpRanges) {
      conditions.push({
        id: `pcp-axis:${axisId}`,
        label: pcpAxisLabels[axisId] ?? axisId,
        rule: {
          kind: "range",
          axisId,
          valueKind: pcpValueKind,
          lower: range[0],
          upper: range[1],
          source,
        },
        mask: applyPcpBrushZoomMask({
          values: pcpValues,
          axisIds: pcpAxisIds,
          candidateMask: analysisMask,
          ranges: { [axisId]: range },
        }),
      });
    }
    if (visualClusterFilter !== "all" && activeVisualAnalysis) {
      const clusterNumber = String(Number(visualClusterFilter) + 1).padStart(2, "0");
      conditions.push({
        id: "visual-cluster",
        label: `Visual cluster ${clusterNumber}`,
        rule: {
          kind: "category",
          field: "visual-cluster",
          value: `${activeVisualClusterScheme?.label ?? "Visual embedding"}${activeVisualClusterScheme ? ` · K${activeVisualClusterScheme.clusters}` : ""} · ${clusterNumber}`,
        },
        mask: visualStructuralMask,
      });
    }
    if (projectionSelection && projectionMask) {
      conditions.push({
        id: "projection-brush",
        label: `Embedding brush · ${projectionSelection.projection.toUpperCase()}`,
        rule: {
          kind: "projection",
          projection: projectionSelection.projection,
          xDomain: projectionSelection.xDomain,
          yDomain: projectionSelection.yDomain,
        },
        mask: projectionMask,
      });
    }
    return conditions;
  }, [
    activeRankClusterScheme,
    activeVisualAnalysis,
    activeVisualClusterScheme,
    analysisMask,
    effectivePcpRanges,
    pcpAxisIds,
    pcpAxisLabels,
    pcpValueKind,
    pcpValues,
    rankClusterFilter,
    rankClustersReady,
    rankClusterScheme,
    rankStructuralMask,
    visualClusterFilter,
    visualStructuralMask,
    projectionMask,
    projectionSelection,
  ]);
  const selectionOverlapStages = useMemo(() => {
    const stages: SelectionOverlapStageInput[] = [];
    if (smartFilterKind !== null && smartFilterResult) {
      stages.push({ id: "diagnostic", label: "Diagnostic Top 50", mask: resultMask });
    }
    return stages;
  }, [resultMask, smartFilterKind, smartFilterResult]);
  const selectionRuleAttributes = useMemo(() => {
    if (!dataset) return [];
    const staticMethodIndex = methodIndex.get(rankMethod);
    return attributeTargets.flatMap((target) => {
      const currentTargetIndex = targetIndex.get(target.id);
      if (currentTargetIndex === undefined) return [];
      const refinementIndex = activeRefinementVisualization
        ? refinementComponentIndex.get(hierarchicalAttributeAxisId(target.id))
        : undefined;
      const ranks = new Float32Array(dataset.manifest.rowCount);
      const calibratedScores = new Float32Array(dataset.manifest.rowCount);
      for (let rowIndex = 0; rowIndex < dataset.manifest.rowCount; rowIndex += 1) {
        const virtualOffset = rowIndex * dataset.manifest.targetCount + currentTargetIndex;
        const staticOffset = staticMethodIndex === undefined
          ? -1
          : scoreOffset(
              rowIndex,
              staticMethodIndex,
              currentTargetIndex,
              dataset.manifest.methodCount,
              dataset.manifest.targetCount,
            );
        if (activeRefinementVisualization && refinementIndex !== undefined) {
          const refinementOffset = (
            rowIndex * activeRefinementVisualization.componentCount + refinementIndex
          );
          ranks[rowIndex] = activeRefinementVisualization.ranks[refinementOffset];
          calibratedScores[rowIndex] = activeRefinementVisualization.scores[refinementOffset];
        } else if (activeRuntimeFusionResult) {
          ranks[rowIndex] = activeRuntimeFusionResult.ranks[virtualOffset];
          calibratedScores[rowIndex] = activeRuntimeFusionResult.calibratedScores[virtualOffset];
        } else {
          ranks[rowIndex] = activeTunedRanks && target.id === retrievalTarget
            ? activeTunedRanks[rowIndex]
            : staticOffset >= 0
              ? dataset.ranks[staticOffset]
              : Number.NaN;
          calibratedScores[rowIndex] = activeTunedCalibratedScores
            && target.id === retrievalTarget
            ? activeTunedCalibratedScores[rowIndex]
            : staticOffset >= 0
              ? dataset.calibratedScores[staticOffset]
              : Number.NaN;
        }
      }
      return [{ id: target.id, label: target.label, ranks, calibratedScores }];
    });
  }, [
    activeRefinementVisualization,
    activeRuntimeFusionResult,
    activeTunedCalibratedScores,
    activeTunedRanks,
    attributeTargets,
    dataset,
    methodIndex,
    rankMethod,
    refinementComponentIndex,
    retrievalTarget,
    targetIndex,
  ]);
  const selectionRuleSummary = useMemo(
    () => summarizeSelectionRules({
      selectionMask: resultMask,
      baseMask: analysisMask,
      attributes: selectionRuleAttributes,
    }),
    [analysisMask, resultMask, selectionRuleAttributes],
  );
  const selectionOverlapInput = useMemo(() => ({
    baseMask: analysisMask,
    conditions: selectionOverlapConditions,
    stages: selectionOverlapStages,
    finalMask: resultMask,
    ruleAttributes: selectionRuleAttributes,
  }), [
    analysisMask,
    resultMask,
    selectionOverlapConditions,
    selectionOverlapStages,
    selectionRuleAttributes,
  ]);
  const deferredSelectionOverlapInput = useDeferredValue(selectionOverlapInput);
  const deferredSelectionOverlapConditions = useMemo<SelectionOverlapConditionInput[]>(
    () => deferredSelectionOverlapInput.conditions.map((condition) => ({
      ...condition,
      ruleResult: summarizeSelectionRules({
        selectionMask: condition.mask,
        baseMask: deferredSelectionOverlapInput.baseMask,
        attributes: deferredSelectionOverlapInput.ruleAttributes,
      }),
    })),
    [deferredSelectionOverlapInput],
  );
  const selectionOverlapSummary = useMemo(
    () => summarizeSelectionOverlap({
      baseMask: deferredSelectionOverlapInput.baseMask,
      conditions: deferredSelectionOverlapConditions,
      stages: deferredSelectionOverlapInput.stages,
      finalMask: deferredSelectionOverlapInput.finalMask,
    }),
    [deferredSelectionOverlapConditions, deferredSelectionOverlapInput],
  );
  const selectionOverlapReady = deferredSelectionOverlapInput === selectionOverlapInput;
  const selectionOverlapActive = selectionOverlapReady && (
    deferredSelectionOverlapInput.conditions.length > 0
    || deferredSelectionOverlapInput.stages.length > 0
  );

  const projectionPoints = useMemo<ProjectionPoint[]>(() => {
    if (!dataset || !activeVisualAnalysis) return [];
    const pcaSource = activeVisualAnalysis.pca2d;
    const umapSource = activeVisualAnalysis.umap2d;
    const points = new Array<ProjectionPoint>(dataset.manifest.rowCount);
    for (let index = 0; index < dataset.manifest.rowCount; index += 1) {
      points[index] = {
        id: dataset.imageIds[index],
        rowIndex: index,
        cluster: visualClusterLabels[index],
        pca: [pcaSource[index * 2], pcaSource[index * 2 + 1]],
        umap: umapSource
          ? [umapSource[index * 2], umapSource[index * 2 + 1]]
          : null,
        label: fileLabel(dataset.imageIds[index]),
      };
    }
    return points;
  }, [activeVisualAnalysis, dataset, visualClusterLabels]);
  const scopedProjectionPoints = useMemo(
    () => filterAnalysisRows(
      projectionPoints,
      (point) => point.rowIndex,
      analysisMask,
    ) as readonly ProjectionPoint[],
    [analysisMask, projectionPoints],
  );

  const galleryItems = useMemo<GalleryItem[]>(() => {
    if (!dataset) return [];
    const thumbnailMetadata = dataset.metadata.thumbnails.available
      ? dataset.metadata.thumbnails
      : dataset.manifest.thumbnails;
    return dataset.imageIds.map((id, rowIndex) => ({
      id,
      rowIndex,
      label: fileLabel(id),
      cluster: activeVisualAnalysis ? visualClusterLabels[rowIndex] : undefined,
      fullImageSrc: `/api/tuning/images/${encodeURIComponent(selectedTaskId)}/${rowIndex}`,
      thumbnailAtlas: thumbnailFor(rowIndex, thumbnailMetadata, dataset.dataRoot),
    }));
  }, [activeVisualAnalysis, dataset, selectedTaskId, visualClusterLabels]);

  const bulkSelectedItems = useMemo(
    () => selectionRuleActive
      ? galleryItems.filter((item) => resultMask[item.rowIndex] === 1 && canAnnotate(item))
      : [],
    [canAnnotate, galleryItems, resultMask, selectionRuleActive],
  );
  const bulkPositiveChanges = useMemo(
    () => bulkSelectedItems.filter((item) => (tuning.annotations.get(item.id)?.label ?? 0) <= 0),
    [bulkSelectedItems, tuning.annotations],
  );
  const bulkNegativeChanges = useMemo(
    () => bulkSelectedItems.filter((item) => (tuning.annotations.get(item.id)?.label ?? 0) >= 0),
    [bulkSelectedItems, tuning.annotations],
  );

  const queryPreviewItems = useMemo<GalleryItem[]>(() => {
    if (!dataset?.manifest.query) return [];
    return dataset.manifest.query.images.map((image, index) => {
      const source = dataPath(dataset.dataRoot, image.path);
      return {
        id: image.imageId,
        rowIndex: image.imageIndex,
        label: `Query ${index + 1} · ${fileLabel(image.imageId)}`,
        imageSrc: source,
        fullImageSrc: source,
      };
    });
  }, [dataset]);

  const galleryItemById = useMemo(
    () => new Map(galleryItems.map((item) => [item.id, item])),
    [galleryItems],
  );
  const manualHighlights = useMemo(
    () => [...tuning.annotations.values()]
      .map((annotation) => ({
        rowIndex: annotation.rowIndex,
        label: annotation.label,
      }))
      .sort((left, right) => left.rowIndex - right.rowIndex),
    [tuning.annotations],
  );
  const focusedRowIndex = selectedId
    ? galleryItemById.get(selectedId)?.rowIndex ?? null
    : null;
  const scopedManualHighlights = useMemo(
    () => manualHighlightScope(manualHighlights, pcpEligibleMask),
    [manualHighlights, pcpEligibleMask],
  );
  const visibleManualHighlights = useMemo(
    () => showManualHighlights ? scopedManualHighlights.visible : [],
    [scopedManualHighlights.visible, showManualHighlights],
  );
  const hiddenManualHighlightCount = scopedManualHighlights.hiddenCount;
  const visibleFocusedRowIndex = showManualHighlights
    && focusedRowIndex !== null
    && Boolean(pcpEligibleMask[focusedRowIndex])
    ? focusedRowIndex
    : null;

  const activeFusionTuneRanking = activeFusionTuneResult && appliedFusionTune
    ? {
        id: `personal-tune:${appliedFusionTune.runId}`,
        label: appliedFusionTune.label,
      }
    : null;
  const activeTargetTunedRanking = activeTunedRanks && tuning.run?.status === "succeeded"
    ? {
        id: `personal-tune:${tuning.run.id}`,
        label: tuningRankingLabel(tuning.run.mode, tuning.run.algorithmVersion),
      }
    : null;
  const activeTunedRanking = activeFusionTuneRanking ?? activeTargetTunedRanking;

  const validationComparisonRows = useMemo<ValidationComparisonRow[]>(() => {
    if (!dataset || !validationMode) return [];
    const currentTargetIndex = targetIndex.get(retrievalTarget);
    if (currentTargetIndex === undefined) return [];
    const rows: ValidationComparisonRow[] = dataset.manifest.methods.map((method, index) => {
      const ranks = new Float32Array(dataset.manifest.rowCount);
      for (let rowIndex = 0; rowIndex < dataset.manifest.rowCount; rowIndex += 1) {
        ranks[rowIndex] = dataset.ranks[scoreOffset(
          rowIndex,
          index,
          currentTargetIndex,
          dataset.manifest.methodCount,
          dataset.manifest.targetCount,
        )];
      }
      return {
        id: method,
        label: methodDisplayLabel(method),
        baseline: method === "Ours-Full",
        current: isStaticValidationRankingCurrent(
          method,
          rankMethod,
          Boolean(activeHierarchicalFusionResult),
        ),
        metrics: evaluateRankedScope({
          ranks,
          groundTruth: dataset.groundTruth,
          scopeMask: dataset.validationMask,
          rowCount: dataset.manifest.rowCount,
          targetIndex: currentTargetIndex,
          targetCount: dataset.manifest.targetCount,
          k: 50,
        }),
      };
    });
    if (activeHierarchicalFusionResult) {
      const ranks = new Float32Array(dataset.manifest.rowCount);
      for (let rowIndex = 0; rowIndex < dataset.manifest.rowCount; rowIndex += 1) {
        ranks[rowIndex] = activeHierarchicalFusionResult.ranks[
          rowIndex * dataset.manifest.targetCount + currentTargetIndex
        ];
      }
      rows.push({
        id: HIERARCHICAL_FUSION_METHOD_ID,
        label: HIERARCHICAL_FUSION_METHOD_LABEL,
        detail: "Applied per-attribute 13-method and outer attribute rank weights",
        current: rankMethod === HIERARCHICAL_FUSION_METHOD_ID,
        metrics: evaluateRankedScope({
          ranks,
          groundTruth: dataset.groundTruth,
          scopeMask: dataset.validationMask,
          rowCount: dataset.manifest.rowCount,
          targetIndex: currentTargetIndex,
          targetCount: dataset.manifest.targetCount,
          k: 50,
        }),
      });
    }
    return rows;
  }, [
    activeHierarchicalFusionResult,
    dataset,
    rankMethod,
    retrievalTarget,
    targetIndex,
    validationMode,
  ]);

  const getRank = useCallback(
    (item: GalleryItem, learnerId: string) => {
      if (!dataset) return Number.NaN;
      if (activeRuntimeFusionResult) {
        const currentTarget = targetIndex.get(retrievalTarget);
        return currentTarget === undefined
          ? Number.NaN
          : activeRuntimeFusionResult.ranks[
              item.rowIndex * dataset.manifest.targetCount + currentTarget
            ];
      }
      if (activeTunedRanks) return activeTunedRanks[item.rowIndex];
      if (activeInitialBaseline && learnerId === "Ours-Full") {
        const column = initialBaselineTargetColumn(activeInitialBaseline.manifest, retrievalTarget);
        return column === null ? Number.NaN : activeInitialBaseline.snapshot.ranks[
          item.rowIndex * activeInitialBaseline.snapshot.componentCount + column
        ];
      }
      const column = methodIndex.get(learnerId);
      const currentTarget = targetIndex.get(retrievalTarget);
      return column === undefined || currentTarget === undefined
        ? Number.NaN
        : dataset.ranks[scoreOffset(
            item.rowIndex,
            column,
            currentTarget,
            dataset.manifest.methodCount,
            dataset.manifest.targetCount,
          )];
    },
    [
      activeTunedRanks,
      activeInitialBaseline,
      activeRuntimeFusionResult,
      dataset,
      methodIndex,
      retrievalTarget,
      targetIndex,
    ],
  );

  const getScoreReadout = useCallback(
    (item: GalleryItem, learnerId: string) => {
      if (!dataset) return null;
      const currentTarget = targetIndex.get(retrievalTarget);
      if (currentTarget === undefined) return null;
      if (
        activeTunedRanks
        && activeTunedScores
        && learnerId === rankMethod
      ) {
        return {
          raw: activeTunedScores[item.rowIndex],
          calibrated: activeTunedScores[item.rowIndex],
          rank: activeTunedRanks[item.rowIndex],
        };
      }
      if (activeRuntimeFusionResult) {
        const offset = item.rowIndex * dataset.manifest.targetCount + currentTarget;
        return {
          raw: activeRuntimeFusionResult.rawScores[offset],
          calibrated: activeRuntimeFusionResult.calibratedScores[offset],
          rank: activeRuntimeFusionResult.ranks[offset],
        };
      }
      if (activeInitialBaseline && learnerId === "Ours-Full") {
        const column = initialBaselineTargetColumn(activeInitialBaseline.manifest, retrievalTarget);
        if (column === null) return null;
        const offset = item.rowIndex * activeInitialBaseline.snapshot.componentCount + column;
        const score = activeInitialBaseline.snapshot.scores[offset];
        return { raw: score, calibrated: score, rank: activeInitialBaseline.snapshot.ranks[offset] };
      }
      const column = methodIndex.get(learnerId);
      if (column === undefined) return null;
      const offset = scoreOffset(
        item.rowIndex,
        column,
        currentTarget,
        dataset.manifest.methodCount,
        dataset.manifest.targetCount,
      );
      return {
        raw: dataset.rawScores[offset],
        calibrated: dataset.calibratedScores[offset],
        rank: dataset.ranks[offset],
      };
    },
    [
      activeTunedRanks,
      activeTunedScores,
      activeInitialBaseline,
      activeRuntimeFusionResult,
      dataset,
      methodIndex,
      rankMethod,
      retrievalTarget,
      targetIndex,
    ],
  );

  const getGroundTruth = useCallback(
    (item: GalleryItem) => {
      if (!dataset) return false;
      const currentTarget = targetIndex.get(retrievalTarget);
      return currentTarget !== undefined && isGroundTruthPositive(
        dataset.groundTruth,
        item.rowIndex,
        currentTarget,
        dataset.manifest.targetCount,
      );
    },
    [dataset, retrievalTarget, targetIndex],
  );

  const selectedItem = useMemo(() => {
    if (!selectedId) return null;
    const item = galleryItemById.get(selectedId);
    return item && finalMask[item.rowIndex] ? item : null;
  }, [finalMask, galleryItemById, selectedId]);

  const readoutItem = hoveredId
    ? (galleryItemById.get(hoveredId) ?? null)
    : selectedItem;

  const handleProjectionSelection = useCallback(
    (selection: ProjectionSelection | null) => {
      setSelectedId(null);
      setHoveredId(null);
      if (!dataset || !selection) {
        setProjectionSelection(null);
        return;
      }
      setProjectionSelection(selection);
    },
    [dataset],
  );

  const updatePreference = useCallback(async (item: GalleryItem, preference: PreferenceState) => {
    if (!canAnnotate(item)) return false;
    try {
      await tuning.updateAnnotation(item, preference, "top-gallery");
      return true;
    } catch {
      return false;
    }
  }, [canAnnotate, tuning]);

  const updateReviewedPreference = useCallback(async (
    item: GalleryItem,
    preference: PreferenceState,
  ) => {
    if (preference === "unmarked" ? !canRemoveAnnotation(item) : !canAnnotate(item)) return false;
    try {
      await tuning.updateAnnotation(item, preference, "annotation-review");
      return true;
    } catch {
      return false;
    }
  }, [canAnnotate, canRemoveAnnotation, tuning]);

  const confirmReviewedFailureAttributes = useCallback(async (
    item: GalleryItem,
    failedAttributeIds: readonly string[],
  ) => {
    if (!canAnnotate(item)) return false;
    try {
      await tuning.updateFailureAttributes(item, failedAttributeIds);
      return true;
    } catch {
      return false;
    }
  }, [canAnnotate, tuning]);

  const bulkMarkSelection = useCallback(async (label: "positive" | "negative") => {
    const items = label === "positive" ? bulkPositiveChanges : bulkNegativeChanges;
    if (items.length === 0) return;
    if (!items.every(canAnnotate)) return;
    await tuning.bulkUpdateAnnotations(
      items,
      label,
      `selection-bulk-${label}`,
    );
  }, [bulkNegativeChanges, bulkPositiveChanges, canAnnotate, tuning]);

  const clearWorkingPcpSelection = useCallback(() => {
    setBrushes({});
    setSelectedId(null);
    setHoveredId(null);
  }, []);

  const clearPcpSelection = useCallback(() => {
    clearWorkingPcpSelection();
    setPcpBrushZoom(resetPcpBrushZoom());
  }, [clearWorkingPcpSelection]);

  useEffect(() => {
    const previousSource = rankClusterInteractionSourceRef.current;
    rankClusterInteractionSourceRef.current = rankClusterSourceKey;
    if (previousSource === null || previousSource === rankClusterSourceKey) return;
    clearPcpSelection();
  }, [clearPcpSelection, rankClusterSourceKey]);

  const zoomCurrentPcpBrush = useCallback((currentBrushes: PcpBrushMap) => {
    const next = drillIntoPcpBrushZoom(
      pcpBrushZoom,
      currentBrushes,
      pcpAxisIds,
    );
    if (next === pcpBrushZoom) return;
    setPcpBrushZoom(next);
    clearWorkingPcpSelection();
  }, [clearWorkingPcpSelection, pcpAxisIds, pcpBrushZoom]);

  const backPcpBrushZoom = useCallback(() => {
    if (currentPcpZoomDepth === 0) return;
    setPcpBrushZoom((current) => stepBackPcpBrushZoom(current));
    clearWorkingPcpSelection();
  }, [clearWorkingPcpSelection, currentPcpZoomDepth]);

  const clearPcpBrushZoom = useCallback(() => {
    if (!pcpZoomActive && !hasActivePcpBrush) return;
    setPcpBrushZoom(resetPcpBrushZoom());
    clearWorkingPcpSelection();
  }, [clearWorkingPcpSelection, hasActivePcpBrush, pcpZoomActive]);

  const clearRankClusterInteraction = useCallback(() => {
    setRankClusterFilter("all");
    setClusterDrilldown(null);
    clearPcpSelection();
  }, [clearPcpSelection]);

  const prepareRuntimeRankClusters = useCallback((
    config: Readonly<HierarchicalPcpConfig>,
    scheme: ClusterScheme,
  ) => {
    if (!dataset) throw new Error("The task data is not ready.");
    return requestRuntimeRankProfileClusters({
      taskId: selectedTaskId,
      rowCount: dataset.manifest.rowCount,
      targetCount: dataset.manifest.targetCount,
      attributeIds: hierarchicalAttributeIds,
      config,
      scheme: scheme as RuntimeRankProfileClusterScheme,
    });
  }, [dataset, hierarchicalAttributeIds, selectedTaskId]);

  const changePcpFullExpanded = (expanded: boolean) => {
    setPcpFullExpanded(expanded);
    clearPcpSelection();
  };

  const changePcpAttributeExpanded = (attributeId: string, expanded: boolean) => {
    setExpandedPcpAttributes((current) => {
      const next = new Set(current);
      if (expanded) next.add(attributeId);
      else next.delete(attributeId);
      return next;
    });
    clearPcpSelection();
  };

  const changePcpHolisticExpanded = (expanded: boolean) => {
    setPcpHolisticExpanded(expanded);
    clearPcpSelection();
  };

  const changePcpAttributeEvidenceExpanded = (expanded: boolean) => {
    setPcpAttributeEvidenceExpanded(expanded);
    clearPcpSelection();
  };

  const discardAppliedFusionTune = () => {
    fusionTuneApplyGenerationRef.current += 1;
    setAppliedFusionTune(null);
    setFusionTuneApplyError(null);
    fusionTuneHierarchy.clear();
  };

  const applyTunedRanking = async () => {
    const run = tuning.run;
    const session = tuning.session;
    if (!run || !session || run.status !== "succeeded") return;

    if (!isHierarchicalRankFusionRun(run)) {
      // Residual and historical v1-v3 Fusion runs intentionally retain their
      // audited current-target artifact path.
      const replacedFullFusion = Boolean(appliedFusionTune);
      await tuning.applyRun({
        refinementClusterScheme: isWeightRefinementRun(run)
          ? rankClusterScheme
          : undefined,
      });
      discardAppliedFusionTune();
      if (replacedFullFusion) setHierarchicalDraft(defaultHierarchicalConfig);
      clearRankClusterInteraction();
      return;
    }

    const generation = ++fusionTuneApplyGenerationRef.current;
    rankClusterSchemeGenerationRef.current += 1;
    setFusionTuneApplyError(null);
    setRankClusterApplyError(null);
    try {
      const learnedConfig = hierarchicalPcpConfigFromTuningWeights(
        run.after!.modelSummary!.methodWeightsByAttribute!,
        hierarchicalAttributeIds,
        run.after!.modelSummary!.attributeWeights,
      );
      const result = await fusionTuneHierarchy.apply(learnedConfig);
      const activeScope = tuningApplyScopeRef.current;
      if (
        !result
        || generation !== fusionTuneApplyGenerationRef.current
        || activeScope.taskId !== selectedTaskId
        || activeScope.runId !== run.id
        || activeScope.sessionId !== session.id
      ) return;

      setRankClusterRequestBusy(true);
      await prepareRuntimeRankClusters(result.config, rankClusterScheme);
      const committedScope = tuningApplyScopeRef.current;
      if (
        generation !== fusionTuneApplyGenerationRef.current
        || committedScope.taskId !== selectedTaskId
        || committedScope.runId !== run.id
        || committedScope.sessionId !== session.id
      ) return;

      const configFingerprint = hierarchicalPcpConfigFingerprint(
        result.config,
        hierarchicalAttributeIds,
      );
      // Commit all consumers together only after the complete [row,target]
      // cube has been verified. No current-target tuned artifact is needed.
      tuning.revertRun();
      setHierarchicalDraft(result.config);
      setAppliedFusionTune({
        taskId: selectedTaskId,
        runId: run.id,
        sessionId: session.id,
        sourceTargetId: session.targetId,
        baseMethod: run.baseMethod,
        label: tuningRankingLabel(run.mode, run.algorithmVersion),
        configFingerprint,
      });
      setSmartFilterKind((current) => current === "prototype-rank-gap" ? null : current);
      clearRankClusterInteraction();
    } catch (reason) {
      if (generation !== fusionTuneApplyGenerationRef.current) return;
      const message = reason instanceof Error ? reason.message : String(reason);
      setFusionTuneApplyError(message);
      setRankClusterApplyError(message);
      throw reason;
    } finally {
      if (generation === fusionTuneApplyGenerationRef.current) {
        setRankClusterRequestBusy(false);
      }
    }
  };

  const revertTunedRanking = () => {
    fusionTuneApplyGenerationRef.current += 1;
    rankClusterSchemeGenerationRef.current += 1;
    tuning.revertRun();
    if (appliedFusionTune) {
      setAppliedFusionTune(null);
      fusionTuneHierarchy.clear();
      setHierarchicalDraft(defaultHierarchicalConfig);
    }
    setFusionTuneApplyError(null);
    setRankClusterApplyError(null);
    setRankClusterRequestBusy(false);
    clearRankClusterInteraction();
  };

  const applyHierarchicalWeights = async () => {
    const generation = ++hierarchicalApplyGenerationRef.current;
    rankClusterSchemeGenerationRef.current += 1;
    setRankClusterApplyError(null);
    try {
      const result = await hierarchicalFusion.apply(hierarchicalDraft);
      if (!result || generation !== hierarchicalApplyGenerationRef.current) return;
      setRankClusterRequestBusy(true);
      await prepareRuntimeRankClusters(result.config, rankClusterScheme);
      if (generation !== hierarchicalApplyGenerationRef.current) return;
      discardAppliedFusionTune();
      tuning.revertRun();
      setHierarchicalDraft(result.config);
      setRankMethod(HIERARCHICAL_FUSION_METHOD_ID);
      setSmartFilterKind((current) => current === "prototype-rank-gap" ? null : current);
      clearRankClusterInteraction();
    } catch (reason) {
      if (generation !== hierarchicalApplyGenerationRef.current) return;
      setRankClusterApplyError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (generation === hierarchicalApplyGenerationRef.current) {
        setRankClusterRequestBusy(false);
      }
    }
  };

  const resetHierarchicalWeights = () => {
    hierarchicalApplyGenerationRef.current += 1;
    rankClusterSchemeGenerationRef.current += 1;
    discardAppliedFusionTune();
    tuning.revertRun();
    setHierarchicalDraft(defaultHierarchicalConfig);
    hierarchicalFusion.clear();
    setRankMethod((current) => (
      current === HIERARCHICAL_FUSION_METHOD_ID ? "Ours-Full" : current
    ));
    setRankClusterApplyError(null);
    setRankClusterRequestBusy(false);
    clearRankClusterInteraction();
  };

  const changeRetrievalTarget = (target: RetrievalTarget) => {
    setRetrievalTarget(target);
    setSelectedId(null);
    setHoveredId(null);
  };

  const changePcpValueKind = (valueKind: PcpValueKind) => {
    setPcpValueKind(valueKind);
    clearPcpSelection();
  };

  const changePcpLineMode = (lineMode: PcpLineMode) => {
    setPcpLineMode(lineMode);
    clearPcpSelection();
  };

  const changeRankClusterScheme = async (scheme: ClusterScheme) => {
    if (scheme === rankClusterScheme) return;
    fusionTuneApplyGenerationRef.current += 1;
    hierarchicalApplyGenerationRef.current += 1;
    const generation = ++rankClusterSchemeGenerationRef.current;
    setRankClusterApplyError(null);
    clearRankClusterInteraction();
    if (!activeRefinementVisualization && !activeRuntimeFusionResult) {
      setRankClusterScheme(scheme);
      setRankClusterRequestBusy(false);
      return;
    }
    setRankClusterRequestBusy(true);
    try {
      if (activeInitialBaseline) {
        await initialBaseline.prepareScheme(scheme);
      } else if (activePersonalRefinementVisualization && tuning.run) {
        await requestRefinementClusters({
          run: tuning.run,
          snapshot: activePersonalRefinementVisualization,
          scheme,
        });
      } else if (activeRuntimeFusionResult) {
        await prepareRuntimeRankClusters(activeRuntimeFusionResult.config, scheme);
      }
      if (generation !== rankClusterSchemeGenerationRef.current) return;
      setRankClusterScheme(scheme);
    } catch (reason) {
      if (generation !== rankClusterSchemeGenerationRef.current) return;
      setRankClusterApplyError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (generation === rankClusterSchemeGenerationRef.current) {
        setRankClusterRequestBusy(false);
      }
    }
  };

  const changeVisualClusterScheme = (scheme: VisualClusterScheme) => {
    setVisualClusterScheme(scheme);
    setVisualClusterFilter("all");
    setProjectionSelection(null);
    setSelectedId(null);
    setHoveredId(null);
  };

  const changeDataset = (datasetId: string) => {
    const nextDataset = catalog?.datasets.find((entry) => entry.id === datasetId);
    const nextTask = nextDataset?.tasks[0];
    if (!nextDataset || !nextTask) return;
    hierarchicalApplyGenerationRef.current += 1;
    rankClusterSchemeGenerationRef.current += 1;
    discardAppliedFusionTune();
    setRankClusterRequestBusy(false);
    setRankClusterApplyError(null);
    setDataset(null);
    setLoadError(undefined);
    setQueryPreviewIndex(null);
    setSelectedDatasetId(nextDataset.id);
    setSelectedTaskId(nextTask.id);
  };

  const changeTask = (taskId: string) => {
    if (!selectedDatasetEntry?.tasks.some((entry) => entry.id === taskId)) return;
    hierarchicalApplyGenerationRef.current += 1;
    rankClusterSchemeGenerationRef.current += 1;
    discardAppliedFusionTune();
    setRankClusterRequestBusy(false);
    setRankClusterApplyError(null);
    setDataset(null);
    setLoadError(undefined);
    setQueryPreviewIndex(null);
    setSelectedTaskId(taskId);
  };

  const reloadCurrentTask = () => {
    if (!selectedTaskEntry) return;
    hierarchicalApplyGenerationRef.current += 1;
    rankClusterSchemeGenerationRef.current += 1;
    discardAppliedFusionTune();
    setRankClusterRequestBusy(false);
    setRankClusterApplyError(null);
    setDataset(null);
    setLoadError(undefined);
    setQueryPreviewIndex(null);
    setReloadNonce((current) => current + 1);
  };

  const toggleRankClusterSummary = (cluster: ClusterSummaryId) => {
    const clusterId = String(cluster);
    setRankClusterFilter((current) => current === clusterId ? "all" : clusterId);
    clearPcpSelection();
  };

  const selectRankClusterSummary = useCallback((cluster: ClusterSummaryId) => {
    setRankClusterFilter(String(cluster));
    clearPcpSelection();
  }, [clearPcpSelection]);

  const toggleVisualClusterSummary = (cluster: ClusterSummaryId) => {
    const clusterId = String(cluster);
    setVisualClusterFilter((current) => current === clusterId ? "all" : clusterId);
    setSelectedId(null);
    setHoveredId(null);
  };

  const openClusterDrilldown = useCallback((cluster: ClusterSummaryId) => {
    const clusterId = String(cluster);
    setRankClusterFilter(clusterId);
    setClusterDrilldown({
      scheme: rankClusterScheme,
      scope: resultScope,
      clusterId,
    });
    setPcpLineMode("samples");
    clearPcpSelection();
  }, [clearPcpSelection, rankClusterScheme, resultScope]);

  const closeClusterDrilldown = useCallback(() => {
    setClusterDrilldown(null);
    setRankClusterFilter("all");
    setPcpLineMode("clusters");
    clearPcpSelection();
  }, [clearPcpSelection]);

  const clearFilters = () => {
    setGeneratedOnly(false);
    setRankClusterFilter("all");
    setVisualClusterFilter("all");
    setClusterDrilldown(null);
    setSmartFilterKind(null);
    setProjectionSelection(null);
    clearPcpSelection();
  };

  const changeResultScope = (scope: ResultScope) => {
    // The historical Web Validation page remains compatibility-only.
    if (scope !== "development" && scope !== "test") return;
    setResultScope(scope);
    setRankClusterFilter("all");
    setVisualClusterFilter("all");
    setClusterDrilldown(null);
    setSmartFilterKind(null);
    setProjectionSelection(null);
    clearPcpSelection();
  };

  useLayoutEffect(() => {
    if (pcpFrameRef.current) pcpFrameRef.current.scrollTop = 0;
  }, [pcpLineMode, pcpValueKind, rankClusterScheme, selectedTaskId]);

  if (loadError) return (
    <LoadingScreen
      error={loadError}
      onRetry={() => {
        setLoadError(undefined);
        setReloadNonce((current) => current + 1);
      }}
    />
  );
  if (!dataset) return <LoadingScreen estimate={currentLoadEstimate} />;

  const task = dataset.metadata.task;
  const activeBrushCount = Object.keys(brushes).length;
  const retrievalTargetLabel = targetLabels.get(retrievalTarget) ?? retrievalTarget;
  const rankMethodLabel = rankMethod === HIERARCHICAL_FUSION_METHOD_ID
    ? HIERARCHICAL_FUSION_METHOD_LABEL
    : activeInitialBaseline ? "F₀ · Ours-Full"
      : initialBaseline.data || legacyBaselineSelected
        ? `Legacy ${methodDisplayLabel(rankMethod)}` : methodDisplayLabel(rankMethod);
  const refinementSourceLabel = activeInitialBaseline ? "F₀ · fixed initial weights"
    : tuningRankingLabel(tuning.run?.mode ?? "weight_joint", tuning.run?.algorithmVersion);
  const attributeStrengthSourceLabel = activeRefinementVisualization
    ? `${refinementSourceLabel} · attribute gates`
    : activeFusionTuneResult && activeTunedRanking
    ? `${activeTunedRanking.label}; all attributes recomputed`
    : activeTunedRanking
      ? `${activeTunedRanking.label}; other attributes use ${rankMethodLabel}`
    : `${rankMethodLabel} per-attribute calibrated strength`;
  const resultScopeDefinition = dataset.manifest.evaluation.scopes.find(
    (scope) => scope.id === resultScope,
  ) ?? dataset.manifest.evaluation.scopes[0];
  const headerActivity = scopeActivitySummary(
    resultScope,
    dataset.manifest.evaluation.validation.rowCount,
    resultCandidateCount,
  );
  const validationPositiveCount = dataset.manifest.evaluation.validation.positiveCounts[
    retrievalTarget
  ] ?? 0;
  const testPositiveCount = dataset.manifest.evaluation.frozenTest.positiveCounts[
    retrievalTarget
  ] ?? 0;
  const hasResultFilters = rankClusterFilter !== "all"
    || visualClusterFilter !== "all"
    || hasPcpRestriction
    || projectionMask !== null
    || smartFilterKind !== null;
  const rankClusterTargetLabel = targetLabels.get(dataset.manifest.clusters.basisTarget)
    ?? dataset.manifest.clusters.basisTarget;
  const rankClusterSchemeOptions = dataset.manifest.clusters.schemes;
  const visualClusterSchemeOptions = dataset.manifest.visualEmbedding.clusterings;
  const readoutScore = readoutItem ? getScoreReadout(readoutItem, rankMethod) : null;
  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">P</span>
          <div>
            <p className="eyebrow">HUMAN-IN-THE-LOOP RANK ANALYSIS</p>
            <h1>ProbeScout: Interactive Retrieval Refinement</h1>
          </div>
        </div>
        <div className="dataset-task-switcher" aria-label="Dataset and task selection">
          <label>
            <span>Dataset</span>
            <select
              aria-label="Dataset"
              value={selectedDatasetId}
              onChange={(event) => changeDataset(event.target.value)}
            >
              {catalog?.datasets.map((entry) => (
                <option key={entry.id} value={entry.id}>{entry.label}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Task</span>
            <select
              aria-label="Task"
              value={selectedTaskId}
              onChange={(event) => changeTask(event.target.value)}
            >
              {selectedDatasetEntry?.tasks.map((entry) => (
                <option key={entry.id} value={entry.id}>{entry.label}</option>
              ))}
            </select>
          </label>
          <label className="data-purpose-switcher">
            <span>Data purpose</span>
            <select
              aria-label="Data purpose"
              value={resultScope}
              onChange={(event) => changeResultScope(event.target.value as ResultScope)}
            >
              {dataset.manifest.evaluation.scopes.filter((scope) => scope.id !== "validation").map((scope) => (
                <option key={scope.id} value={scope.id}>
                  {scope.label} ({scope.id === "development"
                    ? (visibleDevelopmentCount?.toLocaleString() ?? "loading")
                    : scope.rowCount.toLocaleString()})
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="dataset-meta">
          <span>{selectedDatasetEntry?.label ?? task.dataset.toUpperCase()}</span>
          <span>{dataset.manifest.rowCount.toLocaleString()} images</span>
          <span>Target: {retrievalTargetLabel}</span>
          <span>
            PCP: {activeRefinementVisualization
              ? `${refinementSourceLabel} · Development-fitted`
              : activeRuntimeFusionResult
                ? "full hierarchy · Development-fitted"
                : `${rankClusterTargetLabel} rank profile`}
          </span>
          <span>Visual: SigLIP embedding</span>
          <strong>{headerActivity.count.toLocaleString()} {headerActivity.label}</strong>
        </div>
      </header>

      <div className="workspace-controls">
        <UserSessionControls state={tuning} targetLabel={retrievalTargetLabel} baseMethod={rankMethod} />
        <details className="advanced-settings">
          <summary>Advanced Settings</summary>
          <div className="advanced-settings-body">
            <button type="button" className="button-secondary" onClick={reloadCurrentTask}>Reload</button>
            {!validationMode && (
              <div className="projection-controls toolbar-controls">
                <label>
                  <span>Projection</span>
                  <select
                    aria-label="Projection"
                    value={projection}
                    onChange={(event) => {
                      setProjection(event.target.value as ProjectionKind);
                      setProjectionSelection(null);
                    }}
                  >
                    <option value="pca">PCA</option>
                    <option value="umap">UMAP</option>
                  </select>
                </label>
                <label>
                  <span>Embedding clusters</span>
                  <select
                    aria-label="Embedding cluster scheme"
                    value={visualClusterScheme}
                    onChange={(event) => changeVisualClusterScheme(
                      event.target.value as VisualClusterScheme,
                    )}
                  >
                    {visualClusterSchemeOptions.map((scheme) => (
                      <option key={scheme.id} value={scheme.id}>
                        {scheme.label} · K{scheme.clusters}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
            {developmentMode && !hierarchicalFusionSelected && !legacyBaselineSelected && (
              <ProbeSourceControls
                state={tuning}
                taskId={selectedTaskId}
                targetId={retrievalTarget}
                baseMethod={rankMethod}
                validationReady={fixedValidation.status === "ready"}
              />
            )}
          </div>
        </details>
      </div>

      <div className={`scope-purpose-banner scope-purpose-banner--${resultScope}`} role="note">
        <strong>{resultScopeDefinition.label}</strong>
        <span>
          {developmentMode
            ? "Explore results, inspect selections, annotate images and tune models."
            : validationMode
              ? "Compare methods and applied Fusion results with aggregate metrics."
              : "View Top results, PCP, clusters and test metrics."}
        </span>
      </div>

      {developmentMode && fixedValidation.status !== "ready" && (
        <div className="tuning-status" role={fixedValidation.error ? "alert" : "status"}>
          {fixedValidation.error
            ? `Val membership unavailable: ${fixedValidation.error} Reload to retry; exploration and feedback are locked.`
            : "Loading fixed Val membership; exploration and feedback are locked until validation completes."}
        </div>
      )}

      {validationMode ? (
        <div className="validation-workspace">
          <section className="workspace-panel validation-ranking-panel" aria-labelledby="validation-heading">
            <div className="panel-heading">
              <div>
                <p className="panel-index">01 · MODEL SELECTION</p>
                <h2 id="validation-heading">Web Validation comparison</h2>
              </div>
            </div>
            <div className="validation-target-control">
              <label>
                <span>Retrieval target</span>
                <select
                  aria-label="Retrieval target"
                  value={retrievalTarget}
                  onChange={(event) => changeRetrievalTarget(event.target.value as RetrievalTarget)}
                >
                  {dataset.manifest.retrievalTargets.map((target) => (
                    <option key={target.id} value={target.id}>{target.label}</option>
                  ))}
                </select>
              </label>
            </div>
            <ValidationComparison
              targetLabel={retrievalTargetLabel}
              rows={validationComparisonRows}
              validationRowCount={dataset.manifest.evaluation.validation.rowCount}
              validationPositiveCount={validationPositiveCount}
            />
          </section>
        </div>
      ) : (
      <div className="workspace-width-reference">
      <div className={`workspace-grid ${testMode ? "workspace-grid--test" : ""}`.trim()}>
        <section className="workspace-panel dashboard-panel" aria-labelledby="dashboard-heading">
          <div className="panel-heading">
            <div>
              <h2 id="dashboard-heading">Evidence Analysis</h2>
            </div>
            <span className="panel-kicker">{pcpAxes.length} visible PCP rows</span>
          </div>

          <div className="learner-toolbar">
            <div className="section-label"><span>Hierarchical Evidence</span><i /></div>
            <div className="toolbar-controls compact">
              <label>
                <span>Value</span>
                <select aria-label="PCP value" value={pcpValueKind} onChange={(event) => changePcpValueKind(event.target.value as PcpValueKind)}>
                  <option value="rank">Rank</option>
                  <option value="calibrated">{activeRefinementVisualization ? "Model score" : "Calibrated score"}</option>
                </select>
              </label>
              <label>
                <span>Lines</span>
                <select aria-label="PCP line mode" value={pcpLineMode} onChange={(event) => changePcpLineMode(event.target.value as PcpLineMode)}>
                  <option value="samples">Image samples</option>
                  <option value="clusters">Cluster summaries</option>
                </select>
              </label>
              <label>
                <span className="rank-profile-clusters-label">Rank-profile clusters</span>
                <select
                  aria-label="PCP cluster scheme"
                  value={rankClusterScheme}
                  disabled={rankClusterRequestBusy}
                  onChange={(event) => void changeRankClusterScheme(
                    event.target.value as ClusterScheme,
                  )}
                >
                  {rankClusterSchemeOptions.map((scheme) => (
                    <option key={scheme.id} value={scheme.id}>
                      {scheme.label} · K{scheme.clusters}
                    </option>
                  ))}
                </select>
              </label>
              {pcpLineMode === "samples" && (
                <label>
                  <span>Color by</span>
                  <select aria-label="PCP color mode" value={colorMode} onChange={(event) => setColorMode(event.target.value as PcpColorMode)}>
                    <option value="uniform">Uniform</option>
                    <option value="cluster">Cluster</option>
                    <option value="learner">Learner value</option>
                  </select>
                </label>
              )}
              {pcpLineMode === "samples" && colorMode === "learner" && (
                <label>
                  <span>Color row</span>
                  <select aria-label="PCP color row" value={pcpColorAxis} onChange={(event) => setColorLearner(event.target.value)}>
                    {pcpAxes.map((axis) => (
                      <option key={axis.id} value={axis.id}>
                        {pcpAxisLabels[axis.id]}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="hierarchical-weight-actions">
                <button
                  type="button"
                  className="button-primary"
                  disabled={
                    hierarchicalWeightsReadOnly
                    || hierarchicalFusion.status === "loading"
                    || rankClusterRequestBusy
                    || hierarchicalAttributeIds.length === 0
                  }
                  onClick={() => void applyHierarchicalWeights()}
                >
                  {hierarchicalFusion.status === "loading" || rankClusterRequestBusy
                    ? "Applying..."
                    : "Apply weights"}
                </button>
                <button
                  type="button"
                  className="button-secondary"
                  disabled={
                    testMode
                    || hierarchicalFusion.status === "loading"
                    || rankClusterRequestBusy
                  }
                  onClick={resetHierarchicalWeights}
                >
                  Reset weights
                </button>
                <button
                  type="button"
                  className={showManualHighlights ? "button-primary" : "button-secondary"}
                  aria-pressed={showManualHighlights}
                  disabled={manualHighlights.length === 0 && focusedRowIndex === null}
                  onClick={() => setShowManualHighlights((current) => !current)}
                >
                  {showManualHighlights ? "Hide selected" : "Highlight selected"}
                </button>
              </div>
            </div>
            {showManualHighlights && manualHighlights.length > 0 && (
              <p className="loaded-note manual-highlight-note" aria-live="polite">
                {scopedManualHighlights.visible.length > 0
                  ? `${scopedManualHighlights.visible.length.toLocaleString()} selected in current PCP`
                  : "No selected images in current PCP"}
                {hiddenManualHighlightCount > 0
                  ? ` · ${hiddenManualHighlightCount.toLocaleString()} outside current range`
                  : ""}
              </p>
            )}
            {pcpValueKind === "calibrated" && (
              <p className="loaded-note calibration-note">
                {activeRefinementVisualization
                  ? "Model scores: F / C / g / H; leaf scores are normalized."
                  : "Train min–max [0,1] per method × target · not probability."}
              </p>
            )}
            {activeRuntimeFusionResult && (
              <p className="loaded-note calibration-note">
                Rank-profile clusters use the complete hierarchy rank profile fitted on Development.
              </p>
            )}
            {!activeRefinementVisualization && !activeRuntimeFusionResult && (
              <p className="loaded-note calibration-note">
                {initialBaseline.status === "loading" ? "F₀ loading · showing Legacy Ours-Full."
                  : initialBaseline.status === "pending" ? "F₀ pending publication · showing Legacy Ours-Full."
                    : initialBaseline.status === "error" ? "F₀ unavailable · showing Legacy Ours-Full."
                      : "Legacy Ours-Full."}
              </p>
            )}
            {initialBaseline.error && !activeRefinementVisualization && (
              <p className="loaded-note inline-error" role="alert">{initialBaseline.error}</p>
            )}
            {testMode && (
              <p className="loaded-note calibration-note">
                Frozen Test shows the applied hierarchy; edit weights in Development.
              </p>
            )}
            {hierarchicalFusion.error && (
              <p className="loaded-note inline-error" role="alert">{hierarchicalFusion.error}</p>
            )}
            {(rankClusterApplyError
              ?? refinementRankClusters.error
              ?? runtimeRankClusters.error) && (
              <p className="loaded-note inline-error" role="alert">
                {rankClusterApplyError
                  ?? refinementRankClusters.error
                  ?? runtimeRankClusters.error}
              </p>
            )}
          </div>

          {clusterDrilldown && (
            <div className="cluster-drilldown-breadcrumb" role="status">
              <span>
                Cluster {clusterDrilldown.clusterId} · {resultScopeDefinition.label} · member-level PCP
              </span>
              <button type="button" className="text-button" onClick={closeClusterDrilldown}>
                Back to cluster summaries
              </button>
            </div>
          )}
          <div ref={pcpFrameRef}
            style={{ "--pcp-frame-height": `${pcpPanel.frameHeight}px` } as CSSProperties}
            className={`pcp-frame pcp-frame--hierarchical pcp-frame--${pcpLineMode}${activeRefinementVisualization ? " pcp-frame--evidence" : ""}`}>
            {(rankClusterRequestBusy || (
              activeRuntimeFusionResult && runtimeRankClusters.status === "loading"
            ) || (
              activeRefinementVisualization && refinementRankClusters.status === "loading"
            )) && (
              <div className="projection-loading" role="status">
                Recomputing Development-fitted rank-profile clusters...
              </div>
            )}
            {pcpLineMode === "samples" ? (
              <>
                <HierarchicalPcpRail
                  axes={pcpAxes}
                  config={displayedHierarchicalConfig}
                  fullExpanded={pcpFullExpanded}
                  attributeEvidenceExpanded={pcpAttributeEvidenceExpanded}
                  holisticExpanded={pcpHolisticExpanded}
                  expandedAttributeIds={expandedPcpAttributes}
                  height={pcpHeight}
                  disabled={hierarchicalFusion.status === "loading"}
                  weightsReadOnly={hierarchicalWeightsReadOnly}
                  refinementWeights={activeRefinementWeights}
                  onFullExpandedChange={changePcpFullExpanded}
                  onAttributeEvidenceExpandedChange={changePcpAttributeEvidenceExpanded}
                  onHolisticExpandedChange={changePcpHolisticExpanded}
                  onAttributeExpandedChange={changePcpAttributeExpanded}
                  onAttributeWeightChange={(attributeId, weight) => {
                    setHierarchicalDraft((current) => (
                      updateHierarchicalAttributeWeight(current, attributeId, weight)
                    ));
                  }}
                  onMethodWeightChange={(attributeId, methodId, weight) => {
                    setHierarchicalDraft((current) => (
                      updateHierarchicalMethodWeight(current, attributeId, methodId, weight)
                    ));
                  }}
                />
                <ParallelCoordinates
                  methods={pcpAxisIds}
                  axisLabels={pcpAxisLabels}
                  ranks={pcpValues}
                  labels={rankClustersReady ? rankClusterLabels : undefined}
                  enabledMethods={enabledAxisIds}
                  manualHighlights={visibleManualHighlights}
                  focusedRowIndex={visibleFocusedRowIndex}
                  colorMode={!rankClustersReady && colorMode === "cluster" ? "uniform" : colorMode}
                  clusterColors={resolveRankClusterColor}
                  valueLabel={pcpValueKind === "rank" ? "rank" : activeRefinementVisualization ? "model score" : "calibrated score"}
                  colorLearner={pcpColorAxis}
                  candidateMask={pcpCandidateMask}
                  brushes={brushes}
                  axisDomains={pcpBrushZoom.ranges}
                  brushZoomDepth={currentPcpZoomDepth}
                  onBrushChange={(next) => {
                    setBrushes({ ...next });
                    setSelectedId(null);
                    setHoveredId(null);
                  }}
                  onBrushZoom={zoomCurrentPcpBrush}
                  onBrushZoomBack={backPcpBrushZoom}
                  onBrushZoomReset={clearPcpBrushZoom}
                  height={pcpHeight}
                  showAxisLabels={false}
                  maxBackgroundLines={12_000}
                  maxForegroundLines={16_000}
                  ariaLabel={`Hierarchical image ${pcpValueKind} profiles; drag any visible row to brush an interval`}
                />
              </>
            ) : (
              <ClusterSummaryParallelCoordinates
                  className="cluster-summary-pcp"
                  methods={pcpAxisIds}
                  axisLabels={pcpAxisLabels}
                  values={pcpValues}
                  labels={rankClusterLabels}
                  enabledMethods={enabledAxisIds}
                  candidateMask={analysisMask}
                  manualHighlights={visibleManualHighlights}
                  focusedRowIndex={visibleFocusedRowIndex}
                  activeClusters={rankClusterSummaries.map((summary) => summary.cluster_id)}
                  clusterColors={resolveRankClusterColor}
                  clusterLabels={Object.fromEntries(
                    rankClusterSummaries.map((summary) => [String(summary.cluster_id), summary.label]),
                  )}
                  selectedCluster={rankClusterFilter === "all" ? null : rankClusterFilter}
                  onClusterClick={toggleRankClusterSummary}
                  onClusterSelect={selectRankClusterSummary}
                  onClusterDoubleClick={openClusterDrilldown}
                  valueLabel={pcpValueKind === "rank" ? "rank" : activeRefinementVisualization ? "model score" : "calibrated score"}
                  title="Hierarchical Evidence"
                  height={summaryPcpHeight}
                  lineWidthRange={isFineClusterScheme(rankClusterScheme) ? [0.9, 4] : [2.5, 11]}
                  showAxisLabels={false}
                  axisRail={(
                    <HierarchicalPcpRail
                      axes={pcpAxes}
                      config={displayedHierarchicalConfig}
                      fullExpanded={pcpFullExpanded}
                      attributeEvidenceExpanded={pcpAttributeEvidenceExpanded}
                      holisticExpanded={pcpHolisticExpanded}
                      expandedAttributeIds={expandedPcpAttributes}
                      height={summaryPcpHeight}
                      disabled={hierarchicalFusion.status === "loading"}
                      weightsReadOnly={hierarchicalWeightsReadOnly}
                      refinementWeights={activeRefinementWeights}
                      onFullExpandedChange={changePcpFullExpanded}
                      onAttributeEvidenceExpandedChange={changePcpAttributeEvidenceExpanded}
                      onHolisticExpandedChange={changePcpHolisticExpanded}
                      onAttributeExpandedChange={changePcpAttributeExpanded}
                      onAttributeWeightChange={(attributeId, weight) => {
                        setHierarchicalDraft((current) => (
                          updateHierarchicalAttributeWeight(current, attributeId, weight)
                        ));
                      }}
                      onMethodWeightChange={(attributeId, methodId, weight) => {
                        setHierarchicalDraft((current) => (
                          updateHierarchicalMethodWeight(current, attributeId, methodId, weight)
                        ));
                      }}
                    />
                  )}
                  ariaLabel={`${rankClusterSummaries.length} rank-profile cluster centroid profiles; line width encodes cluster size`}
                />
            )}
          </div>
        </section>

        <section className="workspace-panel visualization-panel" aria-labelledby="visualization-heading">
          <div ref={explorationContentRef} className="exploration-content">
          <div className="panel-heading">
            <div>
              <h2 id="visualization-heading">Image Exploration</h2>
            </div>
            <button type="button" className="text-button" onClick={clearFilters}>Reset filters</button>
          </div>

          <div className="query-block">
            <div className="section-label"><span>Query Specification</span><i /></div>
            {isRobustCatTask && (
              <div className="robust-task-note">
                <span>Synthetic robustness pilot · 8 DG / 4 reserved Val{testMode ? " / 6 Test" : ""} · VQA not run</span>
                <label><input type="checkbox" checked={generatedOnly} onChange={(event) => setGeneratedOnly(event.target.checked)} /> Generated images only (current scope)</label>
                <small>Use the existing Human Feedback panel. Tune Val remains the original 44 VQA images.</small>
              </div>
            )}
            <div className="query-composer">
              {dataset.manifest.query ? (
                <>
                  <div className="query-images" aria-label={`${dataset.manifest.query.images.length} fixed query images`}>
                    {dataset.manifest.query.images.map((image, index) => (
                      <figure key={`${image.imageId}-${index}`} className="fixed-query-card">
                        <button
                          type="button"
                          className="fixed-query-image-button"
                          aria-label={`Open enlarged Query ${index + 1}: ${fileLabel(image.imageId)}`}
                          aria-haspopup="dialog"
                          aria-expanded={queryPreviewIndex === index}
                          onClick={() => setQueryPreviewIndex(index)}
                        >
                          {/* Query assets are exported alongside the active task bundle. */}
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={dataPath(dataset.dataRoot, image.path)}
                            alt={`Query ${index + 1}: ${fileLabel(image.imageId)}`}
                          />
                          <span className="fixed-query-zoom-hint" aria-hidden="true">Expand</span>
                        </button>
                        <figcaption>{String(index + 1).padStart(2, "0")}</figcaption>
                      </figure>
                    ))}
                  </div>
                </>
              ) : (
                <div className="query-empty">
                  <strong>No fixed query configured</strong>
                  <span>Add query images to this task&apos;s exported manifest.</span>
                </div>
              )}
            </div>
            {queryPreviewIndex !== null && queryPreviewItems.length > 0 && (
              <GalleryLightbox
                items={queryPreviewItems}
                activeIndex={queryPreviewIndex}
                onActiveIndexChange={setQueryPreviewIndex}
                onClose={() => setQueryPreviewIndex(null)}
                navigationHint="Esc closes · ←/→ moves through the fixed Query images"
              />
            )}
          </div>

          <div className="projection-frame">
            <ProjectionScatter
              showHeading={false}
              points={scopedProjectionPoints}
              projection={projection}
              candidateMask={projectionEligibleMask}
              boxSelectionMask={analysisMask}
              highlightMask={activePcpHighlightMask}
              boxSelectionActive={projectionMask !== null}
              manualHighlights={visibleManualHighlights}
              selectedId={visibleFocusedRowIndex === null ? null : selectedId}
              clusterColors={resolveVisualClusterColor}
              height={250}
              title="Visual Embedding Exploration"
              onPointHover={(point) => setHoveredId(point?.id ?? null)}
              onPointClick={(point) => setSelectedId(point?.id ?? null)}
              onBoxSelect={handleProjectionSelection}
            />
            {visualAnalysis.status === "loading" && (
              <div className="projection-loading" role="status">
                Loading precomputed visual embedding…
              </div>
            )}
            {visualAnalysis.status === "error" && (
              <div className="projection-loading projection-loading--error" role="alert">
                {visualAnalysis.error?.message ?? "Visual embedding analysis failed."}
              </div>
            )}
            <details key={selectedTaskId} className="cluster-disclosure visual-cluster-disclosure">
              <summary>
                <span>Visual clusters</span>
                <small>{visualClusterSummaries.length} clusters{visualClusterFilter !== "all"
                  ? ` · ${visualClusterSummaries.find((summary) => String(summary.cluster_id) === visualClusterFilter)?.label_zh ?? visualClusterFilter}`
                  : ""}</small>
              </summary>
            <div className="cluster-legend cluster-legend--dense" role="group" aria-label="Visual cluster legend" tabIndex={0}>
              {visualClusterSummaries.map((summary) => {
                const clusterKey = String(summary.cluster_id);
                const scopedSize = analysisVisualClusterCounts.counts.get(clusterKey) ?? 0;
                const pcpSelectedSize = pcpSelectedVisualClusterCounts.get(clusterKey) ?? 0;
                const scopedFraction = analysisVisualClusterCounts.total > 0
                  ? scopedSize / analysisVisualClusterCounts.total
                  : 0;
                return (
                <button
                  key={summary.cluster_id}
                  type="button"
                  className={visualClusterFilter === clusterKey ? "legend-item active" : "legend-item"}
                  title={`${summary.label_zh} · ${activePcpHighlightMask ? `PCP ${pcpSelectedSize.toLocaleString()}/` : ""}${scopedSize.toLocaleString()} · ${(scopedFraction * 100).toFixed(1)}%`}
                  disabled={scopedSize === 0}
                  aria-pressed={visualClusterFilter === clusterKey}
                  onClick={() => toggleVisualClusterSummary(summary.cluster_id)}
                >
                  <i style={{ background: resolveVisualClusterColor(summary.cluster_id) }} />
                  <span>{summary.label_zh}</span>
                  <small>
                    {activePcpHighlightMask
                      ? `PCP ${pcpSelectedSize.toLocaleString()}/${scopedSize.toLocaleString()} · `
                      : `${scopedSize.toLocaleString()} · `}
                    {(scopedFraction * 100).toFixed(1)}%
                  </small>
                </button>
                );
              })}
            </div>
            </details>
          </div>

          <div className="result-controls">
            <label>
              <span>Retrieval target</span>
              <select
                aria-label="Retrieval target"
                value={retrievalTarget}
                onChange={(event) => changeRetrievalTarget(event.target.value as RetrievalTarget)}
              >
                {dataset.manifest.retrievalTargets.map((target) => (
                  <option key={target.id} value={target.id}>{target.label}</option>
                ))}
              </select>
            </label>
            <label className="top-limit-control">
              <span>Results</span>
              <select
                aria-label="Number of top-ranked results"
                value={topLimit}
                onChange={(event) => setTopLimit(Number(event.target.value) as TopLimit)}
              >
                <option value={30}>Top 30</option>
                <option value={50}>Top 50</option>
                <option value={100}>Top 100</option>
                <option value={200}>Top 200</option>
              </select>
            </label>
            <label className="rank-method-control">
              <span>Ranked by</span>
              <select
                aria-label="Ranked by"
                value={activeTunedRanking?.id ?? (legacyBaselineSelected && rankMethod === "Ours-Full" ? LEGACY_OURS_FULL_OPTION : rankMethod)}
                onChange={(event) => {
                  const selectedMethod = event.target.value;
                  const method = selectedMethod === LEGACY_OURS_FULL_OPTION ? "Ours-Full" : selectedMethod;
                  if (activeTunedRanking && method === activeTunedRanking.id) return;
                  if (activeTunedRanking) revertTunedRanking();
                  clearRankClusterInteraction();
                  setLegacyBaselineSelected(selectedMethod === LEGACY_OURS_FULL_OPTION);
                  setRankMethod(method);
                  if (
                    (
                      method === "Image Prototype"
                      || method === HIERARCHICAL_FUSION_METHOD_ID
                    )
                    && smartFilterKind === "prototype-rank-gap"
                  ) {
                    setSmartFilterKind(null);
                  }
                }}
              >
                {activeTunedRanking && (
                  <option value={activeTunedRanking.id}>{activeTunedRanking.label}</option>
                )}
                {initialBaseline.data && (
                  <option value={LEGACY_OURS_FULL_OPTION}>Legacy Ours-Full</option>
                )}
                {displayMethods.map((method) => (
                  <option key={method} value={method}>
                    {method === HIERARCHICAL_FUSION_METHOD_ID
                      ? HIERARCHICAL_FUSION_METHOD_LABEL
                      : initialBaseline.data
                        ? method === "Ours-Full" ? "F₀ · Ours-Full" : `Legacy ${methodDisplayLabel(method)}`
                        : methodDisplayLabel(method)}
                  </option>
                ))}
              </select>
            </label>
            <div className="filter-state">
              <strong>{resultCandidateCount.toLocaleString()}</strong>
              <span>{resultScopeDefinition.label} candidates</span>
              {activeBrushCount > 0 && <em>{activeBrushCount} brushes</em>}
              {currentPcpZoomDepth > 0 && <em>PCP zoom {currentPcpZoomDepth}</em>}
              {smartFilterKind && <em>diagnostic Top 50</em>}
            </div>
          </div>

          {readoutItem && (
            <div className="selection-readout">
              <span>{hoveredId ? "Hover" : "Selected"}</span>
              <strong>{fileLabel(readoutItem.id)}</strong>
              {readoutScore && (
                <small>
                  {retrievalTargetLabel} · {activeTunedRanking?.label ?? rankMethodLabel} · raw {readoutScore.raw.toFixed(4)} · cal {readoutScore.calibrated.toFixed(4)} · rank {readoutScore.rank.toFixed(4)}
                  {activeTunedCalibratedScores ? ` · tuned cal ${activeTunedCalibratedScores[readoutItem.rowIndex].toFixed(4)}` : ""}
                  {activeTunedRanks ? ` · tuned rank ${activeTunedRanks[readoutItem.rowIndex].toFixed(4)}` : ""}
                </small>
              )}
            </div>
          )}

          <TopGallery
            items={galleryItems}
            learner={{
              id: activeTunedRanking?.id ?? rankMethod,
              label: activeTunedRanking
                ? `${activeTunedRanking.label} · ${retrievalTargetLabel} · ${resultScopeDefinition.label}`
                : `${rankMethodLabel} · ${retrievalTargetLabel} · ${resultScopeDefinition.label}`,
            }}
            getRank={getRank}
            candidateMask={resultMask}
            preferences={preferences}
            selectedId={selectedId}
            clusterColors={resolveVisualClusterColor}
            title="Ranked Gallery"
            limit={topLimit}
            evaluation={resultScope === "test" ? {
              label: "Frozen Test",
              totalPositiveCount: testPositiveCount,
              isPositive: getGroundTruth,
              filtered: hasResultFilters,
            } : undefined}
            canAnnotate={canAnnotate}
            annotationDisabledReason={annotationDisabledReason}
            getOriginalVqaLabel={getOriginalVqaLabel}
            getAttributeStrengths={getAttributeStrengths}
            attributeStrengthSourceLabel={attributeStrengthSourceLabel}
            onItemSelect={(item) => setSelectedId(item.id)}
            onPreferenceChange={updatePreference}
          />
          </div>
        </section>

        {developmentMode ? (
        <section className="workspace-panel optimization-panel" aria-labelledby="optimization-heading">
          <div className="panel-heading">
            <div>
              <h2 id="optimization-heading">Diagnosis and Refinement</h2>
            </div>
            <span className="panel-kicker">Live filters</span>
          </div>

          <div className="smart-filter-block">
            <div className="section-label"><span>Diagnostic Filters</span><i /></div>
            <SmartFilterPanel
              activeKind={smartFilterKind}
              learnerMethod={smartFilterLearner}
              learnerMethods={smartFilterLearners}
              targetLabel={retrievalTargetLabel}
              comparisonMethod={activeRefinementVisualization ? refinementSourceLabel
                : `Legacy ${methodDisplayLabel(methodIndex.has(rankMethod) ? rankMethod : "Ours-Full")}`}
              sourceLabel={activeRefinementVisualization ? refinementSourceLabel
                : diagnosticSnapshotRequired ? "Current model · snapshot pending" : "Legacy static scores"}
              sourceDetail={activeRefinementVisualization
                ? "Applied snapshot only. Attribute: current gate and probe ranks. Joint: current F rank; each learner is ranked by the product of its normalized attribute scores z (not standalone raw-probability metrics). Ranks use the full Gallery, 1 = best. Prototype is fixed Query, not feedback."
                : "Historical static model scores; these diagnostics do not describe a newer model. Prototype is fixed Query, not feedback."}
              unavailableReason={diagnosticSource.error}
              prototypeComparisonAvailable={
                rankMethod !== "Image Prototype"
                && rankMethod !== HIERARCHICAL_FUSION_METHOD_ID
              }
              result={smartFilterResult}
              onKindChange={(kind) => {
                setSmartFilterKind(kind);
                setSelectedId(null);
                setHoveredId(null);
                if (kind) setTopLimit(50);
              }}
              onLearnerMethodChange={(method) => {
                setSmartFilterLearner(method);
                setSelectedId(null);
                setHoveredId(null);
              }}
            />
          </div>

          <div className="rule-block">
            <div className="section-label"><span>Selected Image Analysis</span><i /></div>
            <SelectionRuleSummaryPanel
              result={selectionRuleSummary}
              active={selectionRuleActive}
            />
            <SelectionOverlapPanel
              result={selectionOverlapSummary}
              active={selectionOverlapActive}
            />
          </div>

          <div className="tuning-block" id="tuning-controls">
            <div className="section-label"><span>Human Feedback</span><i /></div>
            <AnnotationReviewGallery
              taskId={selectedTaskId}
              key={tuning.session?.id ?? `${selectedTaskId}:${retrievalTarget}:${rankMethod}`}
              annotations={tuning.annotations}
              itemById={galleryItemById}
              preferences={preferences}
              busy={tuning.busy}
              canAnnotate={canAnnotate}
              canRemoveAnnotation={canRemoveAnnotation}
              isValidationRow={isValidationRow}
              getOriginalVqaLabel={getOriginalVqaLabel}
              getAttributeStrengths={getAttributeStrengths}
              attributeStrengthSourceLabel={attributeStrengthSourceLabel}
              onPreferenceChange={updateReviewedPreference}
              jointSession={retrievalTarget === jointTarget?.id}
              attributeTargets={attributeTargets}
              onConfirmFailureAttributes={confirmReviewedFailureAttributes}
              clearDisabled={tuning.busy || fixedValidation.status !== "ready"}
              onClearAll={() => tuning.clearAnnotations()}
            />
            <div className="bulk-feedback-actions" aria-live="polite">
              <div>
                <strong>{bulkSelectedItems.length.toLocaleString()} selected</strong>
                {!selectionRuleActive && <span>Brush or choose a cluster first.</span>}
                {selectionRuleActive
                  && bulkSelectedItems.length > MAX_BULK_TUNING_ANNOTATIONS
                  && (
                    <span>
                      Refine to at most {MAX_BULK_TUNING_ANNOTATIONS.toLocaleString()} images.
                    </span>
                  )}
                {selectionRuleActive
                  && bulkSelectedItems.length > 0
                  && bulkSelectedItems.length <= MAX_BULK_TUNING_ANNOTATIONS
                  && (
                    <span>
                      + changes {bulkPositiveChanges.length.toLocaleString()} · − changes {bulkNegativeChanges.length.toLocaleString()}
                    </span>
                  )}
              </div>
              <button
                type="button"
                className="bulk-positive-button"
                disabled={
                  tuning.busy
                  || bulkSelectedItems.length === 0
                  || bulkSelectedItems.length > MAX_BULK_TUNING_ANNOTATIONS
                  || bulkPositiveChanges.length === 0
                }
                onClick={() => void bulkMarkSelection("positive").catch(() => undefined)}
              >
                Mark all positive
              </button>
              <button
                type="button"
                className="bulk-negative-button"
                disabled={
                  tuning.busy
                  || bulkSelectedItems.length === 0
                  || bulkSelectedItems.length > MAX_BULK_TUNING_ANNOTATIONS
                  || bulkNegativeChanges.length === 0
                }
                onClick={() => void bulkMarkSelection("negative").catch(() => undefined)}
              >
                Mark all negative
              </button>
            </div>
            <div className="refinement-actions-block">
              <div className="section-label"><span>Refinement and Validation</span><i /></div>
              {developmentMode && !hierarchicalFusionSelected && !legacyBaselineSelected && (
                <div className="refinement-training-actions">
                  <TuningFunctionActions
                    state={tuning}
                    taskId={selectedTaskId}
                    targetId={retrievalTarget}
                    baseMethod={rankMethod}
                    validationReady={fixedValidation.status === "ready"}
                  />
                </div>
              )}
              {hierarchicalFusionSelected ? (
                <div className="feedback-empty">
                  Custom Fusion cannot be tuned in this session.
                </div>
              ) : (
                <TuningPanel
                  state={tuning}
                  validation={fixedValidation.audit}
                  isApplied={Boolean(
                    tuning.run
                    && (
                      tuning.appliedRunId === tuning.run.id
                      || appliedFusionTune?.runId === tuning.run.id
                    )
                  )}
                  applyBusy={fusionTuneHierarchy.status === "loading"}
                  applyError={fusionTuneApplyError ?? fusionTuneHierarchy.error}
                  onApplyRun={applyTunedRanking}
                  onRevertRun={revertTunedRanking}
                />
              )}
            </div>
          </div>
        </section>
        ) : (
        <section className="workspace-panel test-audit-panel" aria-labelledby="test-audit-heading">
          <div className="panel-heading">
            <div>
              <p className="panel-index">Frozen Test</p>
              <h2 id="test-audit-heading">Diagnosis and Refinement</h2>
            </div>
          </div>
          <dl className="test-audit-facts">
            <div><dt>Population</dt><dd>{dataset.manifest.evaluation.frozenTest.rowCount.toLocaleString()}</dd></div>
            <div><dt>{retrievalTargetLabel} positives</dt><dd>{testPositiveCount.toLocaleString()}</dd></div>
            <div>
              <dt>Current ranking</dt>
              <dd>
                {activeTunedRanking
                  ? `${activeTunedRanking.label} · ${methodDisplayLabel(appliedFusionTune?.baseMethod ?? tuning.run?.baseMethod ?? rankMethodLabel)}`
                  : rankMethodLabel}
              </dd>
            </div>
            <div><dt>Filters</dt><dd>{hasResultFilters ? "Active from Development" : "None"}</dd></div>
          </dl>
          {tuning.run && <FrozenTestEvaluation run={tuning.run} stale={tuning.resultStale} />}
        </section>
        )}
      </div>
      </div>
      )}
    </main>
  );
}
