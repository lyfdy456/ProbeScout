export const DEFAULT_LOAD_BANDWIDTH_MBPS = 3;

export const CONCURRENT_LOAD_WARNING_ZH =
  "多人同时加载会共享带宽，实际用时可能更长。";

const REQUIRED_INITIAL_FILE_KEYS = [
  "metadata",
  "imageIds",
  "rawScores",
  "calibratedScores",
  "ranks",
  "pca2d",
  "metrics",
  "groundTruth",
  "developmentMask",
  "validationMask",
  "testMask",
] as const;

const UMAP_FILE_KEY = "umap2d";

export interface LoadFileSpec {
  path: string;
  bytes?: number | null;
}

export interface InitialLoadManifestLike {
  files: object;
  clusters?: {
    schemes?: Array<{
      labelsFileKey?: string;
    }>;
  };
  projections?: {
    umap?: {
      available?: boolean;
    };
  };
}

export type InitialDownloadSource =
  | "manifest"
  | "catalog"
  | "manifest+head"
  | "partial"
  | "unavailable";

export interface InitialDownloadResolution {
  /** Known bytes transferred before the analysis workspace becomes interactive. */
  bytes: number;
  /** False means `bytes` is a lower bound because one or more sizes are unknown. */
  complete: boolean;
  source: InitialDownloadSource;
  unresolvedFiles: string[];
}

export interface ResolveInitialDownloadOptions {
  dataRoot: string;
  /** Optional total generated into a task catalog, used when file specs omit sizes. */
  catalogInitialDownloadBytes?: number | null;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}

export interface LoadTimeEstimate {
  bytes: number;
  bandwidthMbps: number;
  /** Best case at the full nominal line rate. */
  minSeconds: number;
  /** Typical slower case with protocol/tunnel overhead and normal variation. */
  maxSeconds: number;
}

interface CollectedInitialFiles {
  files: LoadFileSpec[];
  missingKeys: string[];
}

function isKnownByteCount(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function asFileSpec(value: unknown): LoadFileSpec | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as { path?: unknown; bytes?: unknown };
  if (typeof candidate.path !== "string" || candidate.path.length === 0) return null;
  return {
    path: candidate.path,
    bytes: isKnownByteCount(candidate.bytes) ? candidate.bytes : undefined,
  };
}

function collectInitialFiles(manifest: InitialLoadManifestLike): CollectedInitialFiles {
  const fileMap = manifest.files as Record<string, unknown>;
  const clusterFileKeys: string[] = [];
  const missingClusterKeys: string[] = [];
  manifest.clusters?.schemes?.forEach((scheme, index) => {
    if (typeof scheme.labelsFileKey === "string" && scheme.labelsFileKey.length > 0) {
      clusterFileKeys.push(scheme.labelsFileKey);
    } else {
      missingClusterKeys.push(`clusters.schemes[${index}].labelsFileKey`);
    }
  });
  const requestedKeys = [...new Set([
    ...REQUIRED_INITIAL_FILE_KEYS,
    ...clusterFileKeys,
  ])];
  if (manifest.projections?.umap?.available === true && fileMap[UMAP_FILE_KEY]) {
    requestedKeys.push(UMAP_FILE_KEY);
  }

  const missingKeys: string[] = [...missingClusterKeys];
  const byPath = new Map<string, LoadFileSpec>();

  for (const key of requestedKeys) {
    const spec = asFileSpec(fileMap[key]);
    if (!spec) {
      missingKeys.push(key);
      continue;
    }
    const previous = byPath.get(spec.path);
    if (!previous || previous.bytes === undefined && spec.bytes !== undefined) {
      byPath.set(spec.path, spec);
    }
  }

  return { files: [...byPath.values()], missingKeys };
}

/**
 * Resolves bytes synchronously when every initially fetched manifest file has a
 * valid `bytes` field. It deliberately excludes thumbnails and fixed-query
 * images, which load lazily after the workspace is interactive.
 */
export function initialDownloadBytesFromManifest(
  manifest: InitialLoadManifestLike,
): InitialDownloadResolution {
  const { files, missingKeys } = collectInitialFiles(manifest);
  const unresolvedFiles = [
    ...missingKeys.map((key) => key.startsWith("clusters.") ? key : `files.${key}`),
    ...files.filter((file) => file.bytes === undefined).map((file) => file.path),
  ];
  const bytes = files.reduce((sum, file) => sum + (file.bytes ?? 0), 0);

  if (unresolvedFiles.length === 0) {
    return { bytes, complete: true, source: "manifest", unresolvedFiles: [] };
  }
  return {
    bytes,
    complete: false,
    source: bytes > 0 ? "partial" : "unavailable",
    unresolvedFiles,
  };
}

/** Allows a catalog entry to show an estimate before its manifest request finishes. */
export function initialDownloadBytesFromCatalog(
  initialDownloadBytes: number | null | undefined,
): InitialDownloadResolution | null {
  if (!isKnownByteCount(initialDownloadBytes)) return null;
  return {
    bytes: initialDownloadBytes,
    complete: true,
    source: "catalog",
    unresolvedFiles: [],
  };
}

function normalizeDataRoot(dataRoot: string): string {
  const rooted = dataRoot.startsWith("/") ? dataRoot : `/${dataRoot}`;
  return rooted.replace(/\/$/, "");
}

function dataPath(dataRoot: string, path: string): string {
  if (path.startsWith("/")) return path;
  return `${normalizeDataRoot(dataRoot)}/${path}`;
}

async function contentLengthFromHead(
  path: string,
  options: ResolveInitialDownloadOptions,
): Promise<number | null> {
  const fetchImpl = options.fetchImpl ?? fetch;
  try {
    const response = await fetchImpl(dataPath(options.dataRoot, path), {
      method: "HEAD",
      cache: "force-cache",
      signal: options.signal,
    });
    if (!response.ok) return null;
    const header = response.headers.get("content-length");
    if (header === null || header.trim() === "") return null;
    const value = Number(header);
    return isKnownByteCount(value) ? value : null;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    return null;
  }
}

/**
 * Resolves missing sizes without downloading a file body a second time.
 * Resolution order is manifest bytes, an optional catalog total, then HEAD
 * Content-Length. Servers that do not support HEAD return a safe lower bound.
 */
export async function resolveInitialDownloadBytes(
  manifest: InitialLoadManifestLike,
  options: ResolveInitialDownloadOptions,
): Promise<InitialDownloadResolution> {
  const fromManifest = initialDownloadBytesFromManifest(manifest);
  if (fromManifest.complete) return fromManifest;

  const fromCatalog = initialDownloadBytesFromCatalog(options.catalogInitialDownloadBytes);
  if (fromCatalog) return fromCatalog;

  const { files, missingKeys } = collectInitialFiles(manifest);
  const unresolvedSpecs = files.filter((file) => file.bytes === undefined);
  const headLengths = await Promise.all(
    unresolvedSpecs.map((file) => contentLengthFromHead(file.path, options)),
  );
  let bytes = files.reduce((sum, file) => sum + (file.bytes ?? 0), 0);
  const unresolvedFiles = missingKeys.map((key) => (
    key.startsWith("clusters.") ? key : `files.${key}`
  ));

  for (let index = 0; index < unresolvedSpecs.length; index += 1) {
    const contentLength = headLengths[index];
    if (contentLength === null) unresolvedFiles.push(unresolvedSpecs[index].path);
    else bytes += contentLength;
  }

  return {
    bytes,
    complete: unresolvedFiles.length === 0,
    source: unresolvedFiles.length === 0
      ? "manifest+head"
      : bytes > 0
        ? "partial"
        : "unavailable",
    unresolvedFiles,
  };
}

export function estimateLoadTime(
  bytes: number,
  bandwidthMbps = DEFAULT_LOAD_BANDWIDTH_MBPS,
): LoadTimeEstimate {
  if (!isKnownByteCount(bytes)) throw new RangeError("bytes must be a non-negative number");
  if (!Number.isFinite(bandwidthMbps) || bandwidthMbps <= 0) {
    throw new RangeError("bandwidthMbps must be greater than zero");
  }

  const lineRateSeconds = bytes === 0 ? 0 : (bytes * 8) / (bandwidthMbps * 1_000_000);
  const minSeconds = Math.ceil(lineRateSeconds);
  // 65% effective throughput is a conservative normal case for a tunneled
  // development server. Two seconds cover request setup and task initialization.
  const maxSeconds = bytes === 0
    ? 0
    : Math.max(minSeconds, Math.ceil(lineRateSeconds / 0.65 + 2));

  return { bytes, bandwidthMbps, minSeconds, maxSeconds };
}

export function formatByteSize(bytes: number): string {
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const mib = bytes / (1024 * 1024);
  if (mib < 1) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${mib.toFixed(mib >= 10 ? 1 : 2)} MiB`;
}

export function formatDurationZh(seconds: number): string {
  const rounded = Math.max(0, Math.round(seconds));
  if (rounded < 60) return `${rounded} 秒`;
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  if (remainder === 0) return `${minutes} 分钟`;
  return `${minutes} 分 ${remainder} 秒`;
}

export function formatLoadTimeRangeZh(estimate: LoadTimeEstimate): string {
  const lower = formatDurationZh(estimate.minSeconds);
  const upper = formatDurationZh(estimate.maxSeconds);
  return estimate.minSeconds === estimate.maxSeconds ? lower : `${lower}–${upper}`;
}
