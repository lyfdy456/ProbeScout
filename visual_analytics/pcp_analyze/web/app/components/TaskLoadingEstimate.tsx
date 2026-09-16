import {
  CONCURRENT_LOAD_WARNING_ZH,
  DEFAULT_LOAD_BANDWIDTH_MBPS,
  estimateLoadTime,
  formatByteSize,
  formatLoadTimeRangeZh,
  type InitialDownloadResolution,
} from "../lib/taskLoadingEstimate";

interface TaskLoadingEstimateProps {
  resolution: InitialDownloadResolution | null | undefined;
  bandwidthMbps?: number;
  className?: string;
}

/** A presentation-only loading hint; resolving manifest/HEAD sizes stays in the loader. */
export function TaskLoadingEstimate({
  resolution,
  bandwidthMbps = DEFAULT_LOAD_BANDWIDTH_MBPS,
  className,
}: TaskLoadingEstimateProps) {
  if (!resolution || resolution.bytes <= 0) return null;

  const estimate = estimateLoadTime(resolution.bytes, bandwidthMbps);
  const prefix = resolution.complete ? "预计加载约" : "按已知数据预计至少";

  return (
    <div className={className} role="status" aria-live="polite">
      <p>
        {prefix} {formatLoadTimeRangeZh(estimate)}（{bandwidthMbps} Mbps，
        {formatByteSize(resolution.bytes)}）
      </p>
      <p>{CONCURRENT_LOAD_WARNING_ZH}</p>
    </div>
  );
}
