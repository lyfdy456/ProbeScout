"use client";

export interface AttributeStrengthPoint {
  id: string;
  label: string;
  /** Calibrated model strength. Values are plotted on the audited 0..1 scale. */
  value: number;
  active?: boolean;
}

export interface AttributeStrengthProfileProps {
  points: readonly AttributeStrengthPoint[];
  variant?: "compact" | "expanded";
  sourceLabel?: string;
}

function clampUnit(value: number) {
  return Math.max(0, Math.min(1, value));
}

function shortenedLabel(label: string) {
  return label.length > 28 ? `${label.slice(0, 27)}…` : label;
}

export function AttributeStrengthProfile({
  points,
  variant = "compact",
  sourceLabel,
}: AttributeStrengthProfileProps) {
  const finitePoints = points.filter((point) => Number.isFinite(point.value));
  if (finitePoints.length === 0) return null;

  const compact = variant === "compact";
  const width = compact ? 100 : 430;
  const rowStep = compact ? 16 : 30;
  const top = compact ? 8 : 15;
  const height = compact
    ? finitePoints.length * rowStep
    : Math.max(52, top * 2 + (finitePoints.length - 1) * rowStep);
  const plotStart = compact ? 7 : 176;
  const plotEnd = compact ? 93 : 354;
  const pathPoints = finitePoints.map((point, index) => {
    const x = plotStart + clampUnit(point.value) * (plotEnd - plotStart);
    const y = top + index * rowStep;
    return { point, x, y };
  });
  const ariaLabel = `Attribute strengths: ${finitePoints
    .map((point) => `${point.label} ${clampUnit(point.value).toFixed(3)}`)
    .join(", ")}`;

  return (
    <div
      className={`attribute-strength-profile attribute-strength-profile-${variant}`}
      role="img"
      aria-label={ariaLabel}
      title={compact ? ariaLabel : undefined}
    >
      {!compact && sourceLabel && (
        <div className="attribute-strength-profile-source">{sourceLabel}</div>
      )}
      {compact && (
        <div className="attribute-strength-profile-compact-labels" aria-hidden="true">
          {finitePoints.map((point) => (
            <span key={point.id} title={`${point.label}: ${clampUnit(point.value).toFixed(3)}`}>
              {point.label}
            </span>
          ))}
        </div>
      )}
      <svg
        className="attribute-strength-profile-svg"
        viewBox={`0 0 ${width} ${height}`}
        style={compact ? { height } : undefined}
        preserveAspectRatio={compact ? "none" : undefined}
        aria-hidden="true"
        focusable="false"
      >
        {pathPoints.map(({ point, y }) => (
          <g key={`guide:${point.id}`}>
            {!compact && (
              <text className="attribute-strength-profile-label" x="4" y={y + 4}>
                <title>{point.label}</title>
                {shortenedLabel(point.label)}
              </text>
            )}
            <line
              className="attribute-strength-profile-guide"
              x1={plotStart}
              x2={plotEnd}
              y1={y}
              y2={y}
            />
          </g>
        ))}
        {pathPoints.length > 1 && (
          <polyline
            className="attribute-strength-profile-line"
            points={pathPoints.map(({ x, y }) => `${x},${y}`).join(" ")}
          />
        )}
        {pathPoints.map(({ point, x, y }) => (
          <g key={`point:${point.id}`}>
            <circle
              className={`attribute-strength-profile-point ${
                point.active ? "attribute-strength-profile-point-active" : ""
              }`.trim()}
              cx={x}
              cy={y}
              r={compact ? 2.4 : 4}
            >
              <title>{point.label}: {clampUnit(point.value).toFixed(3)}</title>
            </circle>
            {!compact && (
              <text className="attribute-strength-profile-value" x="368" y={y + 4}>
                {clampUnit(point.value).toFixed(3)}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

export default AttributeStrengthProfile;
