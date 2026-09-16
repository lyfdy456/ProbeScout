import type {
  SelectionOverlapConditionSummary,
  SelectionOverlapRegion,
  SelectionOverlapResult,
  SelectionOverlapRule,
} from "../lib/selectionOverlap";
import type { SelectionRuleValueRange } from "../lib/selectionRuleSummary";

interface SelectionOverlapPanelProps {
  result: SelectionOverlapResult;
  active: boolean;
}

interface RegionAnchor {
  key: number;
  x: number;
  y: number;
}

const ONE_SET_ANCHORS: readonly RegionAnchor[] = [{ key: 1, x: 160, y: 101 }];
const TWO_SET_ANCHORS: readonly RegionAnchor[] = [
  { key: 1, x: 82, y: 101 },
  { key: 3, x: 160, y: 101 },
  { key: 2, x: 238, y: 101 },
];
const THREE_SET_ANCHORS: readonly RegionAnchor[] = [
  { key: 1, x: 66, y: 77 },
  { key: 2, x: 254, y: 77 },
  { key: 4, x: 160, y: 232 },
  { key: 3, x: 160, y: 58 },
  { key: 5, x: 105, y: 166 },
  { key: 6, x: 215, y: 166 },
  { key: 7, x: 160, y: 129 },
];

function toneClass(index: number) {
  return `set-tone-${index % 6}`;
}

function describeRegion(region: SelectionOverlapRegion, totalSets: number) {
  const joined = region.codes.join(" and ");
  return region.codes.length === totalSets
    ? `${joined}: ${region.count.toLocaleString()} images`
    : `${joined} only: ${region.count.toLocaleString()} images`;
}

function formatAdaptive(value: number, range: SelectionRuleValueRange) {
  const span = Math.abs(range.upperQuantile - range.lowerQuantile);
  const decimals = Math.min(8, Math.max(4, span > 0 ? Math.ceil(-Math.log10(span)) + 2 : 4));
  return value.toFixed(decimals);
}

function formatDirectNumber(value: number, lower: number, upper: number) {
  const span = Math.abs(upper - lower);
  const decimals = Math.min(12, Math.max(3, span > 0 ? Math.ceil(-Math.log10(span)) + 2 : 3));
  const rounded = Number(value.toFixed(decimals));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

function directRuleText(rule: SelectionOverlapRule) {
  if (rule.kind === "category") {
    const field = rule.field === "rank-cluster" ? "Rank-profile cluster" : "Embedding cluster";
    return `${field} = ${rule.value}`;
  }
  if (rule.kind === "projection") {
    const xLower = formatDirectNumber(rule.xDomain[0], rule.xDomain[0], rule.xDomain[1]);
    const xUpper = formatDirectNumber(rule.xDomain[1], rule.xDomain[0], rule.xDomain[1]);
    const yLower = formatDirectNumber(rule.yDomain[0], rule.yDomain[0], rule.yDomain[1]);
    const yUpper = formatDirectNumber(rule.yDomain[1], rule.yDomain[0], rule.yDomain[1]);
    return `${rule.projection.toUpperCase()} rectangle · ${xLower} ≤ x ≤ ${xUpper} · ${yLower} ≤ y ≤ ${yUpper}`;
  }
  const metric = rule.valueKind === "rank" ? "Rank" : "Calibrated score";
  const source = rule.source === "zoom+brush"
    ? "Zoom + brush"
    : rule.source === "zoom"
      ? "Zoom range"
      : "Brush range";
  return `${source} · ${formatDirectNumber(rule.lower, rule.lower, rule.upper)} ≤ ${metric} ≤ ${formatDirectNumber(rule.upper, rule.lower, rule.upper)}`;
}

function briefRuleText(rule: SelectionOverlapRule) {
  if (rule.kind === "category") {
    return directRuleText(rule);
  }
  if (rule.kind === "projection") {
    return `${rule.projection.toUpperCase()} box`;
  }
  const source = rule.source === "zoom+brush"
    ? "Zoom + brush"
    : rule.source === "zoom" ? "Zoom" : "Brush";
  const metric = rule.valueKind === "rank" ? "Rank" : "Score";
  return `${source} · ${metric} ≈ ${rule.lower.toFixed(3)}–${rule.upper.toFixed(3)}`;
}

function conditionName(condition: SelectionOverlapConditionSummary) {
  if (condition.rule.kind === "projection") return condition.rule.projection.toUpperCase();
  if (condition.rule.kind === "category") {
    return condition.rule.field === "rank-cluster" ? "PCP" : "Embedding cluster";
  }
  return `PCP · ${condition.label}`;
}

function RuleProfile({ condition }: { condition: SelectionOverlapConditionSummary }) {
  if (condition.ruleResult.kind === "none") {
    const messages = {
      "empty-base": "No rows in this scope.",
      "empty-selection": "This condition matches no rows.",
      "entire-base-selected": "This condition covers the full scope.",
      "no-attributes": "No attribute range columns are available.",
      "non-finite-value": "An attribute range is unavailable.",
    } as const;
    return <p className="selection-overlap-profile-empty">{messages[condition.ruleResult.reason]}</p>;
  }

  return (
    <div className="selection-overlap-profile">
      <span>Condition alone · attribute ranges · 5–95%</span>
      {condition.ruleResult.summary.attributes.map((attribute) => (
        <div key={attribute.attributeId}>
          <strong title={attribute.attributeLabel}>{attribute.attributeLabel}</strong>
          <code title={`Observed rank ${attribute.rank.minimum.toFixed(6)}–${attribute.rank.maximum.toFixed(6)}`}>
            R {formatAdaptive(attribute.rank.lowerQuantile, attribute.rank)}–{formatAdaptive(attribute.rank.upperQuantile, attribute.rank)}
          </code>
          <code title={`Observed calibrated score ${attribute.calibratedScore.minimum.toFixed(6)}–${attribute.calibratedScore.maximum.toFixed(6)}`}>
            S {attribute.calibratedScore.lowerQuantile.toFixed(3)}–{attribute.calibratedScore.upperQuantile.toFixed(3)}
          </code>
        </div>
      ))}
    </div>
  );
}

function ConditionCard({
  condition,
  index,
}: {
  condition: SelectionOverlapConditionSummary;
  index: number;
}) {
  return (
    <article className={`selection-overlap-condition ${toneClass(index)}`}>
      <header>
        <i aria-hidden="true">{condition.code}</i>
        <span title={condition.label}>{condition.label}</span>
        <strong title="Matches this condition alone">{condition.count.toLocaleString()}</strong>
      </header>
      <p className="selection-overlap-rule-brief">{briefRuleText(condition.rule)}</p>
      <p className="selection-overlap-direct-rule">
        <span>Direct rule</span>
        <code>{directRuleText(condition.rule)}</code>
      </p>
      <RuleProfile condition={condition} />
    </article>
  );
}

function ExactVennDiagram({
  result,
}: {
  result: Extract<SelectionOverlapResult, { kind: "summary" }>;
}) {
  const { visibleConditions, regions } = result.summary;
  const count = visibleConditions.length;
  if (count === 0) return null;

  const viewBox = count === 3 ? "0 0 320 276" : "0 0 320 196";
  const anchors = count === 1
    ? ONE_SET_ANCHORS
    : count === 2
      ? TWO_SET_ANCHORS
      : THREE_SET_ANCHORS;
  const regionByKey = new Map(regions.map((region) => [region.key, region]));
  const circleGeometry = count === 1
    ? [{ cx: 160, cy: 98, r: 76 }]
    : count === 2
      ? [{ cx: 118, cy: 98, r: 76 }, { cx: 202, cy: 98, r: 76 }]
      : [
          { cx: 112, cy: 101, r: 86 },
          { cx: 208, cy: 101, r: 86 },
          { cx: 160, cy: 181, r: 86 },
        ];
  const codeAnchors = count === 1
    ? [{ x: 160, y: 46 }]
    : count === 2
      ? [{ x: 82, y: 55 }, { x: 238, y: 55 }]
      : [{ x: 65, y: 36 }, { x: 255, y: 36 }, { x: 160, y: 256 }];

  return (
    <svg
      className={`selection-overlap-venn selection-overlap-venn--${count}`}
      viewBox={viewBox}
      aria-hidden="true"
      focusable="false"
    >
      {circleGeometry.map((circle, index) => (
        <circle
          key={visibleConditions[index].id}
          className={`selection-overlap-circle ${toneClass(index)}`}
          cx={circle.cx}
          cy={circle.cy}
          r={circle.r}
        />
      ))}
      {codeAnchors.map((anchor, index) => (
        <text key={visibleConditions[index].id} x={anchor.x} y={anchor.y} className="selection-overlap-set-code">
          {visibleConditions[index].code}
        </text>
      ))}
      {anchors.map((anchor) => {
        const region = regionByKey.get(anchor.key);
        if (!region) return null;
        return (
          <g key={anchor.key}>
            <title>{describeRegion(region, count)}</title>
            <text x={anchor.x} y={anchor.y} className="selection-overlap-region-count">
              {region.count.toLocaleString()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ManySetDiagram({
  conditions,
  intersectionCount,
}: {
  conditions: readonly SelectionOverlapConditionSummary[];
  intersectionCount: number;
}) {
  const centerX = 160;
  const centerY = 126;
  const orbit = 38;
  const radius = 80;
  return (
    <svg
      className="selection-overlap-venn selection-overlap-venn--many"
      viewBox="0 0 320 260"
      aria-hidden="true"
      focusable="false"
    >
      {conditions.map((condition, index) => {
        const angle = -Math.PI / 2 + (2 * Math.PI * index) / conditions.length;
        const cx = centerX + Math.cos(angle) * orbit;
        const cy = centerY + Math.sin(angle) * orbit;
        return (
          <g key={condition.id}>
            <circle
              className={`selection-overlap-circle ${toneClass(index)}`}
              cx={cx}
              cy={cy}
              r={radius}
            />
            <text
              className="selection-overlap-set-code"
              x={centerX + Math.cos(angle) * 103}
              y={centerY + Math.sin(angle) * 103}
            >
              {condition.code}
            </text>
          </g>
        );
      })}
      <text x={centerX} y={centerY - 2} className="selection-overlap-many-count">
        {intersectionCount.toLocaleString()}
      </text>
      <text x={centerX} y={centerY + 16} className="selection-overlap-many-label">
        ALL {conditions.length} · AND
      </text>
    </svg>
  );
}

export function SelectionOverlapPanel({ result, active }: SelectionOverlapPanelProps) {
  if (!active || result.kind === "none") return null;

  const summary = result.summary;
  const conditionCount = summary.conditions.length;
  const exactVenn = conditionCount > 0 && conditionCount <= 3;

  return (
    <figure className="selection-overlap-panel" aria-label="Set Overlap Analysis">
      <div className="selection-overlap-heading">
        <strong>Set Overlap Analysis</strong>
        <span>{summary.baseCount.toLocaleString()} in scope</span>
      </div>

      <div className="selection-overlap-compact">
        <div className="selection-overlap-diagram" title="Exact counts; circle areas are schematic.">
          {exactVenn && <ExactVennDiagram result={result} />}
          {conditionCount > 3 && (
            <ManySetDiagram conditions={summary.conditions} intersectionCount={summary.coreIntersectionCount} />
          )}
          {conditionCount === 0 && <p className="selection-overlap-note">No independent conditions</p>}
        </div>
        <div className="selection-overlap-compact-summary">
          <div className="selection-overlap-counts" aria-label="Independent condition counts">
            {summary.conditions.map((condition, index) => (
              <div key={condition.id} className={toneClass(index)} title={`${condition.label}; ${directRuleText(condition.rule)}`}>
                <i aria-hidden="true">{condition.code}</i>
                <span>{conditionName(condition)}:</span>
                <strong>{condition.count.toLocaleString()}</strong>
              </div>
            ))}
            <div className="selection-overlap-intersection">
              <span>{conditionCount > 0 ? "Intersection:" : "Scope:"}</span>
              <strong>{summary.coreIntersectionCount.toLocaleString()}</strong>
            </div>
            {(summary.stages.length > 0 || summary.finalCount !== summary.coreIntersectionCount) && (
              <div className="selection-overlap-final-count" title="After downstream diagnostic filtering">
                <span>Final:</span><strong>{summary.finalCount.toLocaleString()}</strong>
              </div>
            )}
          </div>
          <details className="selection-overlap-details">
            <summary>Rules &amp; counts</summary>
            <p className="selection-overlap-note">
              {exactVenn
                ? "Exclusive counts are exact; circle areas are schematic."
                : "All conditions are included; the center is their exact intersection."}
            </p>
            <div className="selection-overlap-condition-grid" aria-label="Independent filter rules">
              {summary.conditions.map((condition, index) => (
                <ConditionCard key={condition.id} condition={condition} index={index} />
              ))}
            </div>
            {conditionCount > 3 && (
              <div className="selection-overlap-waterfall" aria-label="Cumulative independent filter intersection">
                {summary.conditions.map((condition) => (
                  <div key={condition.id}>
                    <span>{condition.code} · {condition.label}</span>
                    <small>alone {condition.count.toLocaleString()}</small>
                    <strong>AND {condition.cumulativeCount.toLocaleString()}</strong>
                  </div>
                ))}
                <p>Intermediate AND follows display order; the final intersection is order-independent.</p>
              </div>
            )}
            <div className="selection-overlap-flow" aria-label="Selection reduction stages">
              <div>
                <span>{conditionCount > 0 ? "All independent conditions · AND" : "Current scope"}</span>
                <strong>{summary.coreIntersectionCount.toLocaleString()}</strong>
              </div>
              {summary.stages.map((stage) => (
                <div key={stage.id}>
                  <span><i aria-hidden="true">→</i>{stage.label}</span>
                  <strong>{stage.count.toLocaleString()}</strong>
                </div>
              ))}
              <div className="selection-overlap-final">
                <span>Final candidates</span><strong>{summary.finalCount.toLocaleString()}</strong>
              </div>
            </div>
          </details>
        </div>
      </div>

      <div className="sr-only">
      <table>
        <caption>Exact independent filter overlap counts</caption>
        <thead><tr><th>Condition, region or stage</th><th>Images</th></tr></thead>
        <tbody>
          {summary.conditions.map((condition) => (
            <tr key={condition.id}>
              <th>{condition.code}: {condition.label}; {directRuleText(condition.rule)}; matches alone</th>
              <td>{condition.count}</td>
            </tr>
          ))}
          {exactVenn && summary.regions.map((region) => (
            <tr key={region.key}>
              <th>{describeRegion(region, conditionCount)}</th>
              <td>{region.count}</td>
            </tr>
          ))}
          {exactVenn && (
            <tr><th>Outside all visible conditions</th><td>{summary.outsideVisibleCount}</td></tr>
          )}
          <tr><th>All independent conditions</th><td>{summary.coreIntersectionCount}</td></tr>
          {summary.stages.map((stage) => (
            <tr key={stage.id}><th>{stage.label}</th><td>{stage.count}</td></tr>
          ))}
          <tr><th>Final candidates</th><td>{summary.finalCount}</td></tr>
        </tbody>
      </table>
      </div>
    </figure>
  );
}

export default SelectionOverlapPanel;
