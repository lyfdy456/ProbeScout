import type {
  SelectionRuleSummaryResult,
  SelectionRuleValueRange,
} from "../lib/selectionRuleSummary";

interface SelectionRuleSummaryPanelProps {
  result: SelectionRuleSummaryResult;
  active: boolean;
}

const EMPTY_MESSAGES: Record<
  Exclude<SelectionRuleSummaryResult, { kind: "summary" }>['reason'],
  string
> = {
  "empty-base": "No images are available in this scope.",
  "empty-selection": "The current selection is empty.",
  "entire-base-selected": "Brush or choose a cluster to summarize its rank and score ranges.",
  "no-attributes": "This task has no attribute rank or score columns.",
  "non-finite-value": "A selected attribute rank or score is unavailable.",
};

function formatRankValue(value: number, range: SelectionRuleValueRange): string {
  const span = Math.abs(range.upperQuantile - range.lowerQuantile);
  const decimals = Math.min(
    8,
    Math.max(4, span > 0 ? Math.ceil(-Math.log10(span)) + 2 : 4),
  );
  return value.toFixed(decimals);
}

function formatObservedRange(label: string, range: SelectionRuleValueRange): string {
  return `${label} observed min ${range.minimum.toFixed(6)} · max ${range.maximum.toFixed(6)}`;
}

export function SelectionRuleSummaryPanel({
  result,
  active,
}: SelectionRuleSummaryPanelProps) {
  if (!active || result.kind === "none") {
    const message = result.kind === "none"
      ? EMPTY_MESSAGES[result.reason]
      : EMPTY_MESSAGES["entire-base-selected"];
    return <p className="selection-rule-empty">{message}</p>;
  }

  return (
    <details className="selection-rule-summary selection-rule-disclosure">
      <summary className="selection-rule-meta" aria-live="polite" aria-atomic="true">
        <strong>{result.summary.selectedCount.toLocaleString()} selected</strong>
        <span>5–95% ranges</span>
      </summary>
      <div className="selection-rule-body">
        <div className="selection-rule-table-wrap">
          <table className="selection-rule-table">
            <caption className="sr-only">Selected Image Analysis: rank and calibrated score ranges by attribute</caption>
            <thead>
              <tr>
                <th scope="col">Attribute</th>
                <th scope="col">Rank <small>1 = best</small></th>
                <th scope="col">Score <small>calibrated</small></th>
              </tr>
            </thead>
            <tbody>
              {result.summary.attributes.map((range) => (
                <tr key={range.attributeId}>
                  <th scope="row" title={range.attributeLabel}>{range.attributeLabel}</th>
                  <td title={formatObservedRange("Rank", range.rank)}>
                    {formatRankValue(range.rank.lowerQuantile, range.rank)}
                    <span aria-hidden="true">–</span>
                    {formatRankValue(range.rank.upperQuantile, range.rank)}
                  </td>
                  <td title={formatObservedRange("Calibrated score", range.calibratedScore)}>
                    {range.calibratedScore.lowerQuantile.toFixed(3)}
                    <span aria-hidden="true">–</span>
                    {range.calibratedScore.upperQuantile.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <small>Descriptive only; it does not add another filter.</small>
      </div>
    </details>
  );
}

export default SelectionRuleSummaryPanel;
