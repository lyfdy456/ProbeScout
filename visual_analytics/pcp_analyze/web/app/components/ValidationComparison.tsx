"use client";

import type { RankedScopeEvaluation } from "../lib/validationMetrics";

export interface ValidationComparisonRow {
  id: string;
  label: string;
  detail?: string;
  metrics: RankedScopeEvaluation;
  current?: boolean;
  baseline?: boolean;
}

interface ValidationComparisonProps {
  targetLabel: string;
  rows: readonly ValidationComparisonRow[];
  validationRowCount: number;
  validationPositiveCount: number;
  k?: number;
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export function ValidationComparison({
  targetLabel,
  rows,
  validationRowCount,
  validationPositiveCount,
  k = 50,
}: ValidationComparisonProps) {
  const ordered = [...rows].sort((left, right) => (
    right.metrics.averagePrecision - left.metrics.averagePrecision
    || right.metrics.bestF1 - left.metrics.bestF1
    || left.label.localeCompare(right.label)
  ));
  const best = ordered[0] ?? null;

  return (
    <div className="validation-comparison">
      <div className="validation-summary-cards" aria-live="polite">
        <article>
          <span>Web Validation population</span>
          <strong>{validationRowCount.toLocaleString()}</strong>
          <small>{validationPositiveCount.toLocaleString()} positives for {targetLabel}</small>
        </article>
        <article>
          <span>Best validation AP</span>
          <strong>{best ? percent(best.metrics.averagePrecision) : "—"}</strong>
          <small>{best?.label ?? "No comparable ranking"}</small>
        </article>
        <article>
          <span>Selection rule</span>
          <strong>AP + F1</strong>
          <small>Static / manual configurations</small>
        </article>
      </div>

      <div className="validation-table-wrap">
        <table className="validation-table">
          <caption className="sr-only">
            Validation metrics for {targetLabel}; larger values are better.
          </caption>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Method / configuration</th>
              <th scope="col">AP</th>
              <th scope="col">Best F1</th>
              <th scope="col">P@{k}</th>
              <th scope="col">Recall@{k}</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((row, index) => (
              <tr
                key={row.id}
                className={[
                  row.current ? "current" : "",
                  row.baseline ? "baseline" : "",
                ].filter(Boolean).join(" ")}
              >
                <td>{index + 1}</td>
                <th scope="row">
                  <span>{row.label}</span>
                  <small>
                    {row.detail
                      ?? (row.current ? "Current ranking" : row.baseline ? "Frozen baseline" : "Exported method")}
                  </small>
                </th>
                <td><strong>{percent(row.metrics.averagePrecision)}</strong></td>
                <td>
                  <strong>{percent(row.metrics.bestF1)}</strong>
                  <small>Top {row.metrics.bestCutoff.toLocaleString()}</small>
                </td>
                <td>
                  <strong>{percent(row.metrics.precisionAtK)}</strong>
                  <small>{row.metrics.truePositiveAtK}/{Math.min(k, row.metrics.evaluatedCount)}</small>
                </td>
                <td><strong>{percent(row.metrics.recallAtK)}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ValidationComparison;
