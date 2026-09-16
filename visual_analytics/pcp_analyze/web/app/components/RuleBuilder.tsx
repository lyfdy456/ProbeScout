"use client";

import type {
  RuleClause,
  RuleComparator,
  RuleGroup,
  RuleLogic,
  RuleValueKind,
} from "../ruleFilter";

interface RuleTargetOption {
  id: string;
  label: string;
}

interface RuleBuilderProps {
  groups: readonly RuleGroup[];
  rootLogic: RuleLogic;
  targets: readonly RuleTargetOption[];
  defaultTargetId?: string;
  methodLabel: string;
  onGroupsChange: (groups: RuleGroup[]) => void;
  onRootLogicChange: (logic: RuleLogic) => void;
}

let nextRuleId = 0;

function createId(prefix: string) {
  nextRuleId += 1;
  return `${prefix}-${Date.now().toString(36)}-${nextRuleId.toString(36)}`;
}

function createClause(target: string): RuleClause {
  return {
    id: createId("clause"),
    target,
    valueKind: "calibrated",
    comparator: ">=",
    threshold: 0.75,
    upperThreshold: 0.95,
    negated: false,
  };
}

function createGroup(target: string): RuleGroup {
  return {
    id: createId("group"),
    logic: "AND",
    clauses: [createClause(target)],
  };
}

function clampUnit(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export function RuleBuilder({
  groups,
  rootLogic,
  targets,
  defaultTargetId,
  methodLabel,
  onGroupsChange,
  onRootLogicChange,
}: RuleBuilderProps) {
  const defaultTarget = targets.some((target) => target.id === defaultTargetId)
    ? (defaultTargetId ?? "")
    : (targets[0]?.id ?? "");
  const activeClauseCount = groups.reduce((sum, group) => sum + group.clauses.length, 0);

  const updateGroup = (groupId: string, patch: Partial<RuleGroup>) => {
    onGroupsChange(groups.map((group) => (
      group.id === groupId ? { ...group, ...patch } : group
    )));
  };

  const updateClause = (
    groupId: string,
    clauseId: string,
    patch: Partial<RuleClause>,
  ) => {
    onGroupsChange(groups.map((group) => (
      group.id === groupId
        ? {
            ...group,
            clauses: group.clauses.map((clause) => (
              clause.id === clauseId ? { ...clause, ...patch } : clause
            )),
          }
        : group
    )));
  };

  const removeClause = (groupId: string, clauseId: string) => {
    onGroupsChange(groups.flatMap((group) => {
      if (group.id !== groupId) return [group];
      const clauses = group.clauses.filter((clause) => clause.id !== clauseId);
      return clauses.length > 0 ? [{ ...group, clauses }] : [];
    }));
  };

  const addClause = (groupId: string) => {
    onGroupsChange(groups.map((group) => (
      group.id === groupId
        ? { ...group, clauses: [...group.clauses, createClause(defaultTarget)] }
        : group
    )));
  };

  const addCondition = () => {
    if (groups.length === 0) {
      onGroupsChange([createGroup(defaultTarget)]);
      return;
    }
    const lastGroup = groups.at(-1);
    if (lastGroup) addClause(lastGroup.id);
  };

  return (
    <div className="rule-builder">
      <div className="rule-toolbar">
        <div className="rule-method-note">
          <strong>{activeClauseCount}</strong>
          <span>active conditions</span>
          <small>Model outputs: {methodLabel} · no GT labels</small>
        </div>
        <div className="rule-toolbar-actions">
          {groups.length > 1 && (
            <label className="rule-root-logic">
              <span>Groups</span>
              <select
                aria-label="Logic between rule groups"
                value={rootLogic}
                onChange={(event) => onRootLogicChange(event.target.value as RuleLogic)}
              >
                <option value="AND">AND</option>
                <option value="OR">OR</option>
              </select>
            </label>
          )}
          <button type="button" className="rule-action" onClick={addCondition} disabled={!defaultTarget}>
            + Condition
          </button>
          <button
            type="button"
            className="rule-action"
            onClick={() => onGroupsChange([...groups, createGroup(defaultTarget)])}
            disabled={!defaultTarget}
          >
            + Group
          </button>
          <button
            type="button"
            className="text-button rule-reset"
            onClick={() => {
              onGroupsChange([]);
              onRootLogicChange("AND");
            }}
            disabled={groups.length === 0}
          >
            Reset
          </button>
        </div>
      </div>

      {groups.length === 0 ? (
        <div className="rule-empty">
          <strong>No rule filter</strong>
          <span>Add a condition to narrow the current ranked candidates.</span>
        </div>
      ) : (
        <div className="rule-groups">
          {groups.map((group, groupIndex) => (
            <section className="rule-group" key={group.id} aria-label={`Rule group ${groupIndex + 1}`}>
              <header className="rule-group-header">
                <span>Group {groupIndex + 1}</span>
                <label>
                  <span>Match</span>
                  <select
                    aria-label={`Logic within rule group ${groupIndex + 1}`}
                    value={group.logic}
                    onChange={(event) => updateGroup(group.id, { logic: event.target.value as RuleLogic })}
                  >
                    <option value="AND">ALL · AND</option>
                    <option value="OR">ANY · OR</option>
                  </select>
                </label>
                <button
                  type="button"
                  className="rule-icon-button"
                  aria-label={`Remove rule group ${groupIndex + 1}`}
                  title="Remove group"
                  onClick={() => onGroupsChange(groups.filter((candidate) => candidate.id !== group.id))}
                >
                  ×
                </button>
              </header>

              <div className="rule-clauses">
                {group.clauses.map((clause, clauseIndex) => {
                  const usesRange = clause.comparator === "between" || clause.comparator === "outside";
                  return (
                    <div className="rule-clause" key={clause.id}>
                      <div className="rule-clause-index" aria-hidden="true">
                        {clauseIndex > 0 ? group.logic : "IF"}
                      </div>
                      <label className="rule-not-control">
                        <input
                          type="checkbox"
                          checked={clause.negated}
                          onChange={(event) => updateClause(group.id, clause.id, { negated: event.target.checked })}
                        />
                        <span>NOT</span>
                      </label>
                      <label className="rule-field rule-target-field">
                        <span>Target</span>
                        <select
                          value={clause.target}
                          onChange={(event) => updateClause(group.id, clause.id, { target: event.target.value })}
                        >
                          {targets.map((target) => (
                            <option key={target.id} value={target.id}>{target.label}</option>
                          ))}
                        </select>
                      </label>
                      <label className="rule-field">
                        <span>Value</span>
                        <select
                          value={clause.valueKind}
                          onChange={(event) => updateClause(group.id, clause.id, { valueKind: event.target.value as RuleValueKind })}
                        >
                          <option value="calibrated">Calibrated score</option>
                          <option value="rank">Normalized rank</option>
                        </select>
                      </label>
                      <label className="rule-field rule-comparator-field">
                        <span>Test</span>
                        <select
                          value={clause.comparator}
                          onChange={(event) => updateClause(group.id, clause.id, { comparator: event.target.value as RuleComparator })}
                        >
                          <option value=">">&gt;</option>
                          <option value=">=">≥</option>
                          <option value="<">&lt;</option>
                          <option value="<=">≤</option>
                          <option value="between">Between</option>
                          <option value="outside">Outside</option>
                        </select>
                      </label>
                      <label className="rule-field rule-threshold-field">
                        <span>{usesRange ? "Low" : "Threshold"}</span>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.01"
                          value={clause.threshold}
                          onChange={(event) => updateClause(group.id, clause.id, { threshold: clampUnit(Number(event.target.value)) })}
                        />
                      </label>
                      {usesRange && (
                        <label className="rule-field rule-threshold-field">
                          <span>High</span>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.01"
                            value={clause.upperThreshold}
                            onChange={(event) => updateClause(group.id, clause.id, { upperThreshold: clampUnit(Number(event.target.value)) })}
                          />
                        </label>
                      )}
                      <button
                        type="button"
                        className="rule-icon-button rule-clause-remove"
                        aria-label={`Remove condition ${clauseIndex + 1} from group ${groupIndex + 1}`}
                        title="Remove condition"
                        onClick={() => removeClause(group.id, clause.id)}
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
              </div>

              <button type="button" className="rule-add-within" onClick={() => addClause(group.id)}>
                + Add condition to this group
              </button>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

export default RuleBuilder;
