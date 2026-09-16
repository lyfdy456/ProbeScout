"use client";

import { useMemo, useState, type FormEvent } from "react";
import {
  resolveTextQuery,
  type TextQueryTargetDefinition,
} from "../lib/textQuery";

interface TextQueryControlProps {
  targets: readonly TextQueryTargetDefinition[];
  activeTargetId: string;
  example?: string;
  onTargetChange: (targetId: string) => void;
}

export function TextQueryControl({
  targets,
  activeTargetId,
  example,
  onTargetChange,
}: TextQueryControlProps) {
  const [value, setValue] = useState("");
  const resolution = useMemo(() => resolveTextQuery(value, targets), [targets, value]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (resolution.status !== "ready") return;
    onTargetChange(resolution.target.id);
  };

  return (
    <form className="text-query-form" onSubmit={submit}>
      <label htmlFor="text-query-input">Text query</label>
      <div className="text-query-controls">
        <input
          id="text-query-input"
          aria-label="Text query"
          type="text"
          value={value}
          maxLength={280}
          placeholder={example ? `e.g. ${example}` : "Describe the target attributes"}
          onChange={(event) => setValue(event.target.value)}
        />
        <button
          type="submit"
          className="button-primary"
          disabled={resolution.status !== "ready"}
        >
          Retrieve
        </button>
      </div>

      {resolution.status !== "empty" && (
        <div
          className={`text-query-resolution text-query-resolution--${resolution.status}`}
          role={resolution.status === "ready" ? "status" : "alert"}
          aria-live="polite"
        >
          {resolution.matchedAttributes.length > 0 && (
            <span className="text-query-matches">
              {resolution.matchedAttributes.map((attribute) => (
                <i key={attribute.id}>{attribute.label}</i>
              ))}
            </span>
          )}
          {resolution.status === "ready" ? (
            <strong>
              {activeTargetId === resolution.target.id ? "Active" : "Use"}
              {" · "}{resolution.target.label}
            </strong>
          ) : resolution.status === "no-match" ? (
            <strong>No modeled attribute found.</strong>
          ) : (
            <strong>This attribute combination has no exported ranking.</strong>
          )}
        </div>
      )}
    </form>
  );
}
