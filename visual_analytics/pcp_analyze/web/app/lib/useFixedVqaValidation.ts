"use client";

import { useEffect, useState } from "react";
import { fetchFixedVqaValidation, type FixedVqaValidationResponse } from "./tuningApi";
import { fixedVqaValidationMask } from "./fixedVqaValidation";

interface FixedValidationState {
  taskId: string;
  rowCount: number;
  reloadKey?: number;
  status: "loading" | "ready" | "offline";
  mask: Uint8Array | null;
  audit: FixedVqaValidationResponse | null;
  error: string | null;
}

export function useFixedVqaValidation(input: {
  taskId: string;
  rowCount: number;
  enabled: boolean;
  reloadKey?: number;
}): FixedValidationState {
  const { taskId, rowCount, enabled, reloadKey } = input;
  const [state, setState] = useState<FixedValidationState | null>(null);
  useEffect(() => {
    if (!enabled || !taskId || rowCount <= 0) return;
    const controller = new AbortController();
    let active = true;
    queueMicrotask(() => {
      if (active) setState({ taskId, rowCount, reloadKey, status: "loading", mask: null, audit: null, error: null });
    });
    void fetchFixedVqaValidation(taskId, controller.signal)
      .then((response) => {
        if (!active || controller.signal.aborted) return;
        const mask = fixedVqaValidationMask(response, taskId, rowCount);
        setState({ taskId, rowCount, reloadKey, status: "ready", mask, audit: response, error: null });
      })
      .catch((reason: unknown) => {
        if (!active || controller.signal.aborted) return;
        setState({
          taskId, rowCount, reloadKey, status: "offline", mask: null, audit: null,
          error: reason instanceof Error ? reason.message : String(reason),
        });
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [enabled, taskId, rowCount, reloadKey]);

  // A different task must fail closed immediately, before effect cleanup runs.
  if (!enabled || state?.taskId !== taskId || state.rowCount !== rowCount || state.reloadKey !== reloadKey) {
    return { taskId, rowCount, reloadKey, status: "loading", mask: null, audit: null, error: null };
  }
  return state;
}
