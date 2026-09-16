"use client";

import { useEffect, useState } from "react";
import {
  fetchOriginalVqaSupervision,
  type OriginalVqaSupervisionItem,
} from "./tuningApi";

export type OriginalVqaSupervisionStatus = "idle" | "loading" | "ready" | "offline";

export interface OriginalVqaSupervisionState {
  status: OriginalVqaSupervisionStatus;
  taskId: string;
  targetId: string;
  recordsByRow: ReadonlyMap<number, OriginalVqaSupervisionItem>;
  selectedCount: number;
  developmentCount: number;
  error: string | null;
}

const EMPTY_RECORDS: ReadonlyMap<number, OriginalVqaSupervisionItem> = new Map();

export function useOriginalVqaSupervision(input: {
  enabled: boolean;
  taskId: string;
  targetId: string;
  rowCount: number;
}): OriginalVqaSupervisionState {
  const { enabled, taskId, targetId, rowCount } = input;
  const [state, setState] = useState<OriginalVqaSupervisionState>({
    status: "idle",
    taskId: "",
    targetId: "",
    recordsByRow: EMPTY_RECORDS,
    selectedCount: 0,
    developmentCount: 0,
    error: null,
  });

  useEffect(() => {
    let active = true;
    if (!enabled || !taskId || !targetId || rowCount <= 0) {
      queueMicrotask(() => {
        if (!active) return;
        setState({
          status: "idle",
          taskId: "",
          targetId: "",
          recordsByRow: EMPTY_RECORDS,
          selectedCount: 0,
          developmentCount: 0,
          error: null,
        });
      });
      return () => {
        active = false;
      };
    }

    const controller = new AbortController();
    queueMicrotask(() => {
      if (!active) return;
      setState({
        status: "loading",
        taskId,
        targetId,
        recordsByRow: EMPTY_RECORDS,
        selectedCount: 0,
        developmentCount: 0,
        error: null,
      });
    });
    void fetchOriginalVqaSupervision(taskId, targetId, controller.signal)
      .then((response) => {
        if (!active || controller.signal.aborted) return;
        if (response.taskId !== taskId || response.targetId !== targetId) {
          throw new Error("Original VQA supervision identity does not match the active task.");
        }
        if (response.selectedCount !== response.items.length) {
          throw new Error("Original VQA supervision count does not match its payload.");
        }
        const records = new Map<number, OriginalVqaSupervisionItem>();
        let developmentCount = 0;
        for (const item of response.items) {
          if (
            !Number.isInteger(item.rowIndex)
            || item.rowIndex < 0
            || item.rowIndex >= rowCount
            || !item.imageId
            || item.label !== 0 && item.label !== 1
            || records.has(item.rowIndex)
          ) {
            throw new Error("Original VQA supervision contains an invalid row.");
          }
          if (item.trainableInDevelopment) developmentCount += 1;
          records.set(item.rowIndex, item);
        }
        if (response.developmentCount !== developmentCount) {
          throw new Error("Original VQA Development count does not match its payload.");
        }
        setState({
          status: "ready",
          taskId,
          targetId,
          recordsByRow: records,
          selectedCount: response.selectedCount,
          developmentCount,
          error: null,
        });
      })
      .catch((reason: unknown) => {
        if (!active || controller.signal.aborted) return;
        setState({
          status: "offline",
          taskId,
          targetId,
          recordsByRow: EMPTY_RECORDS,
          selectedCount: 0,
          developmentCount: 0,
          error: reason instanceof Error ? reason.message : String(reason),
        });
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [enabled, rowCount, targetId, taskId]);

  return state;
}

export default useOriginalVqaSupervision;
