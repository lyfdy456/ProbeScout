"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  initialBaselineSourceKey, requestInitialBaselineClusters,
  requestInitialBaselineManifest, requestInitialBaselineVisualization,
  type InitialBaselineData,
} from "./initialBaseline";

export function useInitialBaseline(input: {
  taskId: string; rowCount: number; attributeIds: readonly string[];
  scheme: string; enabled: boolean; reloadKey: number;
}) {
  const taskKey = `${input.taskId}|${input.rowCount}|${input.attributeIds.join("|")}|${input.reloadKey}`;
  const [state, setState] = useState<{
    taskKey: string; status: "idle" | "loading" | "pending" | "ready" | "error";
    data: InitialBaselineData | null; error: string | null;
  }>({ taskKey, status: "idle", data: null, error: null });
  const generation = useRef(0);
  const schemeGeneration = useRef(0);
  const latest = useRef(input);
  useEffect(() => { latest.current = input; }, [input]);
  const current = input.enabled && state.taskKey === taskKey ? state.data : null;
  useEffect(() => {
    const requestGeneration = ++generation.current;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      if (!latest.current.enabled) {
        setState({ taskKey, status: "idle", data: null, error: null });
        return;
      }
      const request = { ...latest.current, signal: controller.signal };
      setState({ taskKey, status: "loading", data: null, error: null });
      try {
        const manifest = await requestInitialBaselineManifest(request);
        if (requestGeneration !== generation.current) return;
        if (!manifest) { setState({ taskKey, status: "pending", data: null, error: null }); return; }
        const snapshot = await requestInitialBaselineVisualization(manifest, controller.signal);
        const clusters = await requestInitialBaselineClusters(manifest, snapshot, latest.current.scheme, controller.signal);
        if (requestGeneration !== generation.current) return;
        setState({ taskKey, status: "ready", data: { manifest, snapshot, clusters }, error: null });
      } catch (reason) {
        if (requestGeneration !== generation.current || controller.signal.aborted) return;
        setState({ taskKey, status: "error", data: null, error: reason instanceof Error ? reason.message : String(reason) });
      }
    }, 0);
    return () => { generation.current += 1; controller.abort(); window.clearTimeout(timer); };
  }, [input.enabled, taskKey]);

  const prepareScheme = useCallback(async (scheme: string) => {
    if (!current) return null;
    const requestGeneration = generation.current;
    const requestSchemeGeneration = ++schemeGeneration.current;
    const clusters = await requestInitialBaselineClusters(current.manifest, current.snapshot, scheme);
    if (requestGeneration !== generation.current || requestSchemeGeneration !== schemeGeneration.current) return null;
    setState({ taskKey, status: "ready", data: { ...current, clusters }, error: null });
    return clusters;
  }, [current, taskKey]);
  useEffect(() => {
    if (!current || current.clusters.scheme === input.scheme) return;
    const timer = window.setTimeout(() => {
      void prepareScheme(input.scheme).catch((reason) => {
        setState((previous) => previous.taskKey === taskKey ? {
          ...previous, error: reason instanceof Error ? reason.message : String(reason),
        } : previous);
      });
    }, 0);
    return () => window.clearTimeout(timer);
  }, [current, input.scheme, prepareScheme, taskKey]);
  return {
    status: input.enabled && state.taskKey !== taskKey ? "loading" as const : state.status,
    data: current,
    error: state.taskKey === taskKey ? state.error : null,
    sourceKey: current ? initialBaselineSourceKey(current.manifest, input.scheme) : `initial|${taskKey}|pending`,
    prepareScheme,
  };
}
