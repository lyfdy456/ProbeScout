"use client";

import type { TuningSessionState } from "../lib/useTuningSession";
import { compatibleProbeUpdates } from "../lib/tuningApi";

interface ProbeSourceControlsProps {
  state: TuningSessionState;
  taskId: string;
  targetId: string;
  baseMethod: string;
  validationReady?: boolean;
}

export function ProbeSourceControls({
  state, taskId, targetId, baseMethod, validationReady = false,
}: ProbeSourceControlsProps) {
  const runInProgress = state.run?.status === "queued" || state.run?.status === "running";
  const annotationsPending = state.pendingAnnotationCount > 0;
  const sessionMatches = Boolean(
    state.session
    && state.session.taskId === taskId
    && state.session.targetId === targetId
    && state.session.baseMethod === baseMethod,
  );
  const commonDisabled = (
    state.busy
    || annotationsPending
    || runInProgress
    || state.probeUpdateInProgress
    || state.serviceState !== "ready"
    || !validationReady
    || !sessionMatches
  );
  const availableUpdates = state.session ? compatibleProbeUpdates(state.probeUpdates, state.session.id, taskId) : [];
  const latestUpdate = state.probeUpdates[0];
  const chosenUpdate = state.selectedProbeUpdate;
  const missingSnapshot = state.probeSource === "updated" && !chosenUpdate;

  return (
    <div className="function-probe-source">
      <label>
        <span>Probe source</span>
        <select aria-label="Probe source" value={state.probeSource}
          disabled={commonDisabled}
          onChange={(event) => state.selectProbeSource(event.target.value === "updated" ? "updated" : "original")}
        >
          <option value="original">Original · 原始</option>
          <option value="updated" disabled={!availableUpdates.length}>Updated · 更新后</option>
        </select>
      </label>
      {state.probeSource === "updated" && (availableUpdates.length > 1 || missingSnapshot && availableUpdates.length > 0) && (
        <select aria-label="Updated Probe snapshot" value={chosenUpdate?.id ?? ""} disabled={commonDisabled}
          onChange={(event) => state.selectProbeSource("updated", event.target.value)}>
          {!chosenUpdate && <option value="" disabled>Choose snapshot</option>}
          {availableUpdates.map((job) => <option key={job.id} value={job.id}>
            {new Date(job.createdAt).toLocaleString()} · {job.id.slice(-8)}
          </option>)}
        </select>
      )}
      {chosenUpdate && <span title={chosenUpdate.snapshotId ?? chosenUpdate.id}>
        Frozen · {chosenUpdate.id.slice(-8)}
        {chosenUpdate.stale ? " · Feedback changed; update again if needed." : ""}
      </span>}
      {missingSnapshot && <span role="alert">Snapshot unavailable; choose a source.</span>}
      {latestUpdate && <span aria-live="polite" title={latestUpdate.error ?? latestUpdate.snapshotId ?? latestUpdate.id}>
        {latestUpdate.status === "queued" ? "Probe update queued…"
          : latestUpdate.status === "running" ? "Updating Probes…"
          : latestUpdate.status === "failed" ? `Probe update failed: ${latestUpdate.error ?? "Unknown error"}`
          : `Snapshot saved · ${latestUpdate.id.slice(-8)}${state.probeSource === "original" ? " · choose Updated to use" : ""}`}
      </span>}
    </div>
  );
}
