"use client";

import type { FormEvent } from "react";
import { methodDisplayLabel } from "../lib/methodDisplay.js";
import type { TuningSessionState } from "../lib/useTuningSession";

interface UserSessionControlsProps {
  state: TuningSessionState;
  targetLabel: string;
  baseMethod: string;
}

export function UserSessionControls({ state, targetLabel, baseMethod }: UserSessionControlsProps) {
  const { serviceState, user, session, busy, saveDisplayName, startNewSession } = state;
  const runInProgress = state.run?.status === "queued" || state.run?.status === "running";
  const sessionId = session?.id;
  const shortSessionId = !sessionId ? "—"
    : sessionId.length > 16 ? `${sessionId.slice(0, 8)}…${sessionId.slice(-5)}` : sessionId;

  const handleNameSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const displayName = String(form.get("displayName") ?? "");
    void saveDisplayName(displayName).catch(() => undefined);
  };

  return (
    <div className="user-session-controls" aria-label="User and session">
      <div className="tuning-identity-row">
        <form className="tuning-identity" onSubmit={handleNameSubmit}>
          <label>
            <span>User</span>
            <input
              key={user?.displayName ?? "anonymous"}
              name="displayName"
              defaultValue={user?.displayName ?? ""}
              placeholder="Your name"
              maxLength={80}
              disabled={serviceState !== "ready" || busy}
            />
          </label>
          <button type="submit" className="button-secondary" disabled={serviceState !== "ready" || busy}>
            Save
          </button>
        </form>
        <button
          type="button"
          className="text-button"
          disabled={serviceState !== "ready" || busy || runInProgress || state.probeUpdateInProgress}
          onClick={() => void startNewSession().catch(() => undefined)}
        >
          New session
        </button>
      </div>
      <div className="tuning-session-meta">
        <span>{user?.identitySource === "openai" ? "Signed-in session" : "Private session"}</span>
        <code title={sessionId}>{shortSessionId}</code>
        <span>{targetLabel} · {methodDisplayLabel(baseMethod)}</span>
      </div>
    </div>
  );
}
