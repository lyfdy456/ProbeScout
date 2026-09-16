import type { Plugin } from "vite";

export function isPrivateDevRequest(rawUrl: unknown): boolean;
export function privateDevBoundary(): Plugin;
