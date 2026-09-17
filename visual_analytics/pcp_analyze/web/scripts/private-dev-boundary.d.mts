import type { Plugin } from "vite";

export function isPrivateDevRequest(rawUrl: unknown, publicModuleUrls?: string[]): boolean;
export function privateDevBoundary(publicModuleUrls?: string[]): Plugin;
