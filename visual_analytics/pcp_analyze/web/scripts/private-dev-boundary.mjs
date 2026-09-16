const PRIVATE_PREFIXES = Object.freeze([
  "/@fs",
  "/runtime",
  "/scripts",
  "/tests",
]);

const PRIVATE_ROOT_FILES = new Set([
  "/cloudflare-env.d.ts",
  "/drizzle.config.ts",
  "/eslint.config.mjs",
  "/next.config.ts",
  "/package-lock.json",
  "/package.json",
  "/postcss.config.mjs",
  "/tsconfig.json",
  "/tsconfig.tsbuildinfo",
  "/vite.config.ts",
]);

function decodePathname(value) {
  let decoded = value;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    } catch {
      return null;
    }
  }
  return decoded.replaceAll("\\", "/").replace(/\/{2,}/g, "/").toLowerCase();
}

export function isPrivateDevRequest(rawUrl) {
  if (typeof rawUrl !== "string" || rawUrl.length === 0) return false;
  let pathname;
  try {
    pathname = new URL(rawUrl, "http://127.0.0.1").pathname;
  } catch {
    return true;
  }
  const normalized = decodePathname(pathname);
  if (normalized === null) return true;
  if (normalized.startsWith("/.")) return true;
  if (PRIVATE_ROOT_FILES.has(normalized)) return true;
  if (
    PRIVATE_PREFIXES.some(
      (prefix) => normalized === prefix || normalized.startsWith(`${prefix}/`),
    )
  ) {
    return true;
  }
  return (
    normalized.includes("/.cookie-secret") ||
    /\.(?:sqlite|sqlite3)(?:-(?:shm|wal))?$/.test(normalized)
  );
}

export function privateDevBoundary() {
  return {
    name: "pcp-private-dev-boundary",
    enforce: "pre",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (!isPrivateDevRequest(request.url)) {
          next();
          return;
        }
        response.statusCode = 404;
        response.setHeader("Cache-Control", "no-store");
        response.setHeader("Content-Type", "text/plain; charset=utf-8");
        response.setHeader("X-Content-Type-Options", "nosniff");
        response.end("Not found\n");
      });
    },
  };
}
