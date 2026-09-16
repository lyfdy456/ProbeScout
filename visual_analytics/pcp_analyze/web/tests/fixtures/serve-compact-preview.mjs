// Run from WEB_ROOT. One loopback-only fixture, no API and no persisted annotations.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { build } from "esbuild";
const { outputFiles } = await build({
  entryPoints: ["tests/fixtures/compact-refinement-preview.tsx"], bundle: true,
  write: false, format: "iife", jsx: "automatic", define: { "process.env.NODE_ENV": '"development"' },
});
const bundle = outputFiles[0].contents;
const server = createServer(async (req, res) => {
  if (req.url === "/") {
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.end('<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic refinement preview</title><link rel="stylesheet" href="/style.css"></head><body><div id="root"></div><script src="/preview.js"></script></body></html>');
  } else if (req.url === "/preview.js") {
    res.setHeader("Content-Type", "text/javascript"); res.end(bundle);
  } else if (req.url === "/style.css") {
    res.setHeader("Content-Type", "text/css");
    res.end((await readFile("app/globals.css", "utf8")).replace('@import "tailwindcss";', ""));
  } else if (["/fonts/inter/inter-latin.woff2", "/fonts/inter/inter-greek.woff2"].includes(req.url)) {
    res.setHeader("Content-Type", "font/woff2"); res.end(await readFile(`public${req.url}`));
  } else { res.writeHead(404); res.end(); }
});
server.listen(0, "127.0.0.1", () => process.stdout.write(`Preview: http://127.0.0.1:${server.address().port}/\n`));
