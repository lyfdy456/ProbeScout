import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { resolveTextQuery } from "../app/lib/textQuery.ts";

const HUGGING_TARGETS = [
  { id: "cat", label: "Cat", kind: "attribute" },
  { id: "hugging", label: "Hugging", kind: "attribute" },
  { id: "joint", label: "Joint", kind: "derived", members: ["cat", "hugging"] },
];

test("text Query resolves one modeled attribute or the exact exported Joint", () => {
  assert.deepEqual(resolveTextQuery("CAT!!!", HUGGING_TARGETS), {
    status: "ready",
    matchedAttributes: [{ id: "cat", label: "Cat" }],
    target: { id: "cat", label: "Cat" },
  });
  assert.deepEqual(resolveTextQuery("a person hugging a cat", HUGGING_TARGETS), {
    status: "ready",
    matchedAttributes: [
      { id: "cat", label: "Cat" },
      { id: "hugging", label: "Hugging" },
    ],
    target: { id: "joint", label: "Joint" },
  });
});

test("text Query keeps unknown and unexported combinations out of retrieval state", () => {
  assert.equal(resolveTextQuery("   ", HUGGING_TARGETS).status, "empty");
  assert.equal(resolveTextQuery("purple umbrella", HUGGING_TARGETS).status, "no-match");

  const celebaTargets = [
    { id: "gray_hair", label: "Gray_Hair", kind: "attribute" },
    { id: "eyeglasses", label: "Eyeglasses", kind: "attribute" },
    { id: "male", label: "Male", kind: "attribute" },
    {
      id: "joint",
      label: "Joint",
      kind: "derived",
      members: ["gray_hair", "eyeglasses", "male"],
    },
  ];
  const partial = resolveTextQuery("gray hair with eyeglasses", celebaTargets);
  assert.equal(partial.status, "unsupported");
  assert.deepEqual(partial.matchedAttributes.map((target) => target.id), [
    "gray_hair",
    "eyeglasses",
  ]);
  assert.equal(resolveTextQuery("female", celebaTargets).status, "no-match");
});

test("text Query handles exported negation and common task wording without substring errors", () => {
  const targets = [
    { id: "bicycle", label: "bicycle", kind: "attribute" },
    { id: "not_jumping", label: "not jumping", kind: "attribute" },
    { id: "not_bike_jump", label: "not bike jump", kind: "attribute" },
    {
      id: "joint",
      label: "Joint",
      kind: "derived",
      members: ["bicycle", "not_jumping", "not_bike_jump"],
    },
  ];
  assert.equal(resolveTextQuery("a bike without jumping", targets).target?.id, "joint");

  const genderTargets = [
    { id: "not_male", label: "not Male", kind: "attribute" },
    { id: "joint", label: "Joint", kind: "derived", members: ["not_male"] },
  ];
  assert.equal(resolveTextQuery("female portrait", genderTargets).target?.id, "not_male");
  assert.equal(resolveTextQuery("male portrait", genderTargets).status, "no-match");
});


test("text Query component remains compatible without mounting in the Dashboard", async () => {
  const [dashboard, component] = await Promise.all([
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/TextQueryControl.tsx", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(dashboard, /TextQueryControl/);
  assert.match(component, /<form className="text-query-form" onSubmit=\{submit\}>/);
  assert.match(component, /aria-label="Text query"/);
  assert.match(component, /maxLength=\{280\}/);
  assert.match(component, /type="submit"[\s\S]*?disabled=\{resolution\.status !== "ready"\}/);
  assert.match(component, /onTargetChange\(resolution\.target\.id\)/);
  assert.doesNotMatch(component, /dangerouslySetInnerHTML|type="file"|FileReader/);
});
