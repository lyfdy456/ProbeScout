import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import ts from "typescript";
import * as feedbackLabels from "../app/lib/feedbackLabels.ts";
import * as tuningApi from "../app/lib/tuningApi.ts";

// Small hook harness: exercise actual component handlers without a browser or API.
async function fixture() {
  const stores = new Map();
  let current = null;
  let cursor = 0;
  const hooks = {
    ...React,
    useState(initial) {
      const index = cursor++;
      const store = current;
      if (!(index in store)) store[index] = typeof initial === "function" ? initial() : initial;
      return [store[index], (next) => {
        store[index] = typeof next === "function" ? next(store[index]) : next;
      }];
    },
    useRef(initial) {
      const index = cursor++;
      if (!(index in current)) current[index] = { current: initial };
      return current[index];
    },
    useMemo: (factory) => factory(),
    useEffect: () => {},
  };
  function render(component, props, key = "gallery") {
    if (!stores.has(key)) stores.set(key, []);
    current = stores.get(key);
    cursor = 0;
    return component(props);
  }
  const source = await readFile(new URL("../app/components/AnnotationReviewGallery.tsx", import.meta.url), "utf8");
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const componentModule = { exports: {} };
  const imports = {
    react: hooks,
    "react/jsx-runtime": jsxRuntime,
    "../lib/feedbackLabels": feedbackLabels,
    "../lib/tuningApi": tuningApi,
    "./GalleryLightbox": { GalleryLightbox: () => null },
    "./TopGallery": { GalleryThumbnail: () => null, PreferenceControls: () => null },
  };
  new Function("require", "module", "exports", compiled)((id) => {
    assert.ok(id in imports, id);
    return imports[id];
  }, componentModule, componentModule.exports);
  const items = ["first", "second", "positive"].map((id, rowIndex) => ({ id, rowIndex, label: id }));
  const annotations = new Map(items.map((item, index) => [item.id, {
    imageId: item.id, rowIndex: item.rowIndex, label: index === 2 ? 1 : -1,
    suggestedFailedAttributeId: index === 0 ? "a" : "b", updatedAt: "fixture",
  }]));
  const props = {
    annotations, itemById: new Map(items.map((item) => [item.id, item])),
    preferences: new Map(items.map((item, index) => [item.id, index === 2 ? "positive" : "negative"])),
    jointSession: true,
    attributeTargets: [{ id: "a", label: "Attribute A" }, { id: "b", label: "Attribute B" }],
    onPreferenceChange: async () => true,
    onConfirmFailureAttributes: async () => true,
  };
  const gallery = () => render(componentModule.exports.AnnotationReviewGallery, props);
  let tree = gallery();
  nodes(tree).find((node) => node.props?.className === "annotation-review-toggle").props.onClick();
  return { gallery, props, render, imports };
}

function nodes(tree) {
  if (!tree || typeof tree !== "object") return [];
  if (Array.isArray(tree)) return tree.flatMap(nodes);
  return [tree, ...nodes(tree.props?.children)];
}

function button(tree, label) {
  return nodes(tree).find((node) => node.type === "button" && node.props["aria-label"] === label);
}

function selector(tree) {
  return nodes(tree).find((node) => node.type?.name === "FailureAttributeSelector");
}

test("only the robust task offers an explicit relation-only confirmation; it never auto-confirms", async () => {
  const f = await fixture();
  const calls = [];
  f.props.onConfirmFailureAttributes = async (item, ids) => { calls.push([item.id, [...ids]]); return true; };
  const field = () => {
    const selected = selector(f.gallery());
    return f.render(selected.type, selected.props, `${selected.key}:${f.props.taskId ?? "ordinary"}`);
  };
  const relationButton = (tree) => nodes(tree).find((node) => node.type === "button"
    && node.props.children === "Confirm relation mismatch");
  assert.equal(relationButton(field()), undefined);
  f.props.taskId = "032_hico_task_hico_hugging_cat";
  assert.equal(relationButton(field()), undefined);
  f.props.taskId = "059_hico_task_hico_hugging_cat_robust_test";
  const explicit = relationButton(field());
  assert.ok(explicit);
  assert.equal(explicit.props.disabled, false);
  assert.deepEqual(calls, [], "rendering or a model suggestion must never confirm a relation");
  explicit.props.onClick();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(calls, [["first", []]], "relation confirmation clears suggestions, not assigning a failed attribute");
  const previous = f.props.annotations.get("first");
  f.props.annotations.set("first", { ...previous, failedAttributeIds: [], failureAttributionConfirmed: true, updatedAt: "confirmed" });
  const confirmed = field();
  assert.deepEqual(nodes(confirmed).filter((node) => node.type === "input").map((node) => node.props.checked), [false, false]);
  const savedButton = nodes(confirmed).find((node) => node.type === "button" && node.props.children === "Relation mismatch confirmed");
  assert.equal(savedButton.props.disabled, true);
  f.props.canAnnotate = () => false;
  assert.equal(field().props.disabled, true, "held-out rows retain the same disabled editor");
});

test("switching selected feedback keys the sole editor to the image and preserves the full strip", async () => {
  const f = await fixture();
  let tree = f.gallery();
  const first = selector(tree);
  assert.equal(first.props.item.id, "first");
  assert.equal(nodes(tree).filter((node) => node.type === "article").length, 3);
  button(tree, "Select feedback image: second").props.onClick();
  tree = f.gallery();
  const second = selector(tree);
  assert.equal(second.props.item.id, "second");
  assert.notEqual(first.key, second.key, "drafts remount when the selected image changes");
  const field = f.render(second.type, second.props, second.key);
  assert.deepEqual(nodes(field).filter((node) => node.type === "input").map((node) => node.props.checked), [false, true]);
  button(tree, "Select feedback image: positive").props.onClick();
  tree = f.gallery();
  assert.equal(selector(tree), undefined, "positive feedback has no failed-attribute form");
  assert.equal(nodes(tree).filter((node) => node.type === "article").length, 3);
});

test("an in-flight confirmation remains bound to its original image while navigation is allowed", async () => {
  const f = await fixture();
  const calls = [];
  let finish;
  f.props.onConfirmFailureAttributes = (item, ids) => {
    calls.push([item.id, [...ids]]);
    return new Promise((resolve) => { finish = resolve; });
  };
  let tree = f.gallery();
  const first = selector(tree);
  const pending = first.props.onConfirm(first.props.item, ["a"]);
  tree = f.gallery();
  button(tree, "Select feedback image: second").props.onClick();
  tree = f.gallery();
  const second = selector(tree);
  assert.equal(second.props.item.id, "second");
  assert.equal(second.props.busy, true);
  assert.equal(await second.props.onConfirm(second.props.item, ["b"]), false);
  assert.deepEqual(calls, [["first", ["a"]]]);
  finish(true);
  assert.equal(await pending, true);
  tree = f.gallery();
  assert.equal(selector(tree).props.item.id, "second");
  assert.equal(selector(tree).props.busy, false);
  assert.deepEqual(calls, [["first", ["a"]]], "completion cannot save the newly selected row");
});

test("enlarged review owns the only attribution editor and follows the active image", async () => {
  const f = await fixture();
  button(f.gallery(), "Open feedback image: second").props.onClick();
  const tree = f.gallery();
  assert.equal(selector(tree), undefined, "background editor is hidden while the modal owns editing");
  const modal = nodes(tree).find((node) => node.type === f.imports["./GalleryLightbox"].GalleryLightbox);
  assert.equal(modal.props.activeIndex, 1);
  const modalControls = modal.props.renderFeedback(modal.props.items[1]);
  assert.equal(selector(modalControls).props.item.id, "second");
  modal.props.onActiveIndexChange(0);
  const nextModal = nodes(f.gallery()).find((node) => node.type === modal.type);
  assert.equal(nextModal.props.activeIndex, 0);
  nextModal.props.onClose();
  assert.equal(selector(f.gallery()).props.item.id, "first");
});

test("held-out history still permits removal only and never confirms failure attributes", async () => {
  const f = await fixture();
  f.props.canAnnotate = () => false;
  f.props.canRemoveAnnotation = () => true;
  f.props.isValidationRow = () => true;
  let writes = 0;
  f.props.onConfirmFailureAttributes = async () => { writes++; return true; };
  const tree = f.gallery();
  const selected = selector(tree);
  assert.equal(selected.props.busy, true);
  assert.equal(await selected.props.onConfirm(selected.props.item, ["a"]), false);
  assert.equal(writes, 0);
  assert.ok(nodes(tree).some((node) => node.type === "button" && node.props.children === "Remove label"));
});
