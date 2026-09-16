export interface TextQueryTargetDefinition {
  id: string;
  label: string;
  kind: "attribute" | "derived";
  members?: readonly string[];
}

export interface TextQueryMatch {
  id: string;
  label: string;
}

export type TextQueryResolution =
  | {
      status: "empty";
      matchedAttributes: readonly TextQueryMatch[];
      target: null;
    }
  | {
      status: "no-match" | "unsupported";
      matchedAttributes: readonly TextQueryMatch[];
      target: null;
    }
  | {
      status: "ready";
      matchedAttributes: readonly TextQueryMatch[];
      target: TextQueryMatch;
    };

const TOKEN_GROUPS = [
  ["bicycle", "bicycles", "bike", "bikes", "cycle", "cycling"],
  ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
  ["eyeglasses", "glasses", "spectacles"],
  ["female", "woman", "women"],
  ["male", "man", "men"],
  ["hug", "hugs", "hugged", "hugging", "embrace", "embraces", "embraced", "embracing"],
  ["jump", "jumps", "jumped", "jumping"],
  ["walk", "walks", "walked", "walking"],
  ["shear", "shears", "sheared", "shearing"],
  ["type", "types", "typed", "typing"],
  ["direct", "directs", "directed", "directing"],
  ["sail", "sails", "sailed", "sailing"],
  ["herd", "herds", "herded", "herding"],
  ["train", "trains", "trained", "training"],
  ["race", "races", "raced", "racing"],
  ["read", "reads", "reading"],
  ["cut", "cuts", "cutting"],
  ["grind", "grinds", "ground", "grinding"],
  ["not", "no", "without", "exclude", "excluding"],
] as const;

const CANONICAL_TOKEN = new Map<string, string>();
for (const group of TOKEN_GROUPS) {
  const canonical = group[0];
  for (const token of group) CANONICAL_TOKEN.set(token, canonical);
}

const GENERIC_ID_TOKENS = new Set([
  "attribute",
  "class",
  "contains",
  "name",
  "startswith",
]);

function normalizedTokens(value: string) {
  return value
    .normalize("NFKD")
    .replace(/\p{Mark}+/gu, "")
    .toLocaleLowerCase("en-US")
    .replace(/[^\p{Letter}\p{Number}]+/gu, " ")
    .trim()
    .split(/\s+/u)
    .filter(Boolean)
    .map((token) => CANONICAL_TOKEN.get(token) ?? token);
}

function uniqueAliases(aliases: readonly (readonly string[])[]) {
  const seen = new Set<string>();
  const result: string[][] = [];
  for (const alias of aliases) {
    const normalized = alias.filter(Boolean);
    if (normalized.length === 0) continue;
    const key = normalized.join("\u0000");
    if (seen.has(key)) continue;
    seen.add(key);
    result.push([...normalized]);
  }
  return result;
}

function targetAliases(target: TextQueryTargetDefinition) {
  const idTokens = normalizedTokens(target.id)
    .filter((token) => !GENERIC_ID_TOKENS.has(token));
  const labelTokens = normalizedTokens(target.label);
  const aliases: string[][] = [];
  if (idTokens.length > 0) aliases.push(idTokens);
  if (labelTokens.length > 0) aliases.push(labelTokens);

  // "Wearing earrings" is naturally queried as simply "earrings".
  if (labelTokens[0] === "wearing" && labelTokens.length > 1) {
    aliases.push(labelTokens.slice(1));
  }

  // The exported CelebA target is written as "not Male", while people will
  // commonly describe the same modeled attribute as female/woman.
  if (idTokens.join(" ") === "not male" || labelTokens.join(" ") === "not male") {
    aliases.push(["female"], ["woman"]);
  }

  return uniqueAliases(aliases);
}

function containsAlias(queryTokens: ReadonlySet<string>, alias: readonly string[]) {
  return alias.every((token) => queryTokens.has(token));
}

function sameMembers(left: readonly string[], right: readonly string[]) {
  if (left.length !== right.length) return false;
  const rightSet = new Set(right);
  return left.every((value) => rightSet.has(value));
}

/**
 * Resolve free text only against model outputs already exported for this task.
 * It never invents an unexported multi-attribute score or consults GT labels.
 */
export function resolveTextQuery(
  value: string,
  targets: readonly TextQueryTargetDefinition[],
): TextQueryResolution {
  const queryTokens = new Set(normalizedTokens(value));
  if (queryTokens.size === 0) {
    return { status: "empty", matchedAttributes: [], target: null };
  }

  const attributes = targets.filter((target) => target.kind === "attribute");
  const matchedAttributes = attributes
    .filter((target) => targetAliases(target).some((alias) => containsAlias(queryTokens, alias)))
    .map((target) => ({ id: target.id, label: target.label }));

  if (matchedAttributes.length === 0) {
    return { status: "no-match", matchedAttributes, target: null };
  }

  if (matchedAttributes.length === 1) {
    return {
      status: "ready",
      matchedAttributes,
      target: matchedAttributes[0],
    };
  }

  const matchedIds = matchedAttributes.map((target) => target.id);
  const derived = targets.find((target) => (
    target.kind === "derived"
    && Array.isArray(target.members)
    && sameMembers(target.members, matchedIds)
  ));
  if (!derived) {
    return { status: "unsupported", matchedAttributes, target: null };
  }

  return {
    status: "ready",
    matchedAttributes,
    target: { id: derived.id, label: derived.label },
  };
}
