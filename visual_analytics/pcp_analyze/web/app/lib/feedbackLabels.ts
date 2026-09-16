import type { FeedbackValue } from "./tuningApi";

export type FeedbackLabel =
  | "strong-positive"
  | "positive"
  | "uncertain"
  | "negative"
  | "strong-negative"
  | "unmarked";

export const FEEDBACK_LABELS: ReadonlyArray<{
  value: Exclude<FeedbackLabel, "unmarked">;
  numeric: FeedbackValue;
  glyph: string;
  shortLabel: string;
  label: string;
}> = [
  {
    value: "strong-positive",
    numeric: 2,
    glyph: "++",
    shortLabel: "Strong +",
    label: "Mark as a strong positive",
  },
  {
    value: "positive",
    numeric: 1,
    glyph: "+",
    shortLabel: "Positive",
    label: "Mark as positive",
  },
  {
    value: "uncertain",
    numeric: 0,
    glyph: "?",
    shortLabel: "Unsure",
    label: "Mark as uncertain (not used for training)",
  },
  {
    value: "negative",
    numeric: -1,
    glyph: "−",
    shortLabel: "Negative",
    label: "Mark as negative",
  },
  {
    value: "strong-negative",
    numeric: -2,
    glyph: "−−",
    shortLabel: "Strong −",
    label: "Mark as a strong negative",
  },
];

export function feedbackValue(label: FeedbackLabel): FeedbackValue | null {
  if (label === "unmarked") return null;
  return FEEDBACK_LABELS.find((entry) => entry.value === label)?.numeric ?? null;
}

export function feedbackLabel(value: FeedbackValue): Exclude<FeedbackLabel, "unmarked"> {
  const match = FEEDBACK_LABELS.find((entry) => entry.numeric === value);
  if (!match) throw new RangeError(`Unsupported feedback value: ${value}`);
  return match.value;
}

export function feedbackGlyph(label: FeedbackLabel) {
  return FEEDBACK_LABELS.find((entry) => entry.value === label)?.glyph ?? "";
}

