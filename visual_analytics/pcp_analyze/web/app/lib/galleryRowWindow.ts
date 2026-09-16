export interface GalleryCardBounds {
  top: number;
  height: number;
}

/** Size a window from laid-out rows, including unequal card heights and real row gaps. */
export function galleryRowWindowHeight(
  cards: readonly GalleryCardBounds[],
  rowLimit = 2,
  paddingTop = 0,
  paddingBottom = 0,
): number | null {
  if (!Number.isFinite(rowLimit) || rowLimit < 1) return null;
  const limit = Math.floor(rowLimit);
  const bounds = cards
    .filter((card) => Number.isFinite(card.top) && Number.isFinite(card.height) && card.height > 0)
    .sort((left, right) => left.top - right.top);
  if (bounds.length === 0) return null;

  const rows: Array<{ top: number; bottom: number }> = [];
  for (const card of bounds) {
    const last = rows.at(-1);
    // offsetTop is pixel-rounded; tolerate subpixel differences in alternate measurements.
    if (last && Math.abs(card.top - last.top) <= 1) {
      last.bottom = Math.max(last.bottom, card.top + card.height);
    } else {
      rows.push({ top: card.top, bottom: card.top + card.height });
    }
  }
  const lastVisible = rows[Math.min(limit, rows.length) - 1];
  const nextRow = rows[limit];
  const topInset = Number.isFinite(paddingTop) ? Math.max(0, paddingTop) : 0;
  const bottomInset = Number.isFinite(paddingBottom) ? Math.max(0, paddingBottom) : 0;
  // Do not expose part of the next row if padding exceeds its inter-row gap.
  const visibleBottomInset = nextRow
    ? Math.min(bottomInset, Math.max(0, nextRow.top - lastVisible.bottom))
    : bottomInset;
  return topInset + lastVisible.bottom - rows[0].top + visibleBottomInset;
}
