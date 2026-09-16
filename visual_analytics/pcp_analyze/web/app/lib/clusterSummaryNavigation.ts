export type ClusterSummaryNavigationId = string | number;
export type ClusterSummaryNavigationDirection = -1 | 1;

export interface ClusterSummaryNavigationProfile {
  id: ClusterSummaryNavigationId;
  means: ArrayLike<number>;
}

export interface ClusterSummaryNavigationEntry<
  Profile extends ClusterSummaryNavigationProfile = ClusterSummaryNavigationProfile,
> {
  profile: Profile;
  key: string;
  mean: number | null;
  sourceIndex: number;
}

export interface ClusterSummaryNavigation<
  Profile extends ClusterSummaryNavigationProfile = ClusterSummaryNavigationProfile,
> {
  entries: Array<ClusterSummaryNavigationEntry<Profile>>;
  indexByKey: ReadonlyMap<string, number>;
}

function clusterKey(cluster: ClusterSummaryNavigationId) {
  return String(cluster);
}

function numericClusterId(cluster: ClusterSummaryNavigationId) {
  if (typeof cluster === "number") return Number.isFinite(cluster) ? cluster : null;
  const trimmed = cluster.trim();
  if (trimmed.length === 0) return null;
  const numeric = Number(trimmed);
  return Number.isFinite(numeric) ? numeric : null;
}

/** Arithmetic mean over the requested finite values; null means no finite value exists. */
export function finiteClusterProfileMean(
  values: ArrayLike<number>,
  activeIndices: readonly number[],
): number | null {
  let sum = 0;
  let count = 0;
  for (const index of activeIndices) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    sum += value;
    count += 1;
  }
  return count > 0 ? sum / count : null;
}

/**
 * Builds the left-to-right cluster order for the currently visible PCP axes.
 * Finite means sort ascending. Profiles without a finite mean follow them.
 * Equal means use numeric cluster IDs when available, then retain source order.
 */
export function buildClusterSummaryNavigation<Profile extends ClusterSummaryNavigationProfile>(
  profiles: readonly Profile[],
  activeIndices: readonly number[],
): ClusterSummaryNavigation<Profile> {
  const entries = profiles.map((profile, sourceIndex) => ({
    profile,
    key: clusterKey(profile.id),
    mean: finiteClusterProfileMean(profile.means, activeIndices),
    sourceIndex,
  }));

  entries.sort((left, right) => {
    const leftFinite = left.mean !== null;
    const rightFinite = right.mean !== null;
    if (leftFinite !== rightFinite) return leftFinite ? -1 : 1;
    if (left.mean !== null && right.mean !== null && left.mean !== right.mean) {
      return left.mean - right.mean;
    }

    const leftNumericId = numericClusterId(left.profile.id);
    const rightNumericId = numericClusterId(right.profile.id);
    if (leftNumericId !== null && rightNumericId !== null && leftNumericId !== rightNumericId) {
      return leftNumericId - rightNumericId;
    }
    if ((leftNumericId !== null) !== (rightNumericId !== null)) {
      return leftNumericId !== null ? -1 : 1;
    }
    return left.sourceIndex - right.sourceIndex;
  });

  return {
    entries,
    indexByKey: new Map(entries.map((entry, index) => [entry.key, index])),
  };
}

/** Returns the neighboring cluster, or null when movement would cross an edge. */
export function moveClusterSummarySelection(
  navigation: ClusterSummaryNavigation,
  selectedCluster: ClusterSummaryNavigationId | null | undefined,
  direction: ClusterSummaryNavigationDirection,
): ClusterSummaryNavigationId | null {
  if (navigation.entries.length === 0) return null;
  if (selectedCluster === null || selectedCluster === undefined) {
    return direction > 0
      ? navigation.entries[0].profile.id
      : navigation.entries[navigation.entries.length - 1].profile.id;
  }

  const currentIndex = navigation.indexByKey.get(clusterKey(selectedCluster));
  if (currentIndex === undefined) {
    return direction > 0
      ? navigation.entries[0].profile.id
      : navigation.entries[navigation.entries.length - 1].profile.id;
  }

  const nextIndex = currentIndex + direction;
  if (nextIndex < 0 || nextIndex >= navigation.entries.length) return null;
  return navigation.entries[nextIndex].profile.id;
}

/** Maps only the four supported, unmodified key values to navigation direction. */
export function clusterSummaryDirectionForKey(
  key: string,
): ClusterSummaryNavigationDirection | null {
  if (key === "ArrowLeft" || key.toLowerCase() === "a") return -1;
  if (key === "ArrowRight" || key.toLowerCase() === "d") return 1;
  return null;
}
