/**
 * Segmentation result cache.
 *
 * Caches background removal results keyed by a hash of the source image,
 * so that processing multiple output sizes from the same source only
 * runs the expensive WASM segmentation once.
 *
 * Limited to MAX_CACHE_ENTRIES to prevent unbounded memory growth.
 * Uses FIFO eviction based on Map insertion order.
 */

/** Maximum number of cached segmentation results. */
const MAX_CACHE_ENTRIES = 20;

const cache = new Map<string, Blob>();

async function hashImage(base64: string): Promise<string> {
  // Hash first 10KB for speed (enough for uniqueness)
  const encoder = new TextEncoder();
  const data = encoder.encode(base64.slice(0, 10000));
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(hashBuffer))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

export async function getCachedSegmentation(base64: string): Promise<Blob | null> {
  const key = await hashImage(base64);
  return cache.get(key) ?? null;
}

export async function setCachedSegmentation(base64: string, blob: Blob): Promise<void> {
  const key = await hashImage(base64);

  // Evict oldest entry if cache is full (FIFO via Map insertion order)
  if (cache.size >= MAX_CACHE_ENTRIES && !cache.has(key)) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) {
      cache.delete(oldestKey);
    }
  }

  cache.set(key, blob);
}

export function clearSegmentationCache(): void {
  cache.clear();
}
