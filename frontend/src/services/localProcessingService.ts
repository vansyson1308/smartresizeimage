import { removeBackground } from "@imgly/background-removal";
import { LayoutRule } from "../types";
import { getCachedSegmentation, setCachedSegmentation } from "./segmentationCache";

// --- CONSTANTS ---

/** Blur radius (px) applied to the blurred background layer. */
const BACKGROUND_BLUR_PX = 20;

/** Mascot height as a fraction of the target canvas height. */
const MASCOT_HEIGHT_RATIO = 0.60;

/** Maximum mascot width as a fraction of the target canvas width. */
const MASCOT_MAX_WIDTH_RATIO = 0.90;

/** Padding from the edge (fraction of canvas dimension) for anchored layouts. */
const EDGE_PADDING_RATIO = 0.05;

/** Extra overflow on the blurred background to avoid transparent seams. */
const BG_OVERFLOW_RATIO = 0.05;

// --- HELPERS ---

const loadImage = (src: string): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
};

/**
 * Calculates Crop & Fill geometry (CSS object-fit: cover equivalent)
 */
const getCoverGeometry = (
  contentW: number,
  contentH: number,
  containerW: number,
  containerH: number
) => {
  const contentRatio = contentW / contentH;
  const containerRatio = containerW / containerH;

  let renderW: number, renderH: number, offsetX: number, offsetY: number;

  if (contentRatio > containerRatio) {
    renderH = containerH;
    renderW = contentW * (containerH / contentH);
    offsetX = (containerW - renderW) / 2;
    offsetY = 0;
  } else {
    renderW = containerW;
    renderH = contentH * (containerW / contentW);
    offsetX = 0;
    offsetY = (containerH - renderH) / 2;
  }

  return { w: renderW, h: renderH, x: offsetX, y: offsetY };
};

/**
 * Calculates mascot position based on layout rule.
 */
const getMascotPosition = (
  mascotW: number,
  mascotH: number,
  targetW: number,
  targetH: number,
  layoutRule: LayoutRule
): { x: number; y: number } => {
  const paddingX = targetW * EDGE_PADDING_RATIO;
  const paddingY = targetH * EDGE_PADDING_RATIO;

  switch (layoutRule) {
    case "center":
      return {
        x: (targetW - mascotW) / 2,
        y: (targetH - mascotH) / 2,
      };

    case "bottom-stack":
      return {
        x: (targetW - mascotW) / 2,
        y: targetH - mascotH - paddingY,
      };

    case "left-anchor":
      return {
        x: paddingX,
        y: (targetH - mascotH) / 2,
      };

    case "right-anchor":
      return {
        x: targetW - mascotW - paddingX,
        y: (targetH - mascotH) / 2,
      };

    default:
      // Fallback to bottom-stack
      return {
        x: (targetW - mascotW) / 2,
        y: targetH - mascotH - paddingY,
      };
  }
};

// --- CORE PIPELINE ---

/**
 * 1. Segments foreground locally using Wasm (with caching).
 * 2. Generates background using "Blurred Zoom" technique.
 * 3. Assembles using layout-rule-based anchoring.
 */
export const processBannerLocally = async (
  originalBase64: string,
  targetW: number,
  targetH: number,
  layoutRule: LayoutRule
): Promise<string> => {

  // 1. Load Original Image
  const originalImg = await loadImage(originalBase64);

  // 2. Perform Segmentation with caching
  let foregroundBitmap: ImageBitmap;
  try {
      // Check cache first
      let segBlob = await getCachedSegmentation(originalBase64);
      if (!segBlob) {
          segBlob = await removeBackground(originalBase64, {
              progress: (_key: string, _current: number, _total: number) => {
                  // Progress callback
              }
          });
          await setCachedSegmentation(originalBase64, segBlob);
      }
      foregroundBitmap = await createImageBitmap(segBlob);
  } catch (e) {
      console.error("Segmentation failed, using original as fallback", e);
      foregroundBitmap = await createImageBitmap(originalImg);
  }

  // 3. Setup Canvas
  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not get canvas context");

  // 4. Background Processing: "Center-Crop & Fill" + "Blur"
  ctx.save();
  ctx.filter = `blur(${BACKGROUND_BLUR_PX}px)`;
  const bgGeo = getCoverGeometry(originalImg.width, originalImg.height, targetW, targetH);
  ctx.drawImage(
      originalImg,
      bgGeo.x - (bgGeo.w * BG_OVERFLOW_RATIO),
      bgGeo.y - (bgGeo.h * BG_OVERFLOW_RATIO),
      bgGeo.w * (1 + BG_OVERFLOW_RATIO * 2),
      bgGeo.h * (1 + BG_OVERFLOW_RATIO * 2)
  );
  ctx.restore();

  // 5. Calculate Mascot Size
  const mascotAspect = foregroundBitmap.width / foregroundBitmap.height;

  let mascotH = targetH * MASCOT_HEIGHT_RATIO;
  let mascotW = mascotH * mascotAspect;

  if (mascotW > targetW * MASCOT_MAX_WIDTH_RATIO) {
      mascotW = targetW * MASCOT_MAX_WIDTH_RATIO;
      mascotH = mascotW / mascotAspect;
  }

  // 6. Position Mascot based on layout rule
  const mascotPos = getMascotPosition(mascotW, mascotH, targetW, targetH, layoutRule);

  // 7. Draw Mascot with shadow
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.3)";
  ctx.shadowBlur = 10;
  ctx.shadowOffsetY = 5;
  ctx.drawImage(foregroundBitmap, mascotPos.x, mascotPos.y, mascotW, mascotH);
  ctx.restore();

  // 8. Output
  return canvas.toDataURL('image/png', 1.0);
};
