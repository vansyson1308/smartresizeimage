import { removeBackground } from "@imgly/background-removal";
import { LayoutRule } from "../types";

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
  
  let renderW, renderH, offsetX, offsetY;

  if (contentRatio > containerRatio) {
    // Content is wider than container: Match height, crop width
    renderH = containerH;
    renderW = contentW * (containerH / contentH);
    offsetX = (containerW - renderW) / 2; // Center horizontally
    offsetY = 0;
  } else {
    // Content is taller than container: Match width, crop height
    renderW = containerW;
    renderH = contentH * (containerW / contentW);
    offsetX = 0;
    offsetY = (containerH - renderH) / 2; // Center vertically
  }
  
  return { w: renderW, h: renderH, x: offsetX, y: offsetY };
};

// --- CORE PIPELINE ---

/**
 * 1. Segments foreground locally using Wasm.
 * 2. Generates background using "Blurred Zoom" technique.
 * 3. Assembles using "Relative Anchoring".
 */
export const processBannerLocally = async (
  originalBase64: string,
  targetW: number,
  targetH: number,
  layoutRule: LayoutRule
): Promise<string> => {

  // 1. Load Original Image
  const originalImg = await loadImage(originalBase64);

  // 2. Perform Segmentation (Remove Background)
  // Note: We perform this once. Ideally we should cache the segmentation if processing multiple sizes of same image.
  // For simplicity in this function, we run it. In production, pass the segmented blob in.
  let foregroundBitmap: ImageBitmap;
  try {
      const blob = await removeBackground(originalBase64, {
          // model: "isnet", // @imgly uses isnet by default which is great for general use
          progress: (key: string, current: number, total: number) => {
              // console.log(`Downloading ${key}: ${current} of ${total}`);
          }
      });
      foregroundBitmap = await createImageBitmap(blob);
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
  // We use the original image as the background source.
  // Step A: Fill background with a heavily blurred version of the original (Edge Padding effect)
  ctx.save();
  ctx.filter = "blur(20px)"; // Heavy blur for abstract background
  // Scale the original to COVER the canvas for the background layer
  const bgGeo = getCoverGeometry(originalImg.width, originalImg.height, targetW, targetH);
  // To avoid white edges from blur, we scale it up slightly more (1.1x)
  ctx.drawImage(
      originalImg, 
      bgGeo.x - (bgGeo.w * 0.05), 
      bgGeo.y - (bgGeo.h * 0.05), 
      bgGeo.w * 1.1, 
      bgGeo.h * 1.1
  );
  ctx.restore();

  // Step B: Optional - Overlay a semi-transparent layer to dampen noise?
  // ctx.fillStyle = "rgba(0,0,0,0.1)";
  // ctx.fillRect(0,0, targetW, targetH);


  // 5. Mascot Anchoring Logic (Celtra Level)
  // Rule: Mascot occupies 60% of banner height.
  // Rule: Maintain Aspect Ratio.
  // Rule: Bottom Center with 5% padding.

  const mascotAspect = foregroundBitmap.width / foregroundBitmap.height;
  
  // Target Height for Mascot
  let mascotH = targetH * 0.60;
  let mascotW = mascotH * mascotAspect;

  // Constraint check: If Mascot becomes wider than the banner (e.g. very wide mascot in skyscraper),
  // constrain by width instead (e.g. 80% width)
  if (mascotW > targetW * 0.9) {
      mascotW = targetW * 0.9;
      mascotH = mascotW / mascotAspect;
  }

  // X Position: Center
  const mascotX = (targetW - mascotW) / 2;

  // Y Position: Bottom with 5% padding
  // 5% of height padding from bottom
  const paddingBottom = targetH * 0.05;
  const mascotY = targetH - mascotH - paddingBottom;

  // 6. Draw Mascot with "Soft Blur" Edges
  ctx.save();
  
  // Edge Softening Hack: Use shadowBlur
  ctx.shadowColor = "rgba(0,0,0,0.3)"; // Subtle shadow to ground it
  ctx.shadowBlur = 10;
  ctx.shadowOffsetY = 5;

  // Draw
  ctx.drawImage(foregroundBitmap, mascotX, mascotY, mascotW, mascotH);
  
  ctx.restore();

  // 7. Output High Quality
  return canvas.toDataURL('image/png', 1.0);
};