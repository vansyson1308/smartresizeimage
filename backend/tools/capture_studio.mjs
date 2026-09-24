// Capture docs/demo/studio.jpg from the running studio (real app, real render).
//
//   python -m app.main                                   # in backend/, serves :7860
//   python -c "from backend.tools.make_demo import tech_master; \
//              tech_master().flat().save('/tmp/volt_flash_sale.png')"
//   npm i playwright@1 && node backend/tools/capture_studio.mjs /tmp/volt_flash_sale.png
//
// Writes a 1440x1180 PNG screenshot next to this script's docs/demo folder
// (convert to JPEG before committing to keep the repo small).
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import path from "node:path";

const input = process.argv[2];
const base = process.env.STUDIO_URL || "http://localhost:7860/";
const out = process.argv[3] ||
  path.join(path.dirname(fileURLToPath(import.meta.url)), "../../docs/demo/studio.png");
if (!input) {
  console.error("usage: node capture_studio.mjs <flat-banner.png> [out.png]");
  process.exit(2);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1180 } });
await page.goto(base);
await page.waitForSelector(".preset-item");
await page.setInputFiles("#file-input", input);
await page.waitForSelector("#source-info:not([hidden])");
await page.click('#packs .chip[data-pack="starter"]');  // untick the default pack
await page.click('#packs .chip[data-pack="meta-ads"]');
await page.click('#packs .chip[data-pack="google-display"]');
await page.selectOption("#format", "webp");
await page.click("#generate");
await page.waitForSelector("#download-all:not([hidden])", { timeout: 180000 });
await page.waitForTimeout(1500);
await page.evaluate(() => window.scrollTo(0, 0));
await page.screenshot({ path: out });
console.log(await page.textContent("#summary"), "->", out);
await browser.close();
