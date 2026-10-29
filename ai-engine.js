// PIXORA AI image engine - Android/Termux compatible (Jimp; no sharp)
const fs = require('fs/promises');
const JimpModule = require('jimp');
const Jimp = JimpModule.Jimp || JimpModule;

const DEFAULT_MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash-lite';
const UPSCALE_FACTOR = Math.max(1, Math.min(4, Number(process.env.PIXORA_UPSCALE_FACTOR || 4)));

function detectMime(buffer) {
  if (buffer.length >= 8 && buffer.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) return 'image/png';
  if (buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return 'image/jpeg';
  if (buffer.length >= 12 && buffer.toString('ascii', 0, 4) === 'RIFF' && buffer.toString('ascii', 8, 12) === 'WEBP') return 'image/webp';
  return 'application/octet-stream';
}
function clamp(n, min, max, fallback) {
  n = Number(n);
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
}

async function analyzeWithGemini(inputPath) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('Gemini API key is not configured.');
  const imageBuffer = await fs.readFile(inputPath);
  const mimeType = detectMime(imageBuffer);
  if (!mimeType.startsWith('image/')) throw new Error('Unsupported image format for Gemini analysis.');
  const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + encodeURIComponent(DEFAULT_MODEL) + ':generateContent?key=' + encodeURIComponent(apiKey);
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [
        { text: 'Analyze this image for visible quality issues. Do not identify people. Return JSON only with keys: issues (array), severity (low/medium/high), recommendation (short string), adjustments (object with brightness from -0.2 to 0.2, contrast from -0.2 to 0.2, saturation from -0.5 to 0.5, sharpen from 0 to 1.5). Recommend conservative natural adjustments; use 0 for no change. These settings will be applied by ordinary image filters, not generative AI.' },
        { inline_data: { mime_type: mimeType, data: imageBuffer.toString('base64') } }
      ] }],
      generationConfig: { temperature: 0.2, responseMimeType: 'application/json' }
    }),
    signal: AbortSignal.timeout(45000)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.error?.message || `Gemini request failed with HTTP ${response.status}.`);
  const text = payload?.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('\n').trim();
  let parsed;
  try { parsed = JSON.parse(text || '{}'); } catch { parsed = { recommendation: text || 'Gemini returned no analysis.' }; }
  return { enabled: true, status: 'success', model: DEFAULT_MODEL, result: parsed };
}

function getAdjustmentValues(analysis) {
  const a = analysis?.result?.adjustments || {};
  return {
    brightness: clamp(a.brightness, -0.2, 0.2, 0),
    contrast: clamp(a.contrast, -0.2, 0.2, 0.05),
    saturation: clamp(a.saturation, -0.5, 0.5, 0.08),
    sharpen: clamp(a.sharpen, 0, 1.5, 0.45)
  };
}

async function saveJpeg(image, outputPath) {
  // Jimp 0.22 uses quality()/writeAsync(); newer Jimp versions use getBuffer().
  if (typeof image.quality === 'function' && typeof image.writeAsync === 'function') {
    image.quality(92);
    await image.writeAsync(outputPath);
    return;
  }
  const mime = Jimp.MIME_JPEG || 'image/jpeg';
  const buffer = await image.getBuffer(mime, { quality: 92 });
  await fs.writeFile(outputPath, buffer);
}

async function enhanceWithJimp(inputPath, outputPath, analysis) {
  let image = await Jimp.read(inputPath);
  const originalWidth = image.bitmap.width;
  const originalHeight = image.bitmap.height;
  if (!originalWidth || !originalHeight) throw new Error('Invalid image dimensions.');
  if (originalWidth * originalHeight > 16000000) throw new Error('Image is too large to process safely on this device (over 16 megapixels).');

  const a = getAdjustmentValues(analysis);
  if (a.brightness && typeof image.brightness === 'function') image.brightness(a.brightness);
  if (a.contrast && typeof image.contrast === 'function') image.contrast(a.contrast);
  if (a.saturation && typeof image.color === 'function') image.color([{ apply: 'saturate', params: [a.saturation * 100] }]);
  if (a.sharpen > 0 && typeof image.convolute === 'function') {
    // Conservative 3x3 sharpening kernel; Jimp applies it to RGB channels.
    image.convolute([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]);
  }

  const width = Math.min(8000, Math.round(originalWidth * UPSCALE_FACTOR));
  const height = Math.min(8000, Math.round(originalHeight * UPSCALE_FACTOR));
  if (typeof image.resize === 'function') {
    const resizeMode = Jimp.RESIZE_LANCZOS3 || Jimp.RESIZE_LANCZOS || 'lanczos3';
    image.resize(width, height, resizeMode);
  }
  await saveJpeg(image, outputPath);
  return { width: image.bitmap.width, height: image.bitmap.height, originalWidth, originalHeight, engine: `Jimp filters + Lanczos ${UPSCALE_FACTOR}x (not AI super-resolution)`, adjustments: a };
}

async function enhanceImage(inputPath, outputPath) {
  let analysis = { enabled: false, status: 'not_run', summary: 'Gemini analysis has not run.' };
  try {
    analysis = await analyzeWithGemini(inputPath);
    console.log(`[PIXORA] Gemini analysis completed using ${analysis.model}.`);
  } catch (error) {
    analysis = { enabled: false, status: 'failed', warning: error.message || String(error), summary: 'Gemini analysis unavailable; safe default filters will be used.' };
    console.warn('[PIXORA] Gemini analysis failed:', analysis.warning);
  }

  try {
    const result = await enhanceWithJimp(inputPath, outputPath, analysis);
    const enhancement = { enabled: true, status: 'success', summary: `Image processed with Jimp filters and ${UPSCALE_FACTOR}× Lanczos resizing. This is conventional upscaling, not AI super-resolution.` };
    return { ok: true, analysis, enhancement, ...result, notice: analysis.status === 'success' ? 'Gemini analysis and image filters completed.' : 'Image processed with default filters; Gemini analysis was unavailable.' };
  } catch (error) {
    console.error('[PIXORA] Jimp enhancement failed:', error.stack || error.message);
    return { ok: false, analysis, enhancement: { enabled: false, status: 'failed', warning: error.message || String(error), summary: 'Image processing failed.' }, notice: 'Image processing failed. Check the image size and Termux logs.' };
  }
}

module.exports = { enhanceImage, analyzeWithGemini };
