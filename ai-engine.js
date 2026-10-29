// PIXORA AI multi-agent pipeline
// Gemini diagnoses and recommends settings; Jimp performs local image processing.
// No hosted legacy upscaler/hosted image service dependency is used.

const fs = require('fs/promises');
const path = require('path');
const JimpModule = require('jimp');
const sharp = require('sharp');
// Jimp v1 uses Promise APIs; older 0.x builds use callbacks. Keep a defensive
// compatibility adapter so a stale deployment cannot fail with `cb must be a function`.
const MODERN_JIMP = Boolean(JimpModule && JimpModule.Jimp);
const Jimp = JimpModule.Jimp || JimpModule.default || JimpModule;
if (!Jimp || typeof Jimp.read !== 'function') {
  throw new Error(
    `Jimp import failed: expected Jimp.read(), got exports: ${Object.keys(JimpModule || {}).join(', ') || 'none'}. ` +
    'Check that package.json installs jimp v1 and redeploy dependencies.'
  );
}

const MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash-lite';
const GEMINI_TIMEOUT_MS = Number(process.env.PIXORA_GEMINI_TIMEOUT_MS || 45000);
const REVIEW_MAX_BYTES = Number(process.env.PIXORA_REVIEW_MAX_BYTES || 1024 * 1024);
const MAX_PIXELS = Number(process.env.PIXORA_MAX_PIXELS || 8000000);
const MAX_DIMENSION = Number(process.env.PIXORA_MAX_DIMENSION || 2600);
const JPEG_QUALITY = Math.max(85, Math.min(98, Number(process.env.PIXORA_JPEG_QUALITY || 95)));

function detectMime(buffer) {
  if (buffer.length >= 8 && buffer.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) return 'image/png';
  if (buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return 'image/jpeg';
  if (buffer.length >= 12 && buffer.toString('ascii', 0, 4) === 'RIFF' && buffer.toString('ascii', 8, 12) === 'WEBP') return 'image/webp';
  return 'application/octet-stream';
}

async function geminiJson(prompt, imageParts) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error('GEMINI_API_KEY is not configured.');
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(MODEL)}:generateContent?key=${encodeURIComponent(key)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }, ...imageParts] }],
      generationConfig: { temperature: 0.15, responseMimeType: 'application/json' }
    }),
    signal: AbortSignal.timeout(GEMINI_TIMEOUT_MS)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.error?.message || `Gemini HTTP ${response.status}`);
  const text = payload?.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('\n').trim();
  try { return JSON.parse(text || '{}'); }
  catch { throw new Error('Gemini returned invalid JSON.'); }
}

async function imagePart(filePath) {
  const buf = await fs.readFile(filePath);
  const mime = detectMime(buf);
  if (!mime.startsWith('image/')) throw new Error('Unsupported image format.');
  return { buffer: buf, part: { inline_data: { mime_type: mime, data: buf.toString('base64') } } };
}

async function diagnoseAgent(inputPath) {
  const { buffer, part } = await imagePart(inputPath);
  const result = await geminiJson(
    'You are PIXORA Diagnostic Agent. Inspect this image for blur, grain/noise, compression artifacts, exposure, contrast, color cast and visible detail. Do not identify people. Return strict JSON: {"issues":[string],"severity":"low|medium|high","blur":"low|medium|high","noise":"low|medium|high","brightness":"dark|normal|bright|uncertain","contrast":"low|normal|high|uncertain","color":"natural|too_warm|too_cool|oversaturated|undersaturated|uncertain","notes":"short"}. Be conservative. Do not claim lost details can be recovered.',
    [part]
  );
  return { status: 'success', model: MODEL, result, inputBytes: buffer.length };
}

async function managerAgent(analysis) {
  const result = await geminiJson(
    'You are PIXORA Enhancement Controller. Jimp is a conventional image-processing library, not a generative super-resolution model. Choose very mild settings only when diagnostics justify them. Return strict JSON: {"brightness":-0.08 to 0.08,"contrast":-0.08 to 0.12,"saturation":-0.08 to 0.08,"sharpen":true|false,"reason":"short"}. Avoid aggressive changes, especially when noise or blur is high. Diagnostic JSON: ' + JSON.stringify(analysis).slice(0, 2500),
    []
  );
  const clamp = (v, min, max, fallback = 0) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
  };
  const settings = {
    brightness: clamp(result.brightness, -0.08, 0.08),
    contrast: clamp(result.contrast, -0.08, 0.12),
    saturation: clamp(result.saturation, -0.08, 0.08),
    // Avoid sharpening if noise/blur is high; sharpening can make grain worse.
    sharpen: result.sharpen === true &&
      !['high'].includes(String(analysis?.noise || '').toLowerCase()) &&
      !['high'].includes(String(analysis?.blur || '').toLowerCase())
  };
  return { status: 'success', result: { ...settings, reason: String(result.reason || 'Conservative image adjustments selected.') }, summary: result.reason || 'Selected conservative Jimp settings.' };
}

// Normalize uploaded images to JPEG before any agent or Jimp processing.
// Sharp supports common formats including JPEG, PNG, WebP, GIF, AVIF, TIFF and SVG.
// The normalized JPEG is written atomically over the temporary upload so every
// downstream stage sees a known, Jimp-compatible format regardless of extension.
async function normalizeInputImage(inputPath) {
  const input = await fs.readFile(inputPath);
  if (!input.length) throw new Error('Uploaded image is empty.');

  const tempPath = `${inputPath}.pixora-normalizing-${process.pid}.jpg`;
  try {
    await sharp(input, { failOn: 'error', limitInputPixels: 40000000 })
      .rotate()
      .flatten({ background: '#ffffff' })
      .jpeg({ quality: JPEG_QUALITY, mozjpeg: true })
      .toFile(tempPath);

    const converted = await fs.readFile(tempPath);
    if (detectMime(converted) !== 'image/jpeg') {
      throw new Error('Image normalization did not produce a valid JPEG.');
    }
    await fs.rename(tempPath, inputPath);
    return {
      format: 'jpeg',
      originalBytes: input.length,
      normalizedBytes: converted.length,
      converted: detectMime(input) !== 'image/jpeg'
    };
  } catch (error) {
    await fs.unlink(tempPath).catch(() => {});
    const message = error?.message || String(error);
    throw new Error(`Image normalization failed: ${message}`);
  }
}

function readJimpImage(inputPath) {
  if (MODERN_JIMP) return Jimp.read(inputPath);
  return new Promise((resolve, reject) => {
    try {
      Jimp.read(inputPath, (error, image) => error ? reject(error) : resolve(image));
    } catch (error) { reject(error); }
  });
}

function resizeJimpImage(image, width, height) {
  if (MODERN_JIMP) return image.resize({ w: width, h: height });
  return image.resize(width, height);
}

function encodeJimpJpeg(image) {
  if (MODERN_JIMP) return image.getBuffer('image/jpeg', { quality: JPEG_QUALITY });
  return new Promise((resolve, reject) => {
    try {
      if (typeof image.quality === 'function') image.quality(JPEG_QUALITY);
      image.getBuffer(Jimp.MIME_JPEG || 'image/jpeg', (error, buffer) => {
        if (error) reject(error);
        else resolve(buffer);
      });
    } catch (error) { reject(error); }
  });
}

async function jimpEnhancer(inputPath, outputPath, analysis, manager) {
  const image = await readJimpImage(inputPath);
  const originalWidth = image.bitmap.width;
  const originalHeight = image.bitmap.height;
  const originalPixels = originalWidth * originalHeight;
  let resizedForMemory = false;

  // Keep decoded RGBA memory bounded on small free hosting plans.
  if (originalPixels > MAX_PIXELS ||
      Math.max(originalWidth, originalHeight) > MAX_DIMENSION) {
    const ratio = Math.min(
      Math.sqrt(MAX_PIXELS / Math.max(1, originalPixels)),
      MAX_DIMENSION / Math.max(originalWidth, originalHeight),
      1
    );
    resizeJimpImage(
      image,
      Math.max(1, Math.round(originalWidth * ratio)),
      Math.max(1, Math.round(originalHeight * ratio))
    );
    resizedForMemory = true;
  }

  const settings = manager?.result || {};
  // Use direct bitmap math rather than optional Jimp plugin methods.
  // This avoids runtime failures when a Jimp build lacks brightness/color plugins.
  const brightness = Math.max(-0.08, Math.min(0.08, Number(settings.brightness || 0)));
  const contrast = Math.max(-0.08, Math.min(0.12, Number(settings.contrast || 0)));
  const saturation = Math.max(-0.08, Math.min(0.08, Number(settings.saturation || 0)));
  const contrastFactor = (259 * (contrast * 255 + 255)) / (255 * (259 - contrast * 255));
  const brightnessShift = brightness * 255;

  if (Math.abs(brightness) >= 0.01 || Math.abs(contrast) >= 0.01 || Math.abs(saturation) >= 0.01) {
    const data = image.bitmap.data;
    for (let i = 0; i < data.length; i += 4) {
      let r = data[i];
      let g = data[i + 1];
      let b = data[i + 2];

      r = contrastFactor * (r - 128) + 128 + brightnessShift;
      g = contrastFactor * (g - 128) + 128 + brightnessShift;
      b = contrastFactor * (b - 128) + 128 + brightnessShift;

      if (Math.abs(saturation) >= 0.01) {
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        const satFactor = 1 + saturation;
        r = gray + (r - gray) * satFactor;
        g = gray + (g - gray) * satFactor;
        b = gray + (b - gray) * satFactor;
      }

      data[i] = Math.max(0, Math.min(255, Math.round(r)));
      data[i + 1] = Math.max(0, Math.min(255, Math.round(g)));
      data[i + 2] = Math.max(0, Math.min(255, Math.round(b)));
      // Preserve alpha channel at data[i + 3].
    }
  }

  // Ensure output directory exists even on ephemeral/restarted hosts.
  await fs.mkdir(path.dirname(outputPath), { recursive: true });

  // Encode explicitly to JPEG, then write bytes ourselves. This avoids relying
  // on Jimp's write(path) behavior and ensures the target file is created before
  // the pipeline tries to stat it.
  const outputBuffer = await encodeJimpJpeg(image);
  if (!Buffer.isBuffer(outputBuffer) || outputBuffer.length === 0) {
    throw new Error('Jimp JPEG encoder returned an empty buffer.');
  }
  await fs.writeFile(outputPath, outputBuffer);

  const stat = await fs.stat(outputPath);
  if (!stat.isFile() || !stat.size) {
    throw new Error(`Jimp output file was not created correctly: ${outputPath}`);
  }

  return {
    status: 'success',
    engine: 'Jimp (local image processing)',
    originalWidth,
    originalHeight,
    width: image.bitmap.width,
    height: image.bitmap.height,
    outputBytes: stat.size,
    resizedForMemory,
    settings
  };
}

async function reviewerAgent(inputPath, outputPath) {
  const inputStat = await fs.stat(inputPath);
  const outputStat = await fs.stat(outputPath);
  if (inputStat.size > REVIEW_MAX_BYTES || outputStat.size > REVIEW_MAX_BYTES) {
    return { status: 'skipped', summary: 'Visual review skipped to reduce memory/API payload; compare before and after manually.', result: null };
  }
  const original = await imagePart(inputPath);
  const enhanced = await imagePart(outputPath);
  const result = await geminiJson(
    'You are PIXORA Quality Reviewer. Compare ORIGINAL and JIMP RESULT. Assess naturalness, visible blur/noise, halos, colors and whether the result is actually better. Do not identify people. Return strict JSON {"verdict":"better|similar|worse|uncertain","original_score":1-10,"result_score":1-10,"issues":[string],"reason":"short"}. First image is ORIGINAL; second is RESULT. Do not assume processing made it better.',
    [original.part, enhanced.part]
  );
  return { status: 'success', result, summary: result.reason || `Reviewer verdict: ${result.verdict || 'uncertain'}.` };
}

async function finalTouchAgent(inputPath, outputPath, review) {
  const verdict = String(review?.result?.verdict || 'uncertain').toLowerCase();
  const scoreA = Number(review?.result?.original_score);
  const scoreB = Number(review?.result?.result_score);
  const clearlyWorse = verdict === 'worse' ||
    (Number.isFinite(scoreA) && Number.isFinite(scoreB) && scoreB + 2 <= scoreA);

  if (clearlyWorse) {
    const head = await fs.open(inputPath, 'r');
    const signature = Buffer.alloc(3);
    try { await head.read(signature, 0, 3, 0); } finally { await head.close(); }
    const isJpeg = signature[0] === 0xff && signature[1] === 0xd8 && signature[2] === 0xff;
    if (isJpeg) {
      await fs.copyFile(inputPath, outputPath);
      return { status: 'fallback_original', summary: 'Reviewer rejected the processed result; original JPEG preserved.', selected: 'original' };
    }
    return { status: 'review_warning', summary: 'Reviewer flagged possible degradation. Processed image retained for manual comparison.', selected: 'processed_review_warning' };
  }
  return {
    status: review?.status === 'skipped' ? 'review_skipped' : 'success',
    summary: review?.status === 'skipped'
      ? 'No automatic visual review was performed; processed image retained for manual comparison.'
      : 'Quality guard accepted the result; no extra destructive filters were applied.',
    selected: 'processed'
  };
}

async function enhanceImage(inputPath, outputPath) {
  // Normalize once up front so diagnostics, enhancement, review and final guard
  // all consume the same supported JPEG representation.
  let normalization;
  try {
    normalization = await normalizeInputImage(inputPath);
  } catch (error) {
    return {
      ok: false,
      analysis: { status: 'skipped', summary: 'Image normalization failed before analysis.' },
      enhancement: { status: 'failed', warning: error.message, summary: 'Input image could not be converted to a supported format.' },
      manager: { status: 'skipped', summary: 'Skipped because image normalization failed.' },
      review: { status: 'skipped', summary: 'Skipped because image normalization failed.' },
      finalTouch: { status: 'skipped', summary: 'Skipped because image normalization failed.' },
      agents: {
        diagnostic: { status: 'skipped', name: 'Diagnostic Agent', summary: 'Skipped because normalization failed.' },
        manager: { status: 'skipped', name: 'Controller Agent', summary: 'Skipped because normalization failed.' },
        enhancer: { status: 'failed', name: 'Jimp Processing Engine', summary: error.message },
        reviewer: { status: 'skipped', name: 'Quality Reviewer', summary: 'Skipped because normalization failed.' },
        finalTouch: { status: 'skipped', name: 'Final Quality Guard', summary: 'Skipped because normalization failed.' }
      },
      notice: 'PIXORA could not normalize this file. Try a standard JPG or PNG image.'
    };
  }

  const agents = {
    diagnostic: { status: 'running', name: 'Diagnostic Agent', summary: 'Inspecting blur, grain, lighting and color.' },
    manager: { status: 'pending', name: 'Controller Agent', summary: 'Choosing conservative image adjustments.' },
    enhancer: { status: 'pending', name: 'Jimp Processing Engine', summary: 'Waiting for diagnostic settings.' },
    reviewer: { status: 'pending', name: 'Quality Reviewer', summary: 'Will compare original and result.' },
    finalTouch: { status: 'pending', name: 'Final Quality Guard', summary: 'Will protect the original if degradation is detected.' }
  };

  let analysis = { status: 'failed', warning: 'Gemini analysis did not run.' };
  let manager = { status: 'fallback', result: { brightness: 0, contrast: 0, saturation: 0, sharpen: false }, summary: 'Neutral settings will be used.' };
  let enhancement = { status: 'failed', summary: 'Jimp has not completed.' };
  let review = { status: 'pending', summary: 'Quality review has not run.' };
  let finalTouch = { status: 'pending', summary: 'Final quality guard has not run.' };

  try {
    analysis = await diagnoseAgent(inputPath);
    agents.diagnostic = { ...agents.diagnostic, status: 'success', summary: `Issues: ${(analysis.result.issues || []).slice(0, 3).join(', ') || 'No major issue reported'}. Severity: ${analysis.result.severity || 'uncertain'}.` };
  } catch (e) {
    analysis = { status: 'failed', warning: e.message, summary: 'Diagnostic failed; neutral processing settings will be used.' };
    agents.diagnostic = { ...agents.diagnostic, status: 'failed', summary: e.message };
  }

  try {
    if (analysis.status === 'success') {
      manager = await managerAgent(analysis.result);
      agents.manager = { ...agents.manager, status: 'success', summary: manager.summary };
    } else {
      manager = { status: 'fallback', result: { brightness: 0, contrast: 0, saturation: 0, sharpen: false }, summary: 'Gemini unavailable; neutral settings selected.' };
      agents.manager = { ...agents.manager, status: 'fallback', summary: manager.summary };
    }
  } catch (e) {
    manager = { status: 'fallback', warning: e.message, result: { brightness: 0, contrast: 0, saturation: 0, sharpen: false }, summary: 'Controller unavailable; neutral settings selected.' };
    agents.manager = { ...agents.manager, status: 'fallback', summary: manager.summary };
  }

  try {
    enhancement = await jimpEnhancer(inputPath, outputPath, analysis, manager);
    enhancement.status = 'success';
    enhancement.summary = `Jimp processing completed at ${enhancement.width}×${enhancement.height}${enhancement.resizedForMemory ? ' (resized to protect memory)' : ''}.`;
    agents.enhancer = { ...agents.enhancer, status: 'success', summary: enhancement.summary };
  } catch (e) {
    enhancement = { status: 'failed', warning: e.message, summary: 'Jimp could not process this image.' };
    agents.enhancer = { ...agents.enhancer, status: 'failed', summary: e.message };
    return {
      ok: false, analysis, enhancement, manager, review, finalTouch, agents, normalization,
      notice: 'Gemini analysis may have completed, but Jimp image processing failed.'
    };
  }

  try {
    review = await reviewerAgent(inputPath, outputPath);
    agents.reviewer = { ...agents.reviewer, status: review.status, summary: review.summary || `Reviewer verdict: ${review.result?.verdict || 'unknown'}.` };
  } catch (e) {
    review = { status: 'failed', warning: e.message, summary: 'Visual review failed; processed output was not automatically rejected.' };
    agents.reviewer = { ...agents.reviewer, status: 'failed', summary: e.message };
  }

  try {
    finalTouch = await finalTouchAgent(inputPath, outputPath, review);
    agents.finalTouch = { ...agents.finalTouch, status: finalTouch.status === 'fallback_original' ? 'fallback' : finalTouch.status === 'success' ? 'success' : 'skipped', summary: finalTouch.summary };
  } catch (e) {
    finalTouch = { status: 'failed', warning: e.message, summary: 'Final quality guard failed; inspect the result manually.' };
    agents.finalTouch = { ...agents.finalTouch, status: 'failed', summary: e.message };
  }

  const outputStat = await fs.stat(outputPath);
  const originalChosen = finalTouch.selected === 'original';
  return {
    ok: true, analysis, manager, enhancement, review, finalTouch, agents, normalization,
    width: enhancement.width || null,
    height: enhancement.height || null,
    engine: originalChosen ? 'PIXORA Quality Guard (original preserved)' : 'Jimp (local image processing)',
    outputBytes: outputStat.size,
    notice: originalChosen
      ? 'The reviewer flagged degradation, so PIXORA preserved the original JPEG.'
      : review.status === 'skipped'
        ? 'Processing completed; visual review was skipped to reduce memory/API payload.'
        : review.result?.verdict === 'better'
          ? 'Multi-agent review accepted the processed result.'
          : 'Processing completed. Compare before and after to confirm the result is preferable.'
  };
}

module.exports = { enhanceImage, analyzeWithGemini: diagnoseAgent };
