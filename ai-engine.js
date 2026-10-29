// PIXORA AI multi-agent pipeline
// Gemini diagnostic + controller + visual reviewer; Real-ESRGAN performs actual pixel enhancement.
// The final quality guard selects the enhanced image only when review does not flag degradation.

const fs = require('fs/promises');
const path = require('path');
const { createWriteStream } = require('fs');
const { pipeline } = require('stream/promises');
const { Readable } = require('stream');

const SPACE = process.env.PIXORA_ESRGAN_SPACE || 'Nick088/Real-ESRGAN_Pytorch';
const MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash-lite';
const DEFAULT_SCALE = Number(process.env.PIXORA_ESRGAN_SCALE || 4);
const GEMINI_TIMEOUT_MS = Number(process.env.PIXORA_GEMINI_TIMEOUT_MS || 45000);
const REVIEW_MAX_BYTES = Number(process.env.PIXORA_REVIEW_MAX_BYTES || 5 * 1024 * 1024);

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
    'You are PIXORA Diagnostic Agent. Inspect the image for blur, noise/grain, compression artifacts, lighting, color cast, oversaturation, and visible detail. Do not identify people. Return strict JSON: {"issues":[string],"severity":"low|medium|high","blur":"low|medium|high","noise":"low|medium|high","color":"natural|too_warm|too_cool|oversaturated|undersaturated|uncertain","recommended_scale":2|4,"notes":"short"}. Be conservative; do not claim detail can be recovered if it is absent.',
    [part]
  );
  return { status: 'success', model: MODEL, result, inputBytes: buffer.length };
}

async function managerAgent(analysis) {
  // Keep the controller call small: it receives only the diagnostic JSON, not the image.
  const result = await geminiJson(
    'You are PIXORA Controller Agent. Coordinate the enhancement workflow from this diagnostic JSON. Real-ESRGAN is an upscaler, not a reliable color-correction or denoising tool. Choose scale 2 or 4 conservatively: choose 2 for high blur/noise/severity or uncertain diagnosis; choose 4 only for low/medium issues when the source is small/clean. Return JSON {"scale":2|4,"reason":"short","review_required":true,"preserve_natural_texture":true}. Diagnostic: ' + JSON.stringify(analysis).slice(0, 3500),
    []
  );
  const requested = Number(result.scale);
  const scale = requested === 2 || requested === 4 ? requested : (analysis?.severity === 'high' ? 2 : DEFAULT_SCALE === 2 ? 2 : 4);
  return { status: 'success', result: { ...result, scale }, summary: result.reason || `Selected conservative ${scale}× upscaling.` };
}

function findOutputFile(data) {
  if (!data) return null;
  if (typeof data === 'string') return { path: data };
  if (Array.isArray(data)) { for (const item of data) { const f = findOutputFile(item); if (f) return f; } return null; }
  if (typeof data === 'object') {
    if (data.url || data.path || data.name) return data;
    for (const value of Object.values(data)) { const f = findOutputFile(value); if (f) return f; }
  }
  return null;
}

async function downloadGradioOutput(fileData, space, outputPath) {
  if (fileData.path && path.isAbsolute(fileData.path)) {
    try { await fs.access(fileData.path); await fs.copyFile(fileData.path, outputPath); return; } catch {}
  }
  const base = `https://${space.toLowerCase().replace('/', '-')}.hf.space`;
  let fileUrl = fileData.url;
  if (!fileUrl && fileData.path) fileUrl = '/gradio_api/file=' + encodeURIComponent(fileData.path);
  if (!fileUrl && fileData.name) fileUrl = '/gradio_api/file=' + encodeURIComponent(fileData.name);
  if (!fileUrl) throw new Error('Real-ESRGAN returned no downloadable output file.');
  if (fileUrl.startsWith('/')) fileUrl = base + fileUrl;
  const response = await fetch(fileUrl, { signal: AbortSignal.timeout(150000) });
  if (!response.ok || !response.body) throw new Error(`Downloading enhanced image failed: HTTP ${response.status}.`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(outputPath));
  const stat = await fs.stat(outputPath);
  if (!stat.size) throw new Error('Real-ESRGAN returned an empty output file.');
}

async function runRealESRGAN(inputPath, outputPath, scale) {
  if (![2, 4].includes(scale)) throw new Error('Controller selected an invalid scale.');
  const gradio = await import('@gradio/client');
  let client;
  try { client = await gradio.Client.connect(SPACE, { timeout: 90000 }); }
  catch (e) { throw new Error(`Cannot connect to Real-ESRGAN Space: ${e.message}`); }
  console.log(`[PIXORA] Real-ESRGAN agent running at ${scale}x on ${SPACE}`);
  let result;
  try {
    result = await client.predict('/predict', [gradio.handle_file(inputPath), scale]);
  } catch (e) {
    throw new Error(`Real-ESRGAN failed: ${e?.message || String(e)}`);
  }
  const output = findOutputFile(result?.data);
  if (!output) throw new Error('Real-ESRGAN did not return an output image.');
  await downloadGradioOutput(output, SPACE, outputPath);
  return { status: 'success', scale, engine: `Real-ESRGAN (${SPACE})` };
}

async function reviewerAgent(inputPath, outputPath) {
  const inputStat = await fs.stat(inputPath);
  const outputStat = await fs.stat(outputPath);
  // Keep memory/request payload bounded on 512 MB hosts. Skip visual comparison if files are too large.
  if (inputStat.size > REVIEW_MAX_BYTES || outputStat.size > REVIEW_MAX_BYTES) {
    return { status: 'skipped', summary: 'Visual review skipped to protect server memory; output was retained but not quality-verified.', result: null };
  }
  const original = await imagePart(inputPath);
  const enhanced = await imagePart(outputPath);
  const result = await geminiJson(
    'You are PIXORA Quality Reviewer. Compare ORIGINAL and REAL-ESRGAN RESULT side by side. Assess natural detail, blur, grain/noise, halos, artifacts, color naturalness and whether the result is actually better. Do not identify people. Be conservative. Return JSON {"verdict":"better|similar|worse|uncertain","original_score":1-10,"result_score":1-10,"issues":[string],"reason":"short"}. The first image is ORIGINAL; the second is RESULT. Do not assume larger means better.',
    [original.part, enhanced.part]
  );
  return { status: 'success', result, summary: result.reason || `Reviewer verdict: ${result.verdict || 'uncertain'}.` };
}

async function finalTouchAgent(inputPath, outputPath, review) {
  const verdict = String(review?.result?.verdict || 'uncertain').toLowerCase();
  const scoreA = Number(review?.result?.original_score);
  const scoreB = Number(review?.result?.result_score);
  const clearlyWorse = verdict === 'worse' || (Number.isFinite(scoreA) && Number.isFinite(scoreB) && scoreB + 2 <= scoreA);
  if (clearlyWorse) {
    const head = await fs.open(inputPath, 'r');
    const signature = Buffer.alloc(3);
    try { await head.read(signature, 0, 3, 0); } finally { await head.close(); }
    const isJpeg = signature[0] === 0xff && signature[1] === 0xd8 && signature[2] === 0xff;
    if (isJpeg) {
      await fs.copyFile(inputPath, outputPath);
      return {
        status: 'fallback_original',
        summary: 'Quality guard rejected a visibly worse result and preserved the original JPEG instead.',
        selected: 'original'
      };
    }
    return {
      status: 'review_warning',
      summary: 'Reviewer flagged possible degradation, but original format conversion is unavailable; enhanced output retained for comparison.',
      selected: 'enhanced_review_warning'
    };
  }
  if (review?.status === 'skipped') {
    return { status: 'review_skipped', summary: 'Final guard retained the Real-ESRGAN output because visual review was skipped for memory safety.', selected: 'enhanced_unverified' };
  }
  return {
    status: 'success',
    summary: 'Final quality guard accepted the enhanced result. It avoids aggressive extra filters that could damage texture or color.',
    selected: 'enhanced'
  };
}

async function enhanceImage(inputPath, outputPath) {
  const agents = {
    diagnostic: { status: 'running', name: 'Diagnostic Agent', summary: 'Inspecting blur, grain, lighting and color.' },
    manager: { status: 'pending', name: 'Controller Agent', summary: 'Choosing a conservative enhancement strategy.' },
    enhancer: { status: 'pending', name: 'Real-ESRGAN Agent', summary: 'Waiting for controller.' },
    reviewer: { status: 'pending', name: 'Quality Reviewer', summary: 'Will compare original and result.' },
    finalTouch: { status: 'pending', name: 'Final Touch Agent', summary: 'Will protect natural quality.' }
  };
  let analysis = { status: 'failed', warning: 'Gemini analysis did not run.' };
  let manager = { status: 'failed', summary: 'Controller could not choose a strategy.' };
  let enhancement = { status: 'failed', summary: 'Real-ESRGAN has not completed.' };
  let review = { status: 'pending', summary: 'Quality review has not run.' };
  let finalTouch = { status: 'pending', summary: 'Final quality guard has not run.' };
  let scale = DEFAULT_SCALE === 2 ? 2 : 4;

  try {
    analysis = await diagnoseAgent(inputPath);
    agents.diagnostic = { ...agents.diagnostic, status: 'success', summary: `Issues: ${(analysis.result.issues || []).slice(0, 3).join(', ') || 'No major issue reported'}. Severity: ${analysis.result.severity || 'uncertain'}.` };
  } catch (e) {
    analysis = { status: 'failed', warning: e.message, summary: 'Diagnostic Agent failed; conservative default will be used.' };
    agents.diagnostic = { ...agents.diagnostic, status: 'failed', summary: e.message };
  }

  try {
    if (analysis.status === 'success') {
      manager = await managerAgent(analysis.result);
      scale = manager.result.scale;
      agents.manager = { ...agents.manager, status: 'success', summary: `${scale}× selected. ${manager.summary}` };
    } else {
      scale = 2;
      manager = { status: 'fallback', result: { scale }, summary: 'Gemini diagnosis unavailable; using conservative 2× scale.' };
      agents.manager = { ...agents.manager, status: 'fallback', summary: manager.summary };
    }
  } catch (e) {
    scale = 2;
    manager = { status: 'fallback', warning: e.message, result: { scale }, summary: 'Controller unavailable; safe 2× fallback selected.' };
    agents.manager = { ...agents.manager, status: 'fallback', summary: manager.summary };
  }

  try {
    enhancement = await runRealESRGAN(inputPath, outputPath, scale);
    enhancement.status = 'success';
    enhancement.summary = `Real-ESRGAN completed at ${scale}×.`;
    agents.enhancer = { ...agents.enhancer, status: 'success', summary: enhancement.summary };
  } catch (e) {
    enhancement = { status: 'failed', warning: e.message, summary: 'Real-ESRGAN could not enhance this image.' };
    agents.enhancer = { ...agents.enhancer, status: 'failed', summary: e.message };
    return {
      ok: false, analysis, enhancement, manager, review, finalTouch, agents,
      notice: analysis.status === 'success' ? 'Diagnosis completed, but Real-ESRGAN failed.' : 'Diagnosis and Real-ESRGAN failed.'
    };
  }

  try {
    review = await reviewerAgent(inputPath, outputPath);
    agents.reviewer = { ...agents.reviewer, status: review.status, summary: review.summary || `Reviewer verdict: ${review.result?.verdict || 'unknown'}.` };
  } catch (e) {
    review = { status: 'failed', warning: e.message, summary: 'Visual review failed; no automatic rejection was made.' };
    agents.reviewer = { ...agents.reviewer, status: 'failed', summary: e.message };
  }

  try {
    finalTouch = await finalTouchAgent(inputPath, outputPath, review);
    agents.finalTouch = { ...agents.finalTouch, status: finalTouch.status === 'fallback_original' ? 'fallback' : finalTouch.status === 'success' ? 'success' : 'skipped', summary: finalTouch.summary };
  } catch (e) {
    finalTouch = { status: 'failed', warning: e.message, summary: 'Final quality guard failed; enhanced output may still be available.' };
    agents.finalTouch = { ...agents.finalTouch, status: 'failed', summary: e.message };
  }

  const outputStat = await fs.stat(outputPath);
  const originalChosen = finalTouch.selected === 'original';
  return {
    ok: true, analysis, manager, enhancement, review, finalTouch, agents,
    width: null, height: null,
    engine: originalChosen ? 'PIXORA Quality Guard (original preserved)' : enhancement.engine,
    outputBytes: outputStat.size,
    notice: originalChosen
      ? 'Real-ESRGAN output looked worse in review, so PIXORA preserved your original photo instead.'
      : review.status === 'skipped'
        ? 'Enhancement completed; visual review was skipped to protect server memory.'
        : review.result?.verdict === 'better'
          ? 'Multi-agent review accepted the enhanced result.'
          : 'Enhancement completed. Review the before/after result before using it.'
  };
}

module.exports = { enhanceImage, analyzeWithGemini: diagnoseAgent };
