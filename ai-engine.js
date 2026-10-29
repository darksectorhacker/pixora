// ============================================================
// PIXORA AI - Complete AI Engine
// Gemini: image quality analysis
// Real-ESRGAN: hosted image enhancement
// ============================================================

const fs = require('fs/promises');
const path = require('path');
const { createWriteStream } = require('fs');
const { pipeline } = require('stream/promises');
const { Readable } = require('stream');

const DEFAULT_SPACE =
  process.env.PIXORA_ESRGAN_SPACE ||
  'Nick088/Real-ESRGAN_Pytorch';

const DEFAULT_MODEL =
  process.env.GEMINI_MODEL ||
  'gemini-3.5-flash-lite';

const DEFAULT_SCALE = Number(
  process.env.PIXORA_ESRGAN_SCALE || 4
);

// ------------------------------------------------------------
// Detect image format
// ------------------------------------------------------------

function detectMime(buffer) {
  if (
    buffer.length >= 8 &&
    buffer.subarray(0, 8).equals(
      Buffer.from([
        137, 80, 78, 71, 13, 10, 26, 10
      ])
    )
  ) {
    return 'image/png';
  }

  if (
    buffer.length >= 3 &&
    buffer[0] === 0xff &&
    buffer[1] === 0xd8 &&
    buffer[2] === 0xff
  ) {
    return 'image/jpeg';
  }

  if (
    buffer.length >= 12 &&
    buffer.toString('ascii', 0, 4) === 'RIFF' &&
    buffer.toString('ascii', 8, 12) === 'WEBP'
  ) {
    return 'image/webp';
  }

  return 'application/octet-stream';
}

// ------------------------------------------------------------
// GEMINI IMAGE ANALYSIS
// Gemini only analyzes; it does not enhance images.
// ------------------------------------------------------------

async function analyzeWithGemini(inputPath) {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error(
      'Gemini API key is not configured.'
    );
  }

  const imageBuffer = await fs.readFile(inputPath);
  const mimeType = detectMime(imageBuffer);

  if (!mimeType.startsWith('image/')) {
    throw new Error(
      'Unsupported image format for Gemini analysis.'
    );
  }

  const url =
    'https://generativelanguage.googleapis.com/v1beta/models/' +
    encodeURIComponent(DEFAULT_MODEL) +
    ':generateContent?key=' +
    encodeURIComponent(apiKey);

  const response = await fetch(url, {
    method: 'POST',

    headers: {
      'Content-Type': 'application/json'
    },

    body: JSON.stringify({
      contents: [
        {
          parts: [
            {
              text:
                'You are PIXORA AI Problem Finder. Analyze the ' +
                'provided image for visible quality issues. ' +
                'Check blur, noise, low resolution, lighting, ' +
                'compression artifacts, and whether a face is ' +
                'present. Do not identify people. Do not generate ' +
                'or edit an image. Return JSON with keys: issues ' +
                '(array), severity (low, medium, or high), and ' +
                'recommendation (short text).'
            },

            {
              inline_data: {
                mime_type: mimeType,
                data: imageBuffer.toString('base64')
              }
            }
          ]
        }
      ],

      generationConfig: {
        temperature: 0.2,
        responseMimeType: 'application/json'
      }
    }),

    signal: AbortSignal.timeout(45000)
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      payload?.error?.message ||
      `Gemini request failed with HTTP ${response.status}.`
    );
  }

  const text = payload?.candidates?.[0]?.content?.parts
    ?.map(part => part.text || '')
    .join('\n')
    .trim();

  let analysis;

  try {
    analysis = JSON.parse(text || '{}');
  } catch {
    analysis = {
      summary: text || 'Gemini returned no analysis.'
    };
  }

  return {
    enabled: true,
    status: 'success',
    model: DEFAULT_MODEL,
    result: analysis
  };
}

// ------------------------------------------------------------
// Find an output file in a Gradio response
// ------------------------------------------------------------

function findOutputFile(data) {
  if (!data) {
    return null;
  }

  if (typeof data === 'string') {
    return { path: data };
  }

  if (Array.isArray(data)) {
    for (const item of data) {
      const found = findOutputFile(item);

      if (found) {
        return found;
      }
    }

    return null;
  }

  if (typeof data === 'object') {
    if (data.url || data.path || data.name) {
      return data;
    }

    for (const value of Object.values(data)) {
      const found = findOutputFile(value);

      if (found) {
        return found;
      }
    }
  }

  return null;
}

// ------------------------------------------------------------
// Download Gradio output directly to disk (avoid loading the
// full enhanced image into RAM or decoding it with Jimp).
// ------------------------------------------------------------

async function downloadGradioOutput(fileData, space, outputPath) {
  if (fileData.path && path.isAbsolute(fileData.path)) {
    try {
      await fs.access(fileData.path);
      await fs.copyFile(fileData.path, outputPath);
      return;
    } catch {
      // The returned path may belong to the remote server.
    }
  }

  const base =
    `https://${space.toLowerCase().replace('/', '-')}.hf.space`;

  let fileUrl = fileData.url;

  if (!fileUrl && fileData.path) {
    fileUrl = '/gradio_api/file=' + encodeURIComponent(fileData.path);
  }

  if (!fileUrl && fileData.name) {
    fileUrl = '/gradio_api/file=' + encodeURIComponent(fileData.name);
  }

  if (!fileUrl) {
    throw new Error('Real-ESRGAN returned no downloadable output file.');
  }

  if (fileUrl.startsWith('/')) fileUrl = base + fileUrl;

  const response = await fetch(fileUrl, {
    signal: AbortSignal.timeout(120000)
  });

  if (!response.ok || !response.body) {
    throw new Error(`Downloading enhanced image failed: HTTP ${response.status}.`);
  }

  await pipeline(
    Readable.fromWeb(response.body),
    createWriteStream(outputPath)
  );

  const stat = await fs.stat(outputPath);
  if (!stat.size) throw new Error('Real-ESRGAN returned an empty output file.');
}

// ------------------------------------------------------------
// REAL-ESRGAN IMAGE ENHANCEMENT
// ------------------------------------------------------------

async function runRealESRGAN(inputPath, outputPath) {
  const space = DEFAULT_SPACE;
  const scale = DEFAULT_SCALE;

  if (![2, 4, 8].includes(scale)) {
    throw new Error(
      'Invalid upscale factor. Use 2, 4, or 8.'
    );
  }

  console.log(
    `[PIXORA] Connecting to Real-ESRGAN Space: ${space}`
  );

  let Client;
  let handle_file;

  try {
    const gradio = await import('@gradio/client');

    Client = gradio.Client;
    handle_file = gradio.handle_file;
  } catch {
    throw new Error(
      'Cannot load @gradio/client. Run npm install in ~/pixora.'
    );
  }

  let client;

  try {
    client = await Client.connect(space, {
      timeout: 60000
    });
  } catch (error) {
    throw new Error(
      `Cannot connect to Real-ESRGAN: ${error.message}`
    );
  }

  let result;

  try {
    console.log(
      `[PIXORA] Running image enhancement at ${scale}x...`
    );

    result = await client.predict('/predict', [
      handle_file(inputPath),
      scale
    ]);
  } catch (error) {
    const details = error?.message || String(error);

    console.error(
      '[PIXORA] Real-ESRGAN request failed:',
      details
    );

    throw new Error(
      'Real-ESRGAN could not process the image. ' +
      'The hosted service may be unavailable, or the image ' +
      'may exceed its limits. Details: ' + details
    );
  }

  const output = findOutputFile(result?.data);

  if (!output) {
    throw new Error(
      'Real-ESRGAN did not return an output image.'
    );
  }

  await downloadGradioOutput(output, space, outputPath);

  const outputStat = await fs.stat(outputPath);
  if (!outputStat.size) {
    throw new Error('Enhancement service returned an empty output image.');
  }

  console.log(
    `[PIXORA] Enhancement downloaded to disk (${outputStat.size} bytes).`
  );

  return {
    width: null,
    height: null,
    engine: `Real-ESRGAN via Hugging Face (${space})`
  };
}

// ------------------------------------------------------------
// MAIN PIPELINE
// Each service is handled independently.
// ------------------------------------------------------------

async function enhanceImage(inputPath, outputPath) {
  let analysis = {
    enabled: false,
    status: 'not_run',
    summary: 'Gemini analysis has not run.'
  };

  let enhancement = {
    enabled: false,
    status: 'not_run',
    summary: 'Image enhancement has not run.'
  };

  let enhancementResult = {};

  // Step 1: Gemini is optional.
  // If it fails, continue to Real-ESRGAN.

  try {
    analysis = await analyzeWithGemini(inputPath);

    console.log(
      `[PIXORA] Gemini analysis completed using ${analysis.model}.`
    );
  } catch (error) {
    const message = error?.message || String(error);

    console.warn(
      '[PIXORA] Gemini analysis failed:',
      message
    );

    analysis = {
      enabled: false,
      status: 'failed',
      warning: message,
      summary: 'Gemini analysis failed. Enhancement will continue.'
    };
  }

  // Step 2: Enhancement runs regardless of Gemini status.

  try {
    enhancementResult = await runRealESRGAN(
      inputPath,
      outputPath
    );

    enhancement = {
      enabled: true,
      status: 'success',
      summary: 'Image enhancement completed successfully.'
    };

    console.log(
      '[PIXORA] Real-ESRGAN enhancement succeeded.'
    );
  } catch (error) {
    const message = error?.message || String(error);

    console.error(
      '[PIXORA] Real-ESRGAN enhancement failed:',
      message
    );

    enhancement = {
      enabled: false,
      status: 'failed',
      warning: message,
      summary: 'Image enhancement could not be completed.'
    };
  }

  // Step 3: Return individual statuses.
  // A service failure does not throw away the other status.

  const success = enhancement.status === 'success';

  return {
    ok: success,

    analysis,

    enhancement,

    ...(success ? enhancementResult : {}),

    notice: success
      ? (
          analysis.status === 'success'
            ? 'Gemini analysis and image enhancement completed.'
            : 'Image enhancement completed, but Gemini analysis failed.'
        )
      : (
          analysis.status === 'success'
            ? 'Gemini analysis completed, but image enhancement failed.'
            : 'Both Gemini analysis and image enhancement failed.'
        )
  };
}

module.exports = {
  enhanceImage,
  analyzeWithGemini
};
