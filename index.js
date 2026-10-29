// ============================================================
// PIXORA AI
// Safe request handling + independent service statuses
// Gemini: image analysis
// Real-ESRGAN: image enhancement
// ============================================================

require('dotenv').config();

const express = require('express');
const multer = require('multer');
const { enhanceImage } = require('./ai-engine');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const app = express();
const PORT = Number(process.env.PORT || 3000);

const UPLOADS = path.join(__dirname, 'uploads');
const OUTPUTS = path.join(__dirname, 'enhanced');

for (const dir of [UPLOADS, OUTPUTS]) {
    fs.mkdirSync(dir, { recursive: true });
}

app.disable('x-powered-by');

const jobs = new Map();
let enhancementInProgress = false;

function removeFile(filePath) {
    if (filePath) {
        fs.promises.unlink(filePath).catch(() => {});
    }
}

// ============================================================
// UPLOAD CONFIGURATION
// ============================================================

const upload = multer({
    dest: UPLOADS,

    limits: {
        fileSize: 12 * 1024 * 1024,
        files: 1
    },

    fileFilter: (_req, file, callback) => {
        const allowedTypes = [
            'image/jpeg',
            'image/png',
            'image/webp',
            'image/tiff'
        ];

        if (!allowedTypes.includes(file.mimetype)) {
            return callback(
                new Error('Use JPG, PNG, WEBP, or TIFF.')
            );
        }

        callback(null, true);
    }
});

// ============================================================
// HEALTH CHECK
// ============================================================

app.get('/api/health', (_req, res) => {
    res.json({
        ok: true,
        app: 'PIXORA AI',

        services: {
            gemini: {
                configured: Boolean(process.env.GEMINI_API_KEY)
            },

            realESRGAN: {
                configured: true,
                space:
                    process.env.PIXORA_ESRGAN_SPACE ||
                    'Nick088/Real-ESRGAN_Pytorch'
            }
        }
    });
});

// ============================================================
// ENHANCE API
// ============================================================

app.post(
    '/api/enhance',
    (req, res, next) => {
        upload.single('photo')(req, res, error => {
            if (error) {
                console.error(
                    '[PIXORA] Upload error:',
                    error.message
                );

                if (error instanceof multer.MulterError) {
                    if (error.code === 'LIMIT_FILE_SIZE') {
                        return res.status(413).json({
                            ok: false,
                            stage: 'upload',
                            error: 'Image exceeds the 12 MB limit.'
                        });
                    }
                }

                return res.status(400).json({
                    ok: false,
                    stage: 'upload',
                    error: error.message || 'Image upload failed.'
                });
            }

            next();
        });
    },

    async (req, res) => {
        if (!req.file) {
            return res.status(400).json({
                ok: false,
                stage: 'upload',
                error: 'Choose an image first.'
            });
        }

        if (enhancementInProgress) {
            removeFile(req.file.path);
            return res.status(429).json({
                ok: false,
                stage: 'queue',
                error: 'PIXORA is processing another image. Please try again in a moment.'
            });
        }

        enhancementInProgress = true;

        const id = crypto.randomUUID();
        const input = req.file.path;
        const output = path.join(OUTPUTS, `${id}.jpg`);

        const requestStarted = Date.now();

        console.log(
            `[PIXORA] Request ${id}: ${req.file.originalname} ` +
            `(${req.file.size} bytes)`
        );

        try {
            // ------------------------------------------------
            // Run the image pipeline.
            // ai-engine.js should independently handle Gemini
            // and Real-ESRGAN failures.
            // ------------------------------------------------

            let result;

            try {
                result = await enhanceImage(input, output);
            } catch (error) {
                console.error(
                    `[PIXORA] Request ${id} pipeline error:`,
                    error.stack || error.message
                );

                removeFile(output);

                return res.status(500).json({
                    ok: false,
                    id,
                    error: 'Image processing could not be completed.',
                    details: 'Check the PIXORA server logs.'
                });
            }

            // ------------------------------------------------
            // Read individual service statuses.
            // ------------------------------------------------

            const analysis = result?.analysis || {
                enabled: false,
                status: 'unknown',
                summary: 'Gemini status was not returned.'
            };

            const enhancement = result?.enhancement || {
                enabled: false,
                status: 'unknown',
                summary: 'Enhancement status was not returned.'
            };

            const enhancementSucceeded =
                (enhancement.status === 'success' || result?.ok === true) &&
                fs.existsSync(output);

            console.log(
                `[PIXORA] Request ${id} status: ` +
                `Gemini=${analysis.status}, ` +
                `Enhancement=${enhancement.status}`
            );

            // ------------------------------------------------
            // If enhancement failed, do not report success.
            // Gemini results are still returned to the website.
            // ------------------------------------------------

            if (!enhancementSucceeded) {
                removeFile(output);

                const response = {
                    ok: false,
                    id,
                    analysis,
                    enhancement,
                    manager: result.manager || null,
                    review: result.review || null,
                    finalTouch: result.finalTouch || null,
                    agents: result.agents || null,

                    notice:
                        'Image enhancement failed. ' +
                        'Check the service statuses below.',

                    durationMs: Date.now() - requestStarted
                };

                return res.status(502).json(response);
            }

            // ------------------------------------------------
            // Save successful output.
            // ------------------------------------------------

            jobs.set(id, {
                output,
                created: Date.now()
            });

            const response = {
                ok: true,
                id,

                url: `/files/${id}.jpg`,
                downloadUrl: `/api/download/${id}`,

                width: result.width || null,
                height: result.height || null,

                engine: result.engine || 'Real-ESRGAN',

                analysis,
                enhancement,
                manager: result.manager || null,
                review: result.review || null,
                finalTouch: result.finalTouch || null,
                agents: result.agents || null,

                notice: result.notice ||
                    'Image enhancement completed.',

                durationMs: Date.now() - requestStarted
            };

            console.log(
                `[PIXORA] Request ${id} completed in ` +
                `${response.durationMs}ms`
            );

            return res.json(response);

        } catch (error) {
            console.error(
                `[PIXORA] Unexpected request error ${id}:`,
                error.stack || error.message
            );

            removeFile(output);

            if (!res.headersSent) {
                return res.status(500).json({
                    ok: false,
                    id,
                    error: 'An unexpected error occurred.',
                    details: 'Check the PIXORA server logs.'
                });
            }

        } finally {
            removeFile(input);
            enhancementInProgress = false;
        }
    }
);

// ============================================================
// SERVE ENHANCED IMAGE
// ============================================================

app.get('/files/:name', (req, res) => {
    if (!/^[a-f0-9-]+\.jpg$/i.test(req.params.name)) {
        return res.sendStatus(404);
    }

    const filePath = path.join(OUTPUTS, req.params.name);

    if (!fs.existsSync(filePath)) {
        return res.status(404).send(
            'Enhanced image is unavailable or has expired.'
        );
    }

    return res.sendFile(filePath, error => {
        if (error) {
            console.error(
                '[PIXORA] Preview delivery failed:',
                error.message
            );
        }
    });
});

// ============================================================
// DOWNLOAD ENHANCED IMAGE
// ============================================================

app.get('/api/download/:id', (req, res) => {
    const job = jobs.get(req.params.id);

    if (!job || !fs.existsSync(job.output)) {
        return res.status(404).json({
            ok: false,
            error: 'Image expired. Please enhance it again.'
        });
    }

    return res.download(
        job.output,
        `pixora-enhanced-${req.params.id}.jpg`,
        error => {
            if (error) {
                console.error(
                    '[PIXORA] Download failed:',
                    error.message
                );
            }
        }
    );
});

// ============================================================
// AUTOMATIC OUTPUT CLEANUP
// ============================================================

const cleanupTimer = setInterval(() => {
    const cutoff = Date.now() - 2 * 60 * 60 * 1000;

    for (const [id, job] of jobs) {
        if (job.created < cutoff) {
            removeFile(job.output);
            jobs.delete(id);
        }
    }
}, 15 * 60 * 1000);

cleanupTimer.unref();

// ============================================================
// WEBSITE
// ============================================================

app.get('/', (_req, res) => {
    res.type('html').send(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIXORA AI — Photo Enhancer</title>

<style>
:root {
    color-scheme: dark;
    --bg: #080b10;
    --panel: #111821;
    --line: #253442;
    --mint: #9cffc8;
    --green: #42f5a1;
    --muted: #9caebd;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background:
        radial-gradient(
            ellipse at 50% -15%,
            #163c31 0,
            transparent 42%
        ),
        var(--bg);
    color: #f4f8fb;
    font: 15px/1.5 system-ui, sans-serif;
    min-height: 100vh;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px clamp(18px, 5vw, 64px);
    border-bottom: 1px solid #ffffff12;
}

.brand {
    font-size: 21px;
    font-weight: 850;
    letter-spacing: 1.8px;
}

.brand span {
    color: var(--green);
}

.badge {
    border: 1px solid #42f5a155;
    border-radius: 99px;
    padding: 6px 11px;
    color: var(--mint);
    font-size: 12px;
}

main {
    width: min(1040px, 92vw);
    margin: 48px auto 65px;
}

.hero {
    text-align: center;
    margin-bottom: 30px;
}

.eyebrow {
    color: var(--green);
    font-weight: 700;
    letter-spacing: 3px;
    font-size: 11px;
}

.hero h1 {
    font-size: clamp(32px, 6vw, 58px);
    line-height: 1.04;
    letter-spacing: -1.8px;
    margin: 12px 0;
}

.hero h1 em {
    font-style: normal;
    color: var(--mint);
}

.hero p {
    color: var(--muted);
    max-width: 620px;
    margin: 12px auto;
}

.workspace {
    background: #ffffff05;
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: clamp(16px, 3vw, 28px);
    box-shadow: 0 24px 80px #0008;
}

.drop {
    min-height: 230px;
    border: 1.5px dashed #426354;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 24px;
    background: linear-gradient(
        135deg,
        #10241e88,
        #10182188
    );
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.drop.drag {
    border-color: var(--green);
    background: #18382a;
}

.drop input {
    position: absolute;
    inset: 0;
    opacity: 0;
    width: 100%;
    height: 100%;
    cursor: pointer;
}

.uploadicon {
    font-size: 36px;
    color: var(--green);
}

.drop h2 {
    margin: 8px 0;
    font-size: 20px;
}

.muted {
    color: var(--muted);
    font-size: 13px;
}

.actions {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 18px;
}

.btn {
    border: 0;
    border-radius: 12px;
    padding: 13px 20px;
    font-weight: 800;
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.primary {
    background: var(--green);
    color: #04140c;
}

.secondary {
    background: #18232c;
    border: 1px solid var(--line);
    color: #edf7f2;
}

.btn:disabled {
    opacity: .45;
    cursor: not-allowed;
}

.status {
    min-height: 24px;
    text-align: center;
    color: var(--mint);
    margin-top: 14px;
    overflow-wrap: anywhere;
}

.service-report {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 16px;
}

.service-card {
    background: #111821;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
    overflow-wrap: anywhere;
}

.service-card h3 {
    margin: 0 0 6px;
    font-size: 13px;
}

.service-card p {
    margin: 0;
    color: var(--muted);
    font-size: 12px;
}

.service-state {
    display: inline-block;
    margin-bottom: 8px;
    font-size: 12px;
    font-weight: 800;
}

.state-pending {
    color: #c6d2dc;
}

.state-success {
    color: #42f5a1;
}

.state-failed {
    color: #ff8585;
}

.previewgrid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 20px;
}

.preview {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
    min-width: 0;
}

.preview h3 {
    font-size: 12px;
    letter-spacing: 1px;
    color: var(--muted);
    padding: 12px 14px;
    margin: 0;
    border-bottom: 1px solid var(--line);
}

.imagebox {
    height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #070a0e;
    padding: 10px;
}

.imagebox img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.imagebox .empty {
    color: #5f7280;
    font-size: 13px;
    text-align: center;
}

.meta {
    padding: 10px 14px;
    color: var(--muted);
    font-size: 12px;
    overflow-wrap: anywhere;
}

.note {
    margin: 22px auto 0;
    max-width: 800px;
    text-align: center;
    color: #81919d;
    font-size: 12px;
}

.foot {
    padding: 22px;
    text-align: center;
    color: #60727e;
    font-size: 12px;
}

@media (max-width: 600px) {
    main {
        margin-top: 30px;
    }

    .previewgrid,
    .service-report {
        grid-template-columns: 1fr;
    }

    .imagebox {
        height: 250px;
    }

    .badge {
        font-size: 10px;
    }
}

.agent-panel{margin:24px 0 8px;padding:22px;border:1px solid rgba(150,190,255,.18);border-radius:24px;background:rgba(12,25,52,.38);backdrop-filter:blur(18px)}
.agent-heading h2{margin:6px 0;font-size:clamp(1.35rem,3vw,1.8rem)}.agent-heading p,.agent-note{color:var(--muted,#9eacc7);line-height:1.55}
.agent-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:18px}.agent-card{padding:16px;border:1px solid rgba(135,177,255,.17);border-radius:18px;background:rgba(255,255,255,.035);min-width:0}.agent-card h3{font-size:.96rem;margin:8px 0 10px}.agent-card p{font-size:.88rem;line-height:1.45;color:var(--muted,#9eacc7);overflow-wrap:anywhere}.agent-icon{display:grid;place-items:center;width:34px;height:34px;border-radius:12px;background:rgba(83,139,255,.14);color:#a9caff;font-size:1.25rem}.agent-note{font-size:.82rem;margin:18px 0 0}

</style>
</head>

<body>

<header>
    <div class="brand">PIXORA<span> AI</span></div>
    <div class="badge">PHOTO ENHANCEMENT LAB</div>
</header>

<main>

<section class="hero">
    <div class="eyebrow">RESTORE THE DETAILS</div>
    <h1>Make every photo<br><em>look its best.</em></h1>
    <p>
        Upload a photo, enhance clarity, and save your improved image.
    </p>
</section>

<section class="workspace">

<label class="drop" id="drop">
    <input
        id="photo"
        type="file"
        accept="image/jpeg,image/png,image/webp,image/tiff"
    >

    <div>
        <div class="uploadicon">＋</div>
        <h2>Drop your photo here</h2>
        <div class="muted">
            or tap to browse · JPG, PNG, WEBP, TIFF · max 12 MB
        </div>
        <div id="filename" class="muted" style="margin-top:10px"></div>
    </div>
</label>

<div class="actions">
    <button id="enhance" class="btn primary" disabled>
        ✦ Enhance photo
    </button>

    <button id="reset" class="btn secondary">Reset</button>

    <a id="download" class="btn secondary" hidden>
        ↓ Download enhanced
    </a>
</div>

<div id="status" class="status" role="status" aria-live="polite"></div>

<div class="service-report" aria-live="polite">
    <div class="service-card">
        <h3>Gemini Analysis</h3>
        <div id="geminiState" class="service-state state-pending">
            ⏳ Waiting
        </div>
        <p id="geminiDetail">
            Gemini checks visible image quality issues.
        </p>
    </div>

    <div class="service-card">
        <h3>Real-ESRGAN Enhancement</h3>
        <div id="enhanceState" class="service-state state-pending">
            ⏳ Waiting
        </div>
        <p id="enhanceDetail">
            Image upscaling has not started.
        </p>
    </div>
</div>

<section class="agent-panel">
  <div class="agent-heading"><span class="eyebrow">MULTI-AGENT WORKFLOW</span><h2>PIXORA AI Team</h2><p>Specialized agents inspect, coordinate, enhance, review and protect the final result.</p></div>
  <div class="agent-grid">
    <article class="agent-card"><div class="agent-icon">⌕</div><h3>01 · Diagnostic Agent</h3><div id="agentDiagnosticState" class="service-state state-pending">⏳ Waiting</div><p id="agentDiagnosticDetail">Checks blur, grain, lighting and color.</p></article>
    <article class="agent-card"><div class="agent-icon">✦</div><h3>02 · Controller Agent</h3><div id="agentManagerState" class="service-state state-pending">⏳ Waiting</div><p id="agentManagerDetail">Chooses a conservative 2× or 4× workflow.</p></article>
    <article class="agent-card"><div class="agent-icon">✧</div><h3>03 · Real-ESRGAN Agent</h3><div id="agentEnhancerState" class="service-state state-pending">⏳ Waiting</div><p id="agentEnhancerDetail">Performs AI upscaling on the hosted service.</p></article>
    <article class="agent-card"><div class="agent-icon">◎</div><h3>04 · Quality Reviewer</h3><div id="agentReviewerState" class="service-state state-pending">⏳ Waiting</div><p id="agentReviewerDetail">Compares original and enhanced versions.</p></article>
    <article class="agent-card"><div class="agent-icon">◇</div><h3>05 · Final Touch Agent</h3><div id="agentFinalState" class="service-state state-pending">⏳ Waiting</div><p id="agentFinalDetail">Protects natural quality if enhancement looks worse.</p></article>
  </div>
  <p class="agent-note">Quality protection is enabled. Real-ESRGAN primarily upscales; it cannot guarantee blur, grain or color correction for every photo. PIXORA may preserve the original if the reviewer flags clear degradation.</p>
</section>

<div class="previewgrid">

    <div class="preview">
        <h3>ORIGINAL</h3>
        <div class="imagebox" id="originalBox">
            <span class="empty">Your photo preview appears here</span>
        </div>
        <div class="meta" id="originalMeta">No image selected</div>
    </div>

    <div class="preview">
        <h3>ENHANCED</h3>
        <div class="imagebox" id="resultBox">
            <span class="empty">Your enhanced result appears here</span>
        </div>
        <div class="meta" id="resultMeta">Waiting for enhancement</div>
    </div>

</div>
</section>

<p class="note">
    Gemini analyzes visible quality issues when configured.
    Image enhancement uses a hosted third-party Real-ESRGAN service,
    whose availability and limits may vary. Your photo is sent to
    that service. Uploads are removed after processing; enhanced
    outputs are cleaned up after about two hours. Do not upload
    sensitive photos.
</p>

</main>

<div class="foot">PIXORA AI · GEMINI + REAL-ESRGAN</div>

<script>
const input = document.getElementById('photo');
const drop = document.getElementById('drop');
const go = document.getElementById('enhance');
const reset = document.getElementById('reset');
const download = document.getElementById('download');
const statusBox = document.getElementById('status');

const geminiState = document.getElementById('geminiState');
const geminiDetail = document.getElementById('geminiDetail');
const enhanceState = document.getElementById('enhanceState');
const enhanceDetail = document.getElementById('enhanceDetail');
const agentElements = {
    diagnostic: [document.getElementById('agentDiagnosticState'), document.getElementById('agentDiagnosticDetail')],
    manager: [document.getElementById('agentManagerState'), document.getElementById('agentManagerDetail')],
    enhancer: [document.getElementById('agentEnhancerState'), document.getElementById('agentEnhancerDetail')],
    reviewer: [document.getElementById('agentReviewerState'), document.getElementById('agentReviewerDetail')],
    finalTouch: [document.getElementById('agentFinalState'), document.getElementById('agentFinalDetail')]
};
function renderAgents(agents) {
    if (!agents) return;
    for (const [key, agent] of Object.entries(agents)) {
        const pair = agentElements[key];
        if (!pair || !agent) continue;
        const [stateEl, detailEl] = pair;
        stateEl.className = 'service-state';
        if (agent.status === 'success') { stateEl.classList.add('state-success'); stateEl.textContent = '✅ Complete'; }
        else if (agent.status === 'failed') { stateEl.classList.add('state-failed'); stateEl.textContent = '❌ Failed'; }
        else if (agent.status === 'fallback') { stateEl.classList.add('state-pending'); stateEl.textContent = '↪ Safe fallback'; }
        else if (agent.status === 'skipped') { stateEl.classList.add('state-pending'); stateEl.textContent = '↷ Skipped'; }
        else if (agent.status === 'running') { stateEl.classList.add('state-pending'); stateEl.textContent = '⏳ Running'; }
        else { stateEl.classList.add('state-pending'); stateEl.textContent = '⏳ Waiting'; }
        detailEl.textContent = agent.summary || '';
    }
}
function setAgentsRunning() {
    renderAgents({
        diagnostic: {status:'running', summary:'Analyzing blur, grain, lighting and color…'},
        manager: {status:'running', summary:'Controller will choose a safe upscale factor…'},
        enhancer: {status:'running', summary:'Waiting for hosted Real-ESRGAN…'},
        reviewer: {status:'running', summary:'Will compare the source and result…'},
        finalTouch: {status:'running', summary:'Will apply the final quality safeguard…'}
    });
}
function resetAgents() { renderAgents({
    diagnostic:{status:'pending',summary:'Checks blur, grain, lighting and color.'},
    manager:{status:'pending',summary:'Chooses a conservative 2× or 4× workflow.'},
    enhancer:{status:'pending',summary:'Performs AI upscaling on the hosted service.'},
    reviewer:{status:'pending',summary:'Compares original and enhanced versions.'},
    finalTouch:{status:'pending',summary:'Protects natural quality if enhancement looks worse.'}
}); }

let selected = null;
let previewUrl = null;
let processing = false;

function setServiceState(element, state, message) {
    element.className = 'service-state';

    if (state === 'success') {
        element.classList.add('state-success');
        element.textContent = '✅ Successful';
    } else if (state === 'failed') {
        element.classList.add('state-failed');
        element.textContent = '❌ Failed';
    } else if (state === 'running') {
        element.classList.add('state-pending');
        element.textContent = '⏳ Running';
    } else {
        element.classList.add('state-pending');
        element.textContent = '⏳ Waiting';
    }

    if (element === geminiState) {
        geminiDetail.textContent = message || '';
    } else {
        enhanceDetail.textContent = message || '';
    }
}

function resetServiceStates() {
    resetAgents();
    setServiceState(
        geminiState,
        'pending',
        'Gemini checks visible image quality issues.'
    );

    setServiceState(
        enhanceState,
        'pending',
        'Image upscaling has not started.'
    );
}

function showImageError(message) {
    statusBox.textContent = message;
}

function choose(file) {
    if (!file) return;

    const types = [
        'image/jpeg',
        'image/png',
        'image/webp',
        'image/tiff'
    ];

    if (!types.includes(file.type)) {
        showImageError(
            'Choose a JPG, PNG, WEBP, or TIFF image.'
        );
        return;
    }

    if (file.size > 12 * 1024 * 1024) {
        showImageError('Image exceeds 12 MB.');
        return;
    }

    selected = file;
    go.disabled = false;
    download.hidden = true;
    statusBox.textContent = '';

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    document.getElementById('originalBox').innerHTML = '';

    const image = new Image();
    image.alt = 'Original photo';
    image.src = previewUrl;

    document.getElementById('originalBox').appendChild(image);

    document.getElementById('filename').textContent = file.name;

    document.getElementById('originalMeta').textContent =
        (file.size / 1048576).toFixed(2) +
        ' MB · ' + file.type;

    document.getElementById('resultBox').innerHTML =
        '<span class="empty">Ready to enhance</span>';

    document.getElementById('resultMeta').textContent =
        'Waiting for enhancement';

    resetServiceStates();
}

input.addEventListener('change', () => {
    choose(input.files[0]);
});

for (const eventName of ['dragenter', 'dragover']) {
    drop.addEventListener(eventName, event => {
        event.preventDefault();
        drop.classList.add('drag');
    });
}

for (const eventName of ['dragleave', 'drop']) {
    drop.addEventListener(eventName, event => {
        event.preventDefault();
        drop.classList.remove('drag');
    });
}

drop.addEventListener('drop', event => {
    choose(event.dataTransfer.files[0]);
});

// ------------------------------------------------------------
// PROCESS IMAGE
// ------------------------------------------------------------

go.addEventListener('click', async () => {
    if (!selected || processing) return;

    processing = true;
    go.disabled = true;
    download.hidden = true;

    statusBox.textContent = 'Uploading photo…';
    setAgentsRunning();

    setServiceState(
        geminiState,
        'running',
        'Checking image quality…'
    );

    setServiceState(
        enhanceState,
        'running',
        'Waiting for the enhancement service…'
    );

    const formData = new FormData();
    formData.append('photo', selected);

    const controller = new AbortController();

    // Allow longer processing time for remote services.
    const timer = setTimeout(
        () => controller.abort(),
        360000
    );

    try {
        statusBox.textContent =
            'Processing image. Please keep this page open.';

        const response = await fetch('/api/enhance', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                'Server returned an unreadable response. Check Termux logs.'
            );
        }

        renderAgents(data.agents);

        // Display Gemini status even if enhancement failed.
        if (data.analysis) {
            const analysis = data.analysis;

            setServiceState(
                geminiState,
                analysis.status === 'success'
                    ? 'success'
                    : analysis.status === 'failed'
                        ? 'failed'
                        : 'pending',

                analysis.status === 'success'
                    ? 'Image analysis completed.'
                    : analysis.warning ||
                      analysis.summary ||
                      'Gemini analysis was unavailable.'
            );
        }

        if (data.enhancement) {
            const enhancement = data.enhancement;

            setServiceState(
                enhanceState,
                enhancement.status === 'success'
                    ? 'success'
                    : enhancement.status === 'failed'
                        ? 'failed'
                        : 'pending',

                enhancement.summary ||
                enhancement.warning ||
                'No enhancement details returned.'
            );
        }

        // Handle failed enhancement without losing Gemini status.
        if (!response.ok || !data.ok) {
            document.getElementById('resultBox').innerHTML =
                '<span class="empty">No enhanced image was produced</span>';

            document.getElementById('resultMeta').textContent =
                'Enhancement unavailable';

            statusBox.textContent =
                data.notice ||
                data.error ||
                'Image enhancement failed. See the service report above.';

            return;
        }

        // A successful API response must include a result URL.
        if (!data.url) {
            throw new Error(
                'Server reported success but did not return an image URL.'
            );
        }

        document.getElementById('resultBox').innerHTML = '';

        const resultImage = new Image();
        resultImage.alt = 'Enhanced photo';

        resultImage.onload = () => {
            document.getElementById('resultMeta').textContent =
                (data.width && data.height)
                    ? data.width + ' × ' + data.height + ' px · JPEG'
                    : 'Enhanced JPEG image';
        };

        resultImage.onerror = () => {
            document.getElementById('resultMeta').textContent =
                'Preview could not load. Try downloading the image.';

            statusBox.textContent =
                'Image processing completed, but the preview failed.';
        };

        resultImage.src = data.url + '?t=' + Date.now();

        document.getElementById('resultBox').appendChild(resultImage);

        download.href = data.downloadUrl;
        download.hidden = false;
        download.setAttribute(
            'download',
            'pixora-enhanced.jpg'
        );

        statusBox.textContent =
            data.notice || 'Image enhancement completed.';

    } catch (error) {
        console.error('[PIXORA] Frontend request error:', error);

        if (error.name === 'AbortError') {
            statusBox.textContent =
                'Processing timed out. Check the Termux logs and try again.';
        } else {
            statusBox.textContent =
                error.message || 'Something went wrong.';
        }

        // Unknown network errors should not be falsely reported
        // as a specific Gemini or Real-ESRGAN failure.
        if (geminiState.textContent.includes('Running')) {
            setServiceState(
                geminiState,
                'failed',
                'No final Gemini status was received.'
            );
        }

        if (enhanceState.textContent.includes('Running')) {
            setServiceState(
                enhanceState,
                'failed',
                'No final enhancement status was received.'
            );
        }

    } finally {
        clearTimeout(timer);
        processing = false;
        go.disabled = !selected;
    }
});

// ------------------------------------------------------------
// RESET PAGE
// ------------------------------------------------------------

reset.addEventListener('click', () => {
    if (processing) {
        statusBox.textContent =
            'Please wait until the current request finishes.';
        return;
    }

    selected = null;
    input.value = '';
    go.disabled = true;
    download.hidden = true;
    statusBox.textContent = '';

    document.getElementById('filename').textContent = '';

    document.getElementById('originalBox').innerHTML =
        '<span class="empty">Your photo preview appears here</span>';

    document.getElementById('resultBox').innerHTML =
        '<span class="empty">Your enhanced result appears here</span>';

    document.getElementById('originalMeta').textContent =
        'No image selected';

    document.getElementById('resultMeta').textContent =
        'Waiting for enhancement';

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = null;

    resetServiceStates();
});
</script>

</body>
</html>`);
});

// ============================================================
// GLOBAL ERROR HANDLER
// ============================================================

app.use((error, _req, res, _next) => {
    console.error(
        '[PIXORA] Unhandled request error:',
        error.stack || error.message
    );

    if (res.headersSent) {
        return;
    }

    res.status(500).json({
        ok: false,
        error: 'The request could not be completed.',
        details: 'Check the PIXORA server logs.'
    });
});

// ============================================================
// START SERVER
// ============================================================

const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(
        `PIXORA AI running at http://localhost:${PORT}`
    );
});

server.on('error', error => {
    console.error(
        '[PIXORA] Server error:',
        error.message
    );
});
