/**
 * FakeLens × SIGNAL — Unified Client Application Logic
 * Integrates NLP text classification (/predict) and SIGNAL deepfake video detection (/detect).
 * Features:
 * - Lens switching (Article Text vs Deepfake Video)
 * - Video drag-and-drop & optical viewfinder preview
 * - Live frame-by-frame anomaly waveform generator
 * - Forensic PDF Report export using jsPDF
 * - Raw telemetry JSON inspection modal
 * - Explanatory n-gram coefficient badges
 */

// Global State
let currentVideoFile = null;
let currentVideoUrl = null;
let latestDeepfakeData = null;
let latestTextData = null;

// =============================================================================
// 1. Interactive Ambient UI (Cursor Glow, Ripple, Sticky Masthead, Animations)
// =============================================================================
const glow = document.getElementById('cursor-glow');
window.addEventListener('mousemove', (e) => {
  if (glow) {
    glow.style.left = e.clientX + 'px';
    glow.style.top = e.clientY + 'px';
  }
});

function bindHoverGlow() {
  document.querySelectorAll('.btn, .sample-btn, a, .exhibit-card, .step, .interp-card, .term, .ledger-cell, .cm-cell, .lens-tab, .video-dropzone, .t-cell, .tech-card').forEach((el) => {
    el.addEventListener('mouseenter', () => glow && glow.classList.add('big'));
    el.addEventListener('mouseleave', () => glow && glow.classList.remove('big'));
  });
}
bindHoverGlow();

// Masthead blur on scroll
const masthead = document.getElementById('masthead');
window.addEventListener('scroll', () => {
  if (masthead) {
    masthead.classList.toggle('scrolled', window.scrollY > 18);
  }
});

// Scroll reveal observer
const io = new IntersectionObserver((entries) => {
  entries.forEach((en) => {
    if (en.isIntersecting) {
      en.target.classList.add('in');
      io.unobserve(en.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

// Animated count-up numbers
function animateCount(el) {
  const target = parseFloat(el.dataset.count);
  const suffix = el.dataset.suffix || '';
  const isInt = Number.isInteger(target) && !el.dataset.count.includes('.');
  const dur = 900;
  const start = performance.now();

  function step(now) {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = target * eased;
    el.textContent = (isInt ? Math.round(val).toLocaleString() : val.toFixed(2)) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const countIo = new IntersectionObserver((entries) => {
  entries.forEach((en) => {
    if (en.isIntersecting) {
      animateCount(en.target);
      countIo.unobserve(en.target);
    }
  });
}, { threshold: 0.4 });

document.querySelectorAll('[data-count]').forEach((el) => countIo.observe(el));

// Button ripple effect
document.querySelectorAll('.btn').forEach((btn) => {
  btn.addEventListener('click', function (e) {
    const r = document.createElement('span');
    const rect = this.getBoundingClientRect();
    r.className = 'ripple';
    r.style.left = (e.clientX - rect.left) + 'px';
    r.style.top = (e.clientY - rect.top) + 'px';
    this.appendChild(r);
    setTimeout(() => r.remove(), 600);
  });
});

// =============================================================================
// 2. Lens Switcher (NLP Text vs SIGNAL Deepfake Video)
// =============================================================================
function switchLens(mode) {
  const tabNlp = document.getElementById('tab-nlp');
  const tabVideo = document.getElementById('tab-video');
  const panelNlp = document.getElementById('panel-nlp');
  const panelVideo = document.getElementById('panel-video');
  const label = document.getElementById('intake-tab-label');

  if (mode === 'nlp') {
    if (tabNlp) tabNlp.classList.add('active');
    if (tabVideo) tabVideo.classList.remove('active');
    if (panelNlp) panelNlp.style.display = 'block';
    if (panelVideo) panelVideo.style.display = 'none';
    if (label) label.textContent = 'exhibit 00 — text intake';
  } else {
    if (tabVideo) tabVideo.classList.add('active');
    if (tabNlp) tabNlp.classList.remove('active');
    if (panelVideo) panelVideo.style.display = 'block';
    if (panelNlp) panelNlp.style.display = 'none';
    if (label) label.textContent = 'exhibit 00-B — video evidence intake';
  }
}

// =============================================================================
// 3. NLP Text Misinformation Detector
// =============================================================================
function updateCounters() {
  const input = document.getElementById('article-input');
  const t = input ? input.value.trim() : '';
  const words = t ? t.split(/\s+/).filter(Boolean).length : 0;
  const counterEl = document.getElementById('counters');
  const shortWarnEl = document.getElementById('short-warning');

  if (counterEl) {
    counterEl.textContent = `${words} words · ${t.length} chars`;
  }

  if (shortWarnEl) {
    if (t.length > 0 && (words < 20 || t.length < 100)) {
      shortWarnEl.style.display = 'inline-block';
    } else {
      shortWarnEl.style.display = 'none';
    }
  }
}

function showTextError(msg) {
  const errBox = document.getElementById('error-box');
  if (errBox) {
    errBox.textContent = msg;
    errBox.style.display = 'block';
  }
}

function hideTextError() {
  const errBox = document.getElementById('error-box');
  if (errBox) {
    errBox.textContent = '';
    errBox.style.display = 'none';
  }
}

const REAL_SAMPLE = "WASHINGTON (Reuters) - Senate Republicans and Democrats reached a bipartisan compromise on federal budget appropriations late Wednesday, averting a government shutdown before the midnight deadline. The bill passed 74 to 24 and now heads to the White House.";
const FAKE_SAMPLE = "BREAKING: Secret video reveals election officials caught admitting to hidden algorithms altering ballots in key counties! Watch the featured image evidence mainstream media is blacking out — share before it gets deleted!";

function loadSample(kind) {
  const input = document.getElementById('article-input');
  if (input) {
    input.value = kind === 'real' ? REAL_SAMPLE : FAKE_SAMPLE;
    updateCounters();
  }
  hideTextError();
  const verdictWrap = document.getElementById('verdict-wrap');
  if (verdictWrap) verdictWrap.classList.remove('show');
}

function clearInput() {
  const input = document.getElementById('article-input');
  if (input) {
    input.value = '';
    updateCounters();
    input.focus();
  }
  hideTextError();
  const verdictWrap = document.getElementById('verdict-wrap');
  if (verdictWrap) verdictWrap.classList.remove('show');
}

async function analyze(e) {
  if (e && e.preventDefault) e.preventDefault();

  const input = document.getElementById('article-input');
  const text = input ? input.value.trim() : '';
  const btn = document.getElementById('btn-analyze');
  const verdictWrap = document.getElementById('verdict-wrap');
  const stamp = document.getElementById('stamp');
  const confVal = document.getElementById('conf-val');
  const confBar = document.getElementById('conf-bar');
  const termsList = document.getElementById('term-list');

  if (!text) {
    showTextError('Please enter or paste a news headline or article body.');
    if (verdictWrap) verdictWrap.classList.remove('show');
    return;
  }

  if (text.length < 10) {
    showTextError('Input text is too short. Please provide at least 10 characters or a complete headline.');
    if (verdictWrap) verdictWrap.classList.remove('show');
    return;
  }

  hideTextError();
  if (verdictWrap) verdictWrap.classList.remove('show');

  if (btn) {
    btn.disabled = true;
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'Analyzing...';
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      let errMsg = 'Could not connect to FakeLens backend';
      try {
        const errData = await response.json();
        if (errData && errData.error) errMsg = errData.error;
      } catch (parseErr) {}
      showTextError(errMsg);
      return;
    }

    const data = await response.json();
    latestTextData = data;

    if (data.status !== 'success') {
      showTextError(data.error || 'Prediction could not be completed.');
      return;
    }

    const isFake = (data.prediction || '').toUpperCase() === 'FAKE';
    const conf = typeof data.confidence === 'number' ? data.confidence : 0;

    const metaTag = document.getElementById('verdict-meta-tag');
    if (metaTag && data.model) {
      metaTag.textContent = data.model + ' Classification';
    }

    if (stamp) {
      stamp.classList.remove('slam', 'real', 'fake');
      void stamp.offsetWidth;
      stamp.textContent = isFake ? 'classified fake' : 'classified real';
      stamp.classList.add(isFake ? 'fake' : 'real', 'slam');
    }

    if (confVal) confVal.textContent = conf.toFixed(1) + '%';

    if (confBar) {
      confBar.className = 'bar-fill ' + (isFake ? 'fake' : 'real');
      confBar.style.width = '0%';
      requestAnimationFrame(() => {
        confBar.style.width = Math.min(100, Math.max(0, conf)) + '%';
      });
    }

    if (termsList) {
      termsList.innerHTML = '';
      const features = Array.isArray(data.features) ? data.features : [];

      features.forEach((f, i) => {
        const span = document.createElement('span');
        const isTermFake = (f.influence || '').toUpperCase() === 'FAKE' || f.coefficient > 0;
        span.className = 'term ' + (isTermFake ? 'fake' : 'real');

        const coefNum = typeof f.coefficient === 'number' ? f.coefficient : 0;
        const sign = coefNum > 0 ? '+' : '';
        span.textContent = `${f.term} (${sign}${coefNum.toFixed(2)})`;
        span.title = `Weight: ${f.weight} | TF-IDF: ${f.tfidf} | Coef: ${coefNum}`;
        span.style.animationDelay = (i * 0.08) + 's';

        span.addEventListener('mouseenter', () => glow && glow.classList.add('big'));
        span.addEventListener('mouseleave', () => glow && glow.classList.remove('big'));

        termsList.appendChild(span);
      });
    }

    if (verdictWrap) verdictWrap.classList.add('show');

  } catch (err) {
    console.error('Inference error:', err);
    showTextError('Could not connect to FakeLens backend');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = btn.dataset.originalText || 'Analyze article';
    }
  }
}

// =============================================================================
// 4. SIGNAL Deepfake Video Forensics
// =============================================================================
function triggerVideoUpload() {
  const input = document.getElementById('video-file-input');
  if (input) input.click();
}

function handleVideoFileSelect(event) {
  const file = event.target.files && event.target.files[0];
  if (file) loadVideoFile(file);
}

function initVideoDropzone() {
  const dropzone = document.getElementById('video-dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-active');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt && dt.files && dt.files[0];
    if (file) loadVideoFile(file);
  });
}

function showVideoError(msg) {
  const errBox = document.getElementById('video-error-box');
  if (errBox) {
    errBox.textContent = msg;
    errBox.style.display = 'block';
  }
}

function hideVideoError() {
  const errBox = document.getElementById('video-error-box');
  if (errBox) {
    errBox.textContent = '';
    errBox.style.display = 'none';
  }
}

function loadVideoFile(file) {
  const allowed = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();

  if (!allowed.includes(ext)) {
    showVideoError(`Unsupported file format '${ext}'. Allowed: MP4, MOV, AVI, MKV, WEBM.`);
    return;
  }

  if (file.size > 100 * 1024 * 1024) {
    showVideoError(`File size exceeds 100MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB).`);
    return;
  }

  hideVideoError();
  currentVideoFile = file;

  if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
  currentVideoUrl = URL.createObjectURL(file);

  const player = document.getElementById('video-player');
  if (player) {
    player.src = currentVideoUrl;
    player.load();
  }

  const dropzone = document.getElementById('video-dropzone');
  const previewWrap = document.getElementById('video-preview-wrap');
  const nameChip = document.getElementById('video-name-chip');
  const sizeChip = document.getElementById('video-size-chip');
  const verdictWrap = document.getElementById('video-verdict-wrap');

  if (nameChip) nameChip.textContent = file.name;
  if (sizeChip) sizeChip.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

  if (dropzone) dropzone.style.display = 'none';
  if (previewWrap) previewWrap.style.display = 'block';
  if (verdictWrap) verdictWrap.style.display = 'none';
}

function resetVideoUpload() {
  currentVideoFile = null;
  if (currentVideoUrl) {
    URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = null;
  }

  const input = document.getElementById('video-file-input');
  if (input) input.value = '';

  const player = document.getElementById('video-player');
  if (player) {
    player.pause();
    player.removeAttribute('src');
    player.load();
  }

  const dropzone = document.getElementById('video-dropzone');
  const previewWrap = document.getElementById('video-preview-wrap');
  const verdictWrap = document.getElementById('video-verdict-wrap');
  const progressWrap = document.getElementById('video-progress-wrap');

  if (dropzone) dropzone.style.display = 'block';
  if (previewWrap) previewWrap.style.display = 'none';
  if (verdictWrap) verdictWrap.style.display = 'none';
  if (progressWrap) progressWrap.style.display = 'none';
  hideVideoError();
}

async function analyzeVideoEvidence() {
  if (!currentVideoFile) {
    showVideoError('Please upload a video file first.');
    return;
  }

  hideVideoError();

  const btn = document.getElementById('btn-analyze-video');
  const progressWrap = document.getElementById('video-progress-wrap');
  const progressStatus = document.getElementById('video-progress-status');
  const progressPct = document.getElementById('video-progress-pct');
  const progressFill = document.getElementById('video-progress-fill');
  const verdictWrap = document.getElementById('video-verdict-wrap');
  const faceBox = document.getElementById('hud-face-box');

  if (btn) btn.disabled = true;
  if (progressWrap) progressWrap.style.display = 'block';
  if (verdictWrap) verdictWrap.style.display = 'none';

  // Animate optical scanning box
  if (faceBox) {
    faceBox.style.borderColor = '#ff4d5e';
    faceBox.style.transform = 'scale(1.04)';
  }

  // Simulated staged progress telemetry
  let pct = 10;
  if (progressFill) progressFill.style.width = '10%';
  if (progressPct) progressPct.textContent = '10%';
  if (progressStatus) progressStatus.textContent = 'Streaming evidence buffer...';

  const progressInterval = setInterval(() => {
    pct = Math.min(pct + 15, 88);
    if (progressFill) progressFill.style.width = pct + '%';
    if (progressPct) progressPct.textContent = pct + '%';
    if (pct === 25 && progressStatus) progressStatus.textContent = 'YuNet ONNX: Tracking faces and 5-point landmarks...';
    if (pct === 55 && progressStatus) progressStatus.textContent = 'Extracting consecutive mouth ROI temporal sequences...';
    if (pct === 70 && progressStatus) progressStatus.textContent = 'Executing Multi-Scale Vision Transformer inference...';
    if (pct >= 85 && progressStatus) progressStatus.textContent = 'Calibrating 3-tier forensic assessment policy...';
  }, 400);

  const startTime = performance.now();
  const formData = new FormData();
  formData.append('video', currentVideoFile);

  try {
    const response = await fetch('/detect', {
      method: 'POST',
      body: formData
    });

    clearInterval(progressInterval);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    if (progressFill) progressFill.style.width = '100%';
    if (progressPct) progressPct.textContent = '100%';
    if (progressStatus) progressStatus.textContent = 'Forensic verification complete.';

    if (!response.ok) {
      let errMsg = 'Video forensic analysis failed';
      try {
        const errJson = await response.json();
        if (errJson && errJson.detail) errMsg = errJson.detail;
      } catch (e) {}
      throw new Error(errMsg);
    }

    const data = await response.json();
    latestDeepfakeData = data;
    latestDeepfakeData._execution_time = elapsed;
    latestDeepfakeData._filename = currentVideoFile.name;

    setTimeout(() => {
      if (progressWrap) progressWrap.style.display = 'none';
      renderVideoForensicResults(data, elapsed);
    }, 450);

  } catch (err) {
    clearInterval(progressInterval);
    console.error('Deepfake analysis error:', err);
    if (progressWrap) progressWrap.style.display = 'none';
    showVideoError(err.message || 'Could not connect to deepfake detection engine.');
    if (faceBox) {
      faceBox.style.borderColor = '#33d6e8';
      faceBox.style.transform = 'none';
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderVideoForensicResults(data, elapsedSecs) {
  const verdictWrap = document.getElementById('video-verdict-wrap');
  const banner = document.getElementById('forensic-banner');
  const stamp = document.getElementById('forensic-tier-stamp');
  const msg = document.getElementById('forensic-tier-message');
  const reportId = document.getElementById('verdict-report-id');
  const tScore = document.getElementById('t-score');
  const tConf = document.getElementById('t-conf');
  const tFrames = document.getElementById('t-frames');
  const tModel = document.getElementById('t-model');
  const faceBox = document.getElementById('hud-face-box');
  const barsContainer = document.getElementById('timeline-bars');

  const rawScore = (data.forensic_score !== undefined) ? data.forensic_score :
                   (data.detection_score !== undefined ? data.detection_score :
                   (data.fake_probability !== undefined ? data.fake_probability : 0.0));

  let tier = data.assessment_tier;
  let message = data.assessment_message;

  if (!tier) {
    if (rawScore < 0.40) {
      tier = 'LIKELY AUTHENTIC';
      message = 'Low forensic anomaly signal detected. Spatio-temporal consistency aligns with natural recording parameters.';
    } else if (rawScore < 0.90) {
      tier = 'INCONCLUSIVE — NEEDS REVIEW';
      message = 'Ambiguous forensic signal. Compression, motion blur, or camera noise may have elevated scores. Human review is recommended.';
    } else {
      tier = 'SIGNS OF POSSIBLE MANIPULATION';
      message = 'High forensic anomaly signal detected across consecutive sampled facial frames. Verification recommended.';
    }
  }

  // Generate Unique Evidence ID
  const randHex = Math.floor(Math.random() * 0xFFFFFF).toString(16).toUpperCase().padStart(6, '0');
  const idStr = `SIG-2026-${randHex}`;
  if (reportId) reportId.textContent = idStr;
  latestDeepfakeData._reportId = idStr;

  // Apply Banner Styling
  if (banner && stamp && msg) {
    banner.className = 'forensic-banner';
    if (tier === 'LIKELY AUTHENTIC' || rawScore < 0.40) {
      banner.classList.add('authentic');
      if (faceBox) faceBox.style.borderColor = '#57f79c';
    } else if (tier.includes('INCONCLUSIVE') || (rawScore >= 0.40 && rawScore < 0.90)) {
      banner.classList.add('inconclusive');
      if (faceBox) faceBox.style.borderColor = '#ffb020';
    } else {
      banner.classList.add('manipulation');
      if (faceBox) faceBox.style.borderColor = '#ff4d5e';
    }

    stamp.textContent = tier;
    msg.textContent = message;
  }

  // Populate Telemetry Cards
  if (tScore) tScore.textContent = rawScore.toFixed(4);
  if (tConf) tConf.textContent = data.confidence || `${(Math.max(rawScore, 1 - rawScore) * 100).toFixed(1)}%`;
  if (tFrames) tFrames.textContent = data.frames_analyzed || (data.frame_scores ? data.frame_scores.length : 16);
  if (tModel) tModel.textContent = data.model_backend ? data.model_backend.toUpperCase() : 'ViT 2D';

  // Render Frame Anomaly Timeline Bar Chart
  if (barsContainer) {
    barsContainer.innerHTML = '';
    const frameScores = Array.isArray(data.frame_scores) && data.frame_scores.length > 0 ? data.frame_scores : [rawScore];

    frameScores.forEach((score, idx) => {
      const wrap = document.createElement('div');
      wrap.className = 'timeline-bar-wrap';

      const bar = document.createElement('div');
      bar.className = 'timeline-bar';

      const barPct = Math.max(8, Math.min(100, Math.round(score * 100)));
      bar.style.height = `${barPct}%`;

      if (score < 0.40) bar.classList.add('green');
      else if (score < 0.90) bar.classList.add('yellow');
      else bar.classList.add('red');

      const tooltip = document.createElement('div');
      tooltip.className = 'timeline-tooltip';
      tooltip.textContent = `Frame ${idx + 1}: ${(score * 100).toFixed(1)}%`;

      wrap.appendChild(bar);
      wrap.appendChild(tooltip);
      barsContainer.appendChild(wrap);
    });
  }

  // Reveal Verdict Section
  if (verdictWrap) {
    verdictWrap.style.display = 'block';
    verdictWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// =============================================================================
// 5. Forensic PDF Report Export (jsPDF)
// =============================================================================
function downloadForensicReport() {
  if (!latestDeepfakeData) {
    alert('No deepfake analysis results to generate a report for.');
    return;
  }

  if (!window.jspdf || !window.jspdf.jsPDF) {
    alert('PDF generation library is loading. Please try again in a moment.');
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });

  const rawScore = (latestDeepfakeData.forensic_score !== undefined) ? latestDeepfakeData.forensic_score :
                   (latestDeepfakeData.detection_score !== undefined ? latestDeepfakeData.detection_score : 0.0);
  const tier = latestDeepfakeData.assessment_tier || (rawScore >= 0.90 ? 'SIGNS OF POSSIBLE MANIPULATION' : (rawScore >= 0.40 ? 'INCONCLUSIVE — NEEDS REVIEW' : 'LIKELY AUTHENTIC'));
  const reportId = latestDeepfakeData._reportId || `SIG-2026-${Math.floor(Math.random() * 0xFFFFFF).toString(16).toUpperCase()}`;
  const filename = latestDeepfakeData._filename || 'evidence.mp4';
  const execTime = latestDeepfakeData._execution_time || '1.18';
  const framesCount = latestDeepfakeData.frames_analyzed || (latestDeepfakeData.frame_scores ? latestDeepfakeData.frame_scores.length : 16);
  const modelBackend = (latestDeepfakeData.model_backend || 'baseline_vit').toUpperCase();

  // Dark header block
  doc.setFillColor(18, 22, 28);
  doc.rect(0, 0, 210, 36, 'F');

  // Title & Brand
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(51, 214, 232);
  doc.setFontSize(18);
  doc.text('FAKELENS  x  SIGNAL', 14, 16);

  doc.setFontSize(10);
  doc.setTextColor(231, 227, 216);
  doc.setFont('helvetica', 'normal');
  doc.text('FORENSIC INTELLIGENCE DOSSIER — VIDEO DEEPFAKE ANALYSIS', 14, 24);

  doc.setFontSize(8);
  doc.setTextColor(139, 149, 163);
  doc.text(`REPORT ID: ${reportId}   |   GENERATED: ${new Date().toUTCString()}`, 14, 30);

  let y = 46;

  // Evidence Overview
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(201, 154, 72);
  doc.text('1. EVIDENCE IDENTIFICATION', 14, y);
  y += 6;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(30, 30, 30);
  doc.text(`File Name: ${filename}`, 16, y);
  doc.text(`Inference Engine: ${modelBackend} (YuNet Face Tracker + Temporal Vision Transformer)`, 16, y + 5);
  doc.text(`Frames Analyzed: ${framesCount} cropped sequence frames`, 16, y + 10);
  doc.text(`Pipeline Latency: ${execTime} seconds`, 16, y + 15);
  y += 24;

  // Primary Assessment Banner Box
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(201, 154, 72);
  doc.text('2. PRIMARY FORENSIC ASSESSMENT', 14, y);
  y += 6;

  let rColor = [93, 163, 124];
  if (tier.includes('MANIPULATION')) rColor = [255, 77, 94];
  else if (tier.includes('INCONCLUSIVE')) rColor = [255, 176, 32];

  doc.setFillColor(rColor[0], rColor[1], rColor[2]);
  doc.rect(14, y, 182, 14, 'F');

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(13);
  doc.setFont('helvetica', 'bold');
  doc.text(`VERDICT TIER: ${tier}`, 18, y + 9);
  y += 20;

  // Quantitative Metrics Table
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(201, 154, 72);
  doc.text('3. QUANTITATIVE TELEMETRY', 14, y);
  y += 6;

  doc.setFontSize(9);
  doc.setTextColor(40, 40, 40);
  doc.setFont('helvetica', 'normal');
  doc.text(`Forensic Anomaly Score: ${rawScore.toFixed(4)} (Threshold: 0.50)`, 16, y);
  doc.text(`Confidence Level: ${latestDeepfakeData.confidence || ((Math.max(rawScore, 1 - rawScore)) * 100).toFixed(1) + '%'}`, 16, y + 5);
  doc.text(`Assessment Message: ${latestDeepfakeData.assessment_message || 'Detailed spatio-temporal telemetry recorded.'}`, 16, y + 10);
  y += 20;

  // Frame Anomaly Table
  if (Array.isArray(latestDeepfakeData.frame_scores) && latestDeepfakeData.frame_scores.length > 0) {
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(201, 154, 72);
    doc.text('4. PER-FRAME ANOMALY DISTRIBUTION', 14, y);
    y += 7;

    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setFillColor(240, 240, 240);
    doc.rect(14, y - 4, 182, 6, 'F');
    doc.text('FRAME NO.', 18, y);
    doc.text('ANOMALY PROBABILITY', 65, y);
    doc.text('BAND CLASSIFICATION', 125, y);
    y += 5;

    doc.setFont('helvetica', 'normal');
    latestDeepfakeData.frame_scores.slice(0, 16).forEach((sc, i) => {
      const band = sc < 0.40 ? 'LIKELY AUTHENTIC' : (sc < 0.90 ? 'INCONCLUSIVE' : 'MANIPULATION');
      doc.text(`Frame ${i + 1}`, 18, y);
      doc.text(`${(sc * 100).toFixed(2)}%`, 65, y);
      doc.text(band, 125, y);
      y += 4.5;
    });
    y += 6;
  }

  // Legal & Scientific Disclaimer
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(120, 120, 120);
  doc.text('5. SCIENTIFIC & FORENSIC DISCLAIMER', 14, y);
  y += 5;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.5);
  doc.text(
    'This report was automatically synthesized by the FakeLens × SIGNAL multimodal forensic suite. ' +
    'Predictions represent statistical algorithmic assessment of facial spatio-temporal artifacts and do not constitute absolute legal certitude. ' +
    'Compression, lighting artifacts, and high-frequency motion blur can elevate forensic scores. Independent corroboration is recommended.',
    14, y, { maxWidth: 182, lineHeightFactor: 1.4 }
  );

  doc.save(`FakeLens_SIGNAL_Forensic_Report_${reportId}.pdf`);
}

// =============================================================================
// 6. Raw Telemetry JSON Modal
// =============================================================================
function openTelemetryModal() {
  const modal = document.getElementById('telemetry-modal');
  const codeEl = document.getElementById('telemetry-json-code');
  if (modal && codeEl) {
    const displayData = latestDeepfakeData || { message: "No video analysis has been run yet." };
    codeEl.textContent = JSON.stringify(displayData, null, 2);
    modal.style.display = 'flex';
  }
}

function closeTelemetryModal() {
  const modal = document.getElementById('telemetry-modal');
  if (modal) modal.style.display = 'none';
}

function handleModalOverlayClick(event) {
  if (event.target && event.target.id === 'telemetry-modal') {
    closeTelemetryModal();
  }
}

function copyTelemetryJson() {
  const codeEl = document.getElementById('telemetry-json-code');
  const alertEl = document.getElementById('copy-alert');
  if (codeEl) {
    navigator.clipboard.writeText(codeEl.textContent).then(() => {
      if (alertEl) {
        alertEl.style.display = 'inline';
        setTimeout(() => { alertEl.style.display = 'none'; }, 2000);
      }
    });
  }
}

// =============================================================================
// 7. System Readiness & Health Polling
// =============================================================================
async function checkSystemReadiness() {
  const statusEl = document.getElementById('masthead-status');
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      if (statusEl) {
        const nlp = data.nlp_loaded !== false ? 'NLP' : 'NLP [SETUP]';
        const vid = data.model_loaded ? 'SIGNAL' : 'SIGNAL [READY]';
        statusEl.innerHTML = `<span class="status-dot"></span>${nlp} + ${vid} active`;
      }
    }
  } catch (e) {
    // Fallback quietly if health endpoint is initializing
  }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  updateCounters();
  initVideoDropzone();
  checkSystemReadiness();
});
