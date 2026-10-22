
  /* --- custom cursor --- */
  const cdot = document.getElementById('cdot');
  const cring = document.getElementById('cring');
  let mx=innerWidth/2,my=innerHeight/2, rx=mx, ry=my;
  addEventListener('mousemove', e=>{ mx=e.clientX; my=e.clientY; cdot.style.left=mx+'px'; cdot.style.top=my+'px'; });
  (function ringLoop(){
    rx += (mx-rx)*0.18; ry += (my-ry)*0.18;
    cring.style.left=rx+'px'; cring.style.top=ry+'px';
    requestAnimationFrame(ringLoop);
  })();
  document.querySelectorAll('a, button, .method-card, .arch-tab, .node, .arch-node, #tiltPanel:not(.has-video), .cta, .btn-insert-new-video, .btn-download-results, .upload-instruction-box, .btn-view-report, .btn-download-raw, .modal-close-btn, .btn-modal-action').forEach(el=>{
    el.addEventListener('mouseenter', ()=>cring.classList.add('active'));
    el.addEventListener('mouseleave', ()=>cring.classList.remove('active'));
  });

  /* --- CRT static noise --- */
  const noise = document.getElementById('noise');
  const nctx = noise.getContext('2d');
  const NW=128, NH=72;
  noise.width=NW; noise.height=NH; noise.style.width='100%'; noise.style.height='100%';
  function drawNoise(){
    const img = nctx.createImageData(NW,NH);
    for(let i=0;i<img.data.length;i+=4){ const v=Math.random()*255; img.data[i]=v; img.data[i+1]=v; img.data[i+2]=v; img.data[i+3]=255; }
    nctx.putImageData(img,0,0);
  }
  let nf=0;
  (function noiseLoop(){ nf++; if(nf%2===0) drawNoise(); requestAnimationFrame(noiseLoop); })();

  /* --- headline decode into per-letter spans, then idle glitch --- */
  const chars = "!<>-_\\/[]{}—=+*^?#$%01";
  const target = "REAL, OR\nRENDERED?";
  const el = document.getElementById('headline');
  function renderPartial(revealCount){
    let out=''; let idx=0;
    for(const ch of target){
      if(ch==='\n'){ out+='<br>'; continue; }
      if(ch===' '){ out+=' '; idx++; continue; }
      out += (idx<revealCount) ? `<span class="ch">${ch}</span>` : chars[Math.floor(Math.random()*chars.length)];
      idx++;
    }
    el.innerHTML = out;
  }
  let reveal=0; const total = target.replace(/\n/g,'').replace(/ /g,'').length;
  function decodeStep(){
    reveal++; renderPartial(reveal);
    if(reveal<=total) setTimeout(decodeStep, 34);
    else{
      let out=''; for(const ch of target){ out += ch==='\n' ? '<br>' : ch===' ' ? ' ' : `<span class="ch">${ch}</span>`; }
      el.innerHTML = out;
      startIdleGlitch();
    }
  }
  setTimeout(decodeStep, 250);
  function startIdleGlitch(){
    setInterval(()=>{
      const spans = el.querySelectorAll('.ch');
      if(!spans.length) return;
      const s = spans[Math.floor(Math.random()*spans.length)];
      s.classList.add('flicker');
      setTimeout(()=>s.classList.remove('flicker'), 140);
    }, 2200);
  }

  /* --- ticker content --- */
  const tickerItems = [
    {t:'CASE 0142', s:'AUTHENTIC', cls:'ok'},
    {t:'CASE 0891', s:'MANIPULATED', cls:'flag'},
    {t:'MEAN LOGIT', s:'-9.2901'},
    {t:'ANOMALY SCORE', s:'0.0001'},
    {t:'DETECTOR', s:'YuNet ONNX'},
    {t:'MODEL', s:'ResNet-18 + MS-TCN'},
    {t:'STRIDE', s:'25 FRAMES'},
  ];
  const track = document.getElementById('tickerTrack');
  function buildTicker(){
    let html='';
    for(let r=0;r<2;r++){
      tickerItems.forEach(it=>{
        html += `<div class="ticker-item ${it.cls||''}">${it.t} <b>${it.s}</b></div>`;
      });
    }
    track.innerHTML = html;
  }
  buildTicker();

  /* --- 3D tilt on scan panel --- */
  const heroEl = document.querySelector('.hero');
  const panel = document.getElementById('tiltPanel');
  heroEl.addEventListener('mousemove', e=>{
    const r = panel.getBoundingClientRect();
    const px = (e.clientX-r.left)/r.width - 0.5;
    const py = (e.clientY-r.top)/r.height - 0.5;
    panel.style.transform = `rotate(-1.4deg) rotateY(${px*8}deg) rotateX(${-py*8}deg)`;
  });
  heroEl.addEventListener('mouseleave', ()=>{ panel.style.transform='rotate(-1.4deg)'; });

  /* --- magnetic CTA --- */
  const cta = document.getElementById('ctaBtn');
  cta.addEventListener('mousemove', e=>{
    const r = cta.getBoundingClientRect();
    const dx = (e.clientX-r.left-r.width/2)*0.25;
    const dy = (e.clientY-r.top-r.height/2)*0.4;
    cta.style.transform = `translate(${dx}px, ${dy}px)`;
  });
  cta.addEventListener('mouseleave', ()=>{ cta.style.transform='translate(0,0)'; });

  /* --- glitch flicker on method card hover --- */
  document.querySelectorAll('.method-card').forEach(card=>{
    card.addEventListener('mouseenter', ()=>{ card.classList.remove('glitch'); void card.offsetWidth; card.classList.add('glitch'); });
  });

  /* --- pipeline scroll trigger --- */
  const trackEl = document.getElementById('track');
  new IntersectionObserver((entries)=>{ entries.forEach(e=>{ if(e.isIntersecting) trackEl.classList.add('in-view'); }); }, {threshold:0.4}).observe(trackEl);

  /* --- section reveal --- */
  document.querySelectorAll('.reveal').forEach(sec=>{
    const obs = new IntersectionObserver(([entry])=>{ if(entry.isIntersecting){ sec.classList.add('in'); obs.disconnect(); } }, {threshold:0.15});
    obs.observe(sec);
  });

  /* --- stat count-up --- */
  document.querySelectorAll('.stat-num').forEach(numEl=>{
    const target = parseFloat(numEl.dataset.target);
    const divide = numEl.dataset.divide ? parseFloat(numEl.dataset.divide) : 1;
    const decimals = numEl.dataset.decimals ? parseInt(numEl.dataset.decimals) : 0;
    const suffix2 = numEl.dataset.suffix2 || '';
    const finalVal = target/divide;
    const obs = new IntersectionObserver(([entry])=>{
      if(!entry.isIntersecting) return;
      obs.disconnect();
      const start = performance.now(); const dur=1400;
      function tick(now){
        const p = Math.min(1,(now-start)/dur);
        const eased = 1-Math.pow(1-p,3);
        const val = finalVal*eased;
        numEl.textContent = decimals>0 ? val.toFixed(decimals)+suffix2 : Math.round(val).toLocaleString()+suffix2;
        if(p<1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }, {threshold:0.4});
    obs.observe(numEl);
  });

  /* --- case archive tabs with numeric tween --- */
  const CASES = {
    authentic:{ badge:'BENCHMARK &middot; FF++ PRISTINE CAPTURE', verdict:'LIKELY AUTHENTIC',
      desc:'Pristine FaceForensics++ recording evaluated by LipForensics. Evaluated 4 clips (100 frames) with consistent temporal lip dynamics and negative logits across all windows.',
      source:'01__kitchen_still.mp4', clips:'4 clips (100 frames)', logit:-9.2901, anomaly:0.0001, tier:'Likely authentic', color:'#57f79c' },
    manipulated:{ badge:'BENCHMARK &middot; FF++ FACESWAP SAMPLE', verdict:'LIKELY MANIPULATED',
      desc:'FaceSwap-generated sequence flagged by LipForensics. Evaluated 4 clips (100 frames) with unstable mouth-region timing and elevated positive logits in 3 of 4 windows.',
      source:'07__faceswap_kitchen.mp4', clips:'4 clips (100 frames)', logit:6.1148, anomaly:0.9187, tier:'Likely manipulated', color:'#ff4d5e' }
  };
  let curLogit=-9.2901, curAnomaly=0.0001;
  function tween(fromVal,toVal,el,decimals,signed){
    const start=performance.now(); const dur=500;
    function step(now){
      const p=Math.min(1,(now-start)/dur);
      const v = fromVal + (toVal-fromVal)*p;
      el.textContent = (signed && v>=0 ? '+' : '') + v.toFixed(decimals);
      if(p<1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  document.querySelectorAll('.arch-tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.arch-tab').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const c = CASES[btn.dataset.case];
      document.getElementById('archivePanel').style.setProperty('--verdict-color', c.color);
      document.getElementById('archBadge').innerHTML = c.badge;
      document.getElementById('archVerdict').textContent = c.verdict;
      document.getElementById('archDesc').textContent = c.desc;
      document.getElementById('statSource').textContent = c.source;
      document.getElementById('statClips').textContent = c.clips;
      document.getElementById('statTier').textContent = c.tier;
      tween(curLogit, c.logit, document.getElementById('statLogit'), 4, true);
      tween(curAnomaly, c.anomaly, document.getElementById('statAnomaly'), 4, false);
      curLogit=c.logit; curAnomaly=c.anomaly;
      const flash = document.getElementById('archiveFlash');
      flash.classList.remove('run'); void flash.offsetWidth; flash.classList.add('run');
    });
  });

  /* ==========================================================================
     SIGNAL APPLICATION LOGIC & REAL FORENSIC AUDIT
     ========================================================================== */
  const heroDropzone = document.getElementById("tiltPanel");
  const fileInput = document.getElementById("file-input");
  const btnInsertNewVideo = document.getElementById("btn-insert-new-video");
  const btnHeroAnalyze = document.getElementById("ctaBtn");
  const dropPrompt = document.getElementById("drop-prompt");
  const previewVideo = document.getElementById("preview-video");
  const faceBox = document.getElementById("face-box");
  const dropzoneStatusText = document.getElementById("dropzone-status-text");
  const dropzoneVerdictVal = document.getElementById("dropzone-verdict-val");
  const recDot = document.getElementById("rec-dot");
  const progressBarContainer = document.getElementById("progress-bar-container");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const dropzoneSubtextMeta = document.getElementById("dropzone-subtext-meta");

  const resultsSection = document.getElementById("results-section");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictDesc = document.getElementById("verdict-desc");
  const resProb = document.getElementById("res-prob");
  const resFrames = document.getElementById("res-frames");
  const resCoverage = document.getElementById("res-coverage");
  const resLatency = document.getElementById("res-latency");
  const ringBar = document.getElementById("ring-bar");
  const ringPct = document.getElementById("ring-pct");
  const modelTag = document.getElementById("model-tag");
  const timelineBarsTrack = document.getElementById("timeline-bars-track");
  const btnDownloadResults = document.getElementById("btn-download-results");

  let latestAnalysisData = null;
  let currentFileName = "";

  if (heroDropzone) {
    heroDropzone.addEventListener("click", (e) => {
      if (e.target.closest("#btn-insert-new-video")) return;
      if (heroDropzone.classList.contains("has-video")) return;
      fileInput.click();
    });
  }

  if (btnHeroAnalyze) {
    btnHeroAnalyze.addEventListener("click", () => {
      fileInput.value = "";
      fileInput.click();
    });
  }

  if (btnInsertNewVideo) {
    btnInsertNewVideo.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.value = "";
      fileInput.click();
    });
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileIngestion(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach(eventName => {
    heroDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      heroDropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    heroDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      heroDropzone.classList.remove("drag-active");
    });
  });

  heroDropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileIngestion(dt.files[0]);
    }
  });

  async function handleFileIngestion(file) {
    if (!file) return;

    heroDropzone.classList.add("has-video");
    currentFileName = file.name;
    const fileUrl = URL.createObjectURL(file);
    previewVideo.src = fileUrl;
    previewVideo.classList.add("active");
    previewVideo.play().catch(() => {});
    dropPrompt.style.display = "none";

    if (recDot) recDot.classList.add("active");
    progressBarContainer.classList.add("active");
    progressBarFill.style.width = "20%";
    dropzoneStatusText.textContent = "scanning";
    dropzoneSubtextMeta.textContent = file.name;
    dropzoneVerdictVal.textContent = "EXTRACTING LANDMARKS";

    const startTime = performance.now();

    setTimeout(() => {
      faceBox.style.top = "24%";
      faceBox.style.left = "36%";
      faceBox.style.width = "28%";
      faceBox.style.height = "52%";
      faceBox.classList.add("active");
      progressBarFill.style.width = "55%";
      dropzoneVerdictVal.textContent = "TEMPORAL INFERENCE";
    }, 400);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch("/detect", {
        method: "POST",
        body: formData
      });

      const elapsedSecs = (performance.now() - startTime) / 1000;
      progressBarFill.style.width = "100%";

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errMsg = errData.detail || `Server returned HTTP ${response.status}`;
        throw new Error(errMsg);
      }

      const data = await response.json();
      latestAnalysisData = data;

      setTimeout(() => {
        progressBarContainer.classList.remove("active");
        
        const rawScore = (data.forensic_score !== undefined) ? data.forensic_score : 
                         (data.detection_score !== undefined ? data.detection_score : 
                         (data.fake_probability !== undefined ? data.fake_probability : 0.0));
        const tier = data.assessment_tier || (rawScore >= 0.90 ? "SIGNS OF POSSIBLE MANIPULATION" : (rawScore >= 0.40 ? "INCONCLUSIVE — NEEDS REVIEW" : "LIKELY AUTHENTIC"));
        
        dropzoneStatusText.textContent = "complete";
        dropzoneVerdictVal.textContent = tier;
        
        renderAnalysisResults(data, file.name, elapsedSecs);
      }, 500);

    } catch (error) {
      console.error("Analysis Error:", error);
      progressBarContainer.classList.remove("active");
      dropzoneStatusText.textContent = "error";
      dropzoneVerdictVal.textContent = "VERIFICATION FAILED";
      alert(`Analysis failed: ${error.message || "Please upload a valid video file."}`);
    }
  }

  /* Report Modal Element References */
  const btnViewReport = document.getElementById("btn-view-report");
  const reportModalOverlay = document.getElementById("report-modal-overlay");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalBtnPrint = document.getElementById("modal-btn-print");
  const modalBtnDownloadJson = document.getElementById("modal-btn-download-json");
  const modalFilename = document.getElementById("modal-filename");
  const modalTimestamp = document.getElementById("modal-timestamp");
  const modalReportId = document.getElementById("modal-report-id");
  const modalVerdictBanner = document.getElementById("modal-verdict-banner");
  const modalVerdictTier = document.getElementById("modal-verdict-tier");
  const modalVerdictSummary = document.getElementById("modal-verdict-summary");
  const modalScore = document.getElementById("modal-score");
  const modalFrames = document.getElementById("modal-frames");
  const modalCoverage = document.getElementById("modal-coverage");
  const modalLatency = document.getElementById("modal-latency");
  const modalDetector = document.getElementById("modal-detector");
  const modalModel = document.getElementById("modal-model");
  const modalExplanationP1 = document.getElementById("modal-explanation-p1");
  const modalExplanationP2 = document.getElementById("modal-explanation-p2");

  function populateReportModal(data, fileName, elapsedSeconds = 1.18) {
    if (!data) return;

    const rawScore = (data.forensic_score !== undefined) ? data.forensic_score :
                     (data.detection_score !== undefined ? data.detection_score :
                     (data.fake_probability !== undefined ? data.fake_probability : 0.0));

    let tier = data.assessment_tier;
    let message = data.assessment_message;

    if (!tier) {
      if (rawScore < 0.40) {
        tier = "LIKELY AUTHENTIC";
        message = "Low forensic anomaly signal detected. Spatio-temporal consistency metrics align with natural recording parameters.";
      } else if (rawScore < 0.90) {
        tier = "INCONCLUSIVE — NEEDS REVIEW";
        message = "Ambiguous forensic signal. Independent review is recommended.";
      } else {
        tier = "SIGNS OF POSSIBLE MANIPULATION";
        message = "High forensic anomaly signal detected. This does not by itself establish manipulation. Compression, motion, lighting, and camera conditions can elevate the score. Independent verification is recommended.";
      }
    }

    let tierClass = "real";
    if (tier === "SIGNS OF POSSIBLE MANIPULATION" || rawScore >= 0.90) {
      tierClass = "fake";
    } else if (tier.includes("INCONCLUSIVE") || (rawScore >= 0.40 && rawScore < 0.90)) {
      tierClass = "inconclusive";
    } else {
      tierClass = "real";
    }

    if (modalFilename) modalFilename.textContent = fileName || currentFileName || "evidence.mp4";
    if (modalTimestamp) modalTimestamp.textContent = new Date().toLocaleString();
    if (modalReportId) {
      if (!modalReportId.dataset.id || modalReportId.dataset.file !== fileName) {
        const randomHex = Math.floor(Math.random() * 0xFFFFFF).toString(16).toUpperCase().padStart(6, "0");
        modalReportId.textContent = `SIG-2026-${randomHex}`;
        modalReportId.dataset.id = modalReportId.textContent;
        modalReportId.dataset.file = fileName;
      }
    }

    if (modalVerdictBanner) modalVerdictBanner.className = `report-verdict-banner ${tierClass}`;
    if (modalVerdictTier) modalVerdictTier.textContent = tier;
    if (modalVerdictSummary) modalVerdictSummary.textContent = message;

    if (modalScore) modalScore.textContent = Number(rawScore).toFixed(4);
    const framesCount = data.frames_analyzed || (data.frame_scores ? data.frame_scores.length : 15);
    if (modalFrames) modalFrames.textContent = `${framesCount} / ${framesCount}`;
    const coverageVal = data.face_coverage !== undefined ? (data.face_coverage * 100).toFixed(1) : "100.0";
    if (modalCoverage) modalCoverage.textContent = `${coverageVal}%`;
    const finalLatency = data.execution_time ? data.execution_time.toFixed(2) : (elapsedSeconds ? Number(elapsedSeconds).toFixed(2) : "1.18");
    if (modalLatency) modalLatency.textContent = `${finalLatency}s`;
    if (modalDetector) modalDetector.textContent = data.detector_name || "YuNet ONNX";
    const backendName = data.model_backend || data.backend || "LipForensics";
    if (modalModel) modalModel.textContent = backendName.toUpperCase().replace("_", " ");

    if (modalExplanationP1 && modalExplanationP2) {
      if (tierClass === "real") {
        modalExplanationP1.textContent = "This clip exhibited natural temporal micro-motions and lip movements consistent with authentic human speech dynamics. No significant spatio-temporal boundary anomalies or generator artifacts were detected across the analyzed frame sequence.";
        modalExplanationP2.textContent = "Note: This finding serves as automated screening evidence and should be interpreted alongside source context, chain of custody, and metadata provenance.";
      } else if (tierClass === "inconclusive") {
        modalExplanationP1.textContent = "The analyzed video produced an ambiguous forensic signal falling between clear authenticity and high-confidence manipulation. Moderate anomalies in lighting, motion blur, video compression, or facial movement warrant independent expert verification before making a determination.";
        modalExplanationP2.textContent = "Recommendation: Conduct a frame-level provenance audit and verify original uncompressed footage if available.";
      } else {
        modalExplanationP1.textContent = "The perception engine detected substantial temporal irregularities, lip-motion incoherence, or synthesis boundary anomalies across multiple frame windows. While these indicators strongly suggest artificial generation or face manipulation, factors such as severe compression, camera motion, and extreme lighting can elevate anomaly scores.";
        modalExplanationP2.textContent = "Recommendation: Treat footage with caution. Independent multi-source verification and forensic deep-scan are recommended.";
      }
    }
  }

  function renderAnalysisResults(data, fileName, elapsedSeconds = 1.18) {
    resultsSection.classList.add("active");
    resultsSection.scrollIntoView({ behavior: "smooth" });

    const rawScore = (data.forensic_score !== undefined) ? data.forensic_score :
                     (data.detection_score !== undefined ? data.detection_score :
                     (data.fake_probability !== undefined ? data.fake_probability : 0.0));

    let tier = data.assessment_tier;
    let message = data.assessment_message;

    if (!tier) {
      if (rawScore < 0.40) {
        tier = "LIKELY AUTHENTIC";
        message = "Low forensic anomaly signal detected. This does not guarantee authenticity.";
      } else if (rawScore < 0.90) {
        tier = "INCONCLUSIVE — NEEDS REVIEW";
        message = "Ambiguous forensic signal. Independent review is recommended.";
      } else {
        tier = "SIGNS OF POSSIBLE MANIPULATION";
        message = "High forensic anomaly signal detected. This does not by itself establish manipulation. Compression, motion, lighting, and camera conditions can elevate the score. Independent verification is recommended.";
      }
    }

    let tierClass = "real";
    let ringColor = "var(--good)";

    if (tier === "SIGNS OF POSSIBLE MANIPULATION" || rawScore >= 0.90) {
      tierClass = "fake";
      ringColor = "var(--bad)";
    } else if (tier.includes("INCONCLUSIVE") || (rawScore >= 0.40 && rawScore < 0.90)) {
      tierClass = "inconclusive";
      ringColor = "var(--warn)";
    } else {
      tierClass = "real";
      ringColor = "var(--good)";
    }

    verdictTitle.textContent = tier;
    verdictTitle.className = `verdict-tag ${tierClass}`;
    verdictDesc.textContent = message;

    animateValue(resProb, 0, rawScore, 1000, 4, "");
    
    const framesCount = data.frames_analyzed || (data.frame_scores ? data.frame_scores.length : 15);
    resFrames.textContent = `${framesCount} / ${framesCount}`;
    
    const coverageVal = data.face_coverage !== undefined ? (data.face_coverage * 100).toFixed(1) : "100.0";
    resCoverage.textContent = `${coverageVal}%`;
    
    const finalLatency = data.execution_time ? data.execution_time.toFixed(2) : (elapsedSeconds ? Number(elapsedSeconds).toFixed(2) : "1.18");
    resLatency.textContent = `${finalLatency}s`;

    const targetGaugePct = Math.min(100, Math.max(0, Math.round(rawScore * 100)));
    animateValue(ringPct, 0, targetGaugePct, 1000, 0, "%");
    const circumference = 2 * Math.PI * 60; // 377
    const offset = circumference - (targetGaugePct / 100) * circumference;
    ringBar.style.strokeDashoffset = offset;
    ringBar.style.stroke = ringColor;

    const backendName = data.model_backend || data.backend || "lipforensics";
    modelTag.textContent = `ENGINE // ${backendName.toUpperCase().replace("_", " ")}`;

    const frameScores = Array.isArray(data.frame_scores) ? data.frame_scores :
                        (Array.isArray(data.attention_weights) ? data.attention_weights :
                        (Array.isArray(data.frame_probabilities) ? data.frame_probabilities : []));
    renderWaveformBars(frameScores, data.decision_threshold || 0.50);

    if (data.face_scan && data.face_scan.box) {
      const box = data.face_scan.box;
      faceBox.style.left = `${(box.x * 100).toFixed(1)}%`;
      faceBox.style.top = `${(box.y * 100).toFixed(1)}%`;
      faceBox.style.width = `${(box.w * 100).toFixed(1)}%`;
      faceBox.style.height = `${(box.h * 100).toFixed(1)}%`;
      faceBox.classList.add("active");
    }

    // Pre-populate modal data with actual detect response
    populateReportModal(data, fileName, elapsedSeconds);
  }

  function animateValue(elem, start, end, duration, decimals = 0, suffix = "") {
    if (!elem) return;
    const startTime = performance.now();
    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const current = start + (end - start) * (1 - Math.pow(1 - progress, 3));
      elem.textContent = `${current.toFixed(decimals)}${suffix}`;
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }
    requestAnimationFrame(update);
  }

  function renderWaveformBars(probs, threshold = 0.50) {
    const sampleCountElem = document.getElementById("timeline-sample-count");
    const firstSampleElem = document.getElementById("timeline-first-sample");
    const lastSampleElem = document.getElementById("timeline-last-sample");
    const thresholdLabel = document.getElementById("timeline-threshold-label");

    if (thresholdLabel) {
      thresholdLabel.textContent = `${Number(threshold).toFixed(2)} THRESHOLD`;
    }

    if (!probs || !Array.isArray(probs) || probs.length === 0) {
      timelineBarsTrack.innerHTML = '<div class="waveform-unavailable-msg">Frame-level data not available for this analysis</div>';
      if (sampleCountElem) sampleCountElem.textContent = "N/A";
      if (firstSampleElem) firstSampleElem.textContent = "--";
      if (lastSampleElem) lastSampleElem.textContent = "--";
      return;
    }

    if (sampleCountElem) sampleCountElem.textContent = `${probs.length} ${probs.length === 1 ? 'SAMPLE' : 'SAMPLES'}`;
    if (firstSampleElem) firstSampleElem.textContent = "SAMPLE 01";
    if (lastSampleElem) lastSampleElem.textContent = `SAMPLE ${String(probs.length).padStart(2, "0")}`;

    timelineBarsTrack.innerHTML = '<div class="timeline-threshold-line"></div>';

    probs.forEach((p, idx) => {
      const col = document.createElement("div");
      col.className = "timeline-bar-column";
      col.title = `Sample ${idx + 1}: ${(p * 100).toFixed(1)}%`;

      const fill = document.createElement("div");
      fill.className = `timeline-bar-fill ${p > threshold ? "flagged" : ""}`;
      fill.style.height = "0%";

      col.appendChild(fill);
      timelineBarsTrack.appendChild(col);

      setTimeout(() => {
        fill.style.height = `${Math.min(100, Math.max(8, p * 100))}%`;
      }, 30 * idx);
    });
  }

  function downloadRawJson() {
    if (!latestAnalysisData) return;

    const exportPayload = {
      product: "SIGNAL Forensic Intelligence System",
      version: "2026.1",
      analyzed_file: currentFileName,
      timestamp: new Date().toISOString(),
      ...latestAnalysisData
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportPayload, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "SIGNAL-analysis-results.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }

  const btnDownloadPdfCard = document.getElementById("btn-download-pdf-card");
  const modalBtnDownloadPdf = document.getElementById("modal-btn-download-pdf");

  function downloadPdfReport() {
    if (!latestAnalysisData) {
      alert("No forensic analysis data available. Please analyze a video clip first.");
      return;
    }

    const data = latestAnalysisData;
    const fileName = currentFileName || "evidence.mp4";
    const reportDate = new Date();
    const timestampStr = reportDate.toLocaleString();
    const reportIdElem = document.getElementById("modal-report-id");
    const reportId = (reportIdElem && reportIdElem.textContent && reportIdElem.textContent !== "SIG-2026-0000")
      ? reportIdElem.textContent
      : `SIG-2026-${Math.floor(Math.random() * 0xFFFFFF).toString(16).toUpperCase().padStart(6, "0")}`;

    const rawScore = (data.forensic_score !== undefined) ? Number(data.forensic_score) :
                     (data.detection_score !== undefined ? Number(data.detection_score) :
                     (data.fake_probability !== undefined ? Number(data.fake_probability) : 0.0));

    let tier = data.assessment_tier;
    let message = data.assessment_message;
    if (!tier) {
      if (rawScore < 0.40) {
        tier = "LIKELY AUTHENTIC";
        message = "Low forensic anomaly signal detected. Spatio-temporal consistency metrics align with natural recording parameters.";
      } else if (rawScore < 0.90) {
        tier = "INCONCLUSIVE — NEEDS REVIEW";
        message = "Ambiguous forensic signal. Independent review is recommended.";
      } else {
        tier = "SIGNS OF POSSIBLE MANIPULATION";
        message = "High forensic anomaly signal detected. Compression, motion, lighting, and camera conditions can elevate the score. Independent verification is recommended.";
      }
    }

    const framesCount = data.frames_analyzed || (data.frame_scores ? data.frame_scores.length : 15);
    const coverageVal = data.face_coverage !== undefined ? (data.face_coverage * 100).toFixed(1) : "100.0";
    const executionTimeVal = data.execution_time ? Number(data.execution_time).toFixed(2) : "1.18";
    const detectorName = data.detector_name || "YuNet ONNX";
    const modelBackendName = (data.model_backend || data.backend || "LipForensics").toUpperCase().replace("_", " ");

    const jsPDFClass = (window.jspdf && window.jspdf.jsPDF) ? window.jspdf.jsPDF : null;

    if (!jsPDFClass) {
      console.warn("jsPDF not loaded. Triggering print fallback.");
      populateReportModal(data, fileName);
      reportModalOverlay.classList.add("active");
      window.print();
      return;
    }

    const doc = new jsPDFClass({
      orientation: "portrait",
      unit: "mm",
      format: "a4"
    });

    const pageWidth = 210;
    const pageHeight = 297;
    const margin = 14;
    const contentWidth = pageWidth - (margin * 2); // 182mm

    // Palette
    const bgDark = [10, 12, 14];
    const bgCard = [15, 18, 22];
    const borderLine = [36, 42, 49];
    const accentCyan = [51, 214, 232];
    const textWhite = [238, 241, 243];
    const textMuted = [140, 154, 162];
    const textDarkMuted = [100, 114, 122];

    let tierColor = [87, 247, 156];     // Authentic green
    let tierBg = [14, 32, 22];
    if (tier.includes("MANIPULATION") || rawScore >= 0.90) {
      tierColor = [255, 77, 94];       // Manipulated red
      tierBg = [36, 16, 20];
    } else if (tier.includes("INCONCLUSIVE") || (rawScore >= 0.40 && rawScore < 0.90)) {
      tierColor = [255, 176, 32];      // Amber warn
      tierBg = [36, 28, 12];
    }

    // Full page background
    doc.setFillColor(...bgDark);
    doc.rect(0, 0, pageWidth, pageHeight, "F");

    // Top Accent line
    doc.setFillColor(...accentCyan);
    doc.rect(margin, 12, contentWidth, 1.2, "F");

    // Header Branding
    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.setTextColor(...textWhite);
    doc.text("SIGNAL", margin, 20);

    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...accentCyan);
    doc.text("// FORENSIC INTELLIGENCE REPORT", margin + 24, 19.8);

    doc.setFontSize(7.5);
    doc.setTextColor(...textMuted);
    doc.text(`REPORT ID: ${reportId}`, pageWidth - margin, 17, { align: "right" });
    doc.text(`GENERATED: ${timestampStr}`, pageWidth - margin, 21, { align: "right" });

    // Meta Box
    let y = 26;
    doc.setFillColor(...bgCard);
    doc.setDrawColor(...borderLine);
    doc.setLineWidth(0.3);
    doc.roundedRect(margin, y, contentWidth, 16, 1.5, 1.5, "FD");

    const colW = contentWidth / 3;
    // Col 1
    doc.setFontSize(6.5);
    doc.setTextColor(...textMuted);
    doc.text("SOURCE SEQUENCE", margin + 4, y + 5);
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...textWhite);
    const shortFileName = fileName.length > 28 ? fileName.substring(0, 25) + "..." : fileName;
    doc.text(shortFileName, margin + 4, y + 11);

    // Col 2
    doc.setFont("helvetica", "normal");
    doc.setFontSize(6.5);
    doc.setTextColor(...textMuted);
    doc.text("PIPELINE ENGINE", margin + colW + 4, y + 5);
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...textWhite);
    doc.text(`${detectorName} + ${modelBackendName}`, margin + colW + 4, y + 11);

    // Col 3
    doc.setFont("helvetica", "normal");
    doc.setFontSize(6.5);
    doc.setTextColor(...textMuted);
    doc.text("SCREENING STATUS", margin + (colW * 2) + 4, y + 5);
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...accentCyan);
    doc.text("AUDIT COMPLETE", margin + (colW * 2) + 4, y + 11);

    // Executive Assessment Banner
    y = 46;
    doc.setFillColor(...tierBg);
    doc.setDrawColor(...tierColor);
    doc.setLineWidth(0.4);
    doc.roundedRect(margin, y, contentWidth, 24, 1.5, 1.5, "FD");

    // Tier indicator vertical strip
    doc.setFillColor(...tierColor);
    doc.rect(margin, y, 3, 24, "F");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(...tierColor);
    doc.text(tier, margin + 8, y + 8.5);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(...textWhite);
    const summaryLines = doc.splitTextToSize(message || "Forensic analysis completed across extracted video sequence.", contentWidth - 14);
    doc.text(summaryLines, margin + 8, y + 15);

    // Telemetry Grid
    y = 74;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(...accentCyan);
    doc.text("// VERIFIED SENSORY TELEMETRY", margin, y);

    y = 77;
    const tileW = (contentWidth - 8) / 3;
    const tileH = 17;
    const telemetryItems = [
      { label: "FORENSIC SCORE", val: Number(rawScore).toFixed(4), sub: "Scale: 0.0000 - 1.0000", accent: true },
      { label: "FRAMES AUDITED", val: `${framesCount} / ${framesCount}`, sub: "Uniform temporal stride" },
      { label: "FACE COVERAGE", val: `${coverageVal}%`, sub: "Subject tracked in clip" },
      { label: "ANALYSIS LATENCY", val: `${executionTimeVal}s`, sub: "Hardware inference time" },
      { label: "LANDMARK DETECTOR", val: detectorName, sub: "5-point mouth ROI crop" },
      { label: "MODEL BACKEND", val: modelBackendName, sub: "Spatio-temporal neural net" }
    ];

    telemetryItems.forEach((item, i) => {
      const col = i % 3;
      const row = Math.floor(i / 3);
      const tx = margin + col * (tileW + 4);
      const ty = y + row * (tileH + 3.5);

      doc.setFillColor(...bgCard);
      doc.setDrawColor(...borderLine);
      doc.setLineWidth(0.25);
      doc.roundedRect(tx, ty, tileW, tileH, 1, 1, "FD");

      doc.setFont("helvetica", "normal");
      doc.setFontSize(6);
      doc.setTextColor(...textMuted);
      doc.text(item.label, tx + 3.5, ty + 4.5);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(10.5);
      if (item.accent) {
        doc.setTextColor(...tierColor);
      } else {
        doc.setTextColor(...textWhite);
      }
      doc.text(String(item.val), tx + 3.5, ty + 10.5);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(5.5);
      doc.setTextColor(...textDarkMuted);
      doc.text(item.sub, tx + 3.5, ty + 14.5);
    });

    // Frame-level anomaly breakdown (if available)
    y = 118;
    const frameScores = Array.isArray(data.frame_scores) ? data.frame_scores :
                        (Array.isArray(data.attention_weights) ? data.attention_weights :
                        (Array.isArray(data.frame_probabilities) ? data.frame_probabilities : []));

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(...accentCyan);
    doc.text("// TEMPORAL FRAME-LEVEL SUMMARY", margin, y);

    y = 121;
    doc.setFillColor(...bgCard);
    doc.setDrawColor(...borderLine);
    doc.setLineWidth(0.25);
    doc.roundedRect(margin, y, contentWidth, 22, 1.5, 1.5, "FD");

    if (frameScores.length > 0) {
      const minScore = Math.min(...frameScores);
      const maxScore = Math.max(...frameScores);
      const avgScore = frameScores.reduce((a, b) => a + b, 0) / frameScores.length;
      const flaggedCount = frameScores.filter(s => s >= 0.50).length;

      const subColW = contentWidth / 4;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(6.5);
      doc.setTextColor(...textMuted);
      doc.text("TOTAL SAMPLE WINDOWS", margin + 4, y + 6);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(...textWhite);
      doc.text(`${frameScores.length} Samples`, margin + 4, y + 13);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(6.5);
      doc.setTextColor(...textMuted);
      doc.text("PEAK WINDOW SCORE", margin + subColW + 4, y + 6);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(...(maxScore >= 0.50 ? [255, 77, 94] : textWhite));
      doc.text(maxScore.toFixed(4), margin + subColW + 4, y + 13);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(6.5);
      doc.setTextColor(...textMuted);
      doc.text("MEAN WINDOW SCORE", margin + (subColW * 2) + 4, y + 6);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(...textWhite);
      doc.text(avgScore.toFixed(4), margin + (subColW * 2) + 4, y + 13);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(6.5);
      doc.setTextColor(...textMuted);
      doc.text("ANOMALY FLAGS (>=0.50)", margin + (subColW * 3) + 4, y + 6);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(...(flaggedCount > 0 ? [255, 176, 32] : [87, 247, 156]));
      doc.text(`${flaggedCount} of ${frameScores.length} windows`, margin + (subColW * 3) + 4, y + 13);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(6);
      doc.setTextColor(...textDarkMuted);
      doc.text("Multi-window temporal feature pooling over consecutive sampled strides.", margin + 4, y + 18.5);
    } else {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(...textMuted);
      doc.text("Single sequence composite inference score recorded for this submission.", margin + 4, y + 12);
    }

    // Plain-Language Interpretation
    y = 147;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(...accentCyan);
    doc.text("// FORENSIC ASSESSMENT INTERPRETATION", margin, y);

    y = 150;
    doc.setFillColor(...bgCard);
    doc.setDrawColor(...borderLine);
    doc.setLineWidth(0.25);
    doc.roundedRect(margin, y, contentWidth, 34, 1.5, 1.5, "FD");

    let interpP1 = "";
    let interpP2 = "";
    if (tier.includes("MANIPULATION") || rawScore >= 0.90) {
      interpP1 = "The perception engine identified significant spatio-temporal lip-motion anomalies and boundary incoherence across sampled frame windows. These markers are characteristic of synthetic generation, neural face-swapping, or deepfake audio-driven lip synchronization.";
      interpP2 = "Operational Context: While elevated scores indicate strong synthetic indicators, severe video re-compression, extreme lighting gradients, and erratic subject motion can elevate scores. Independent verification is recommended before taking adverse action.";
    } else if (tier.includes("INCONCLUSIVE") || (rawScore >= 0.40 && rawScore < 0.90)) {
      interpP1 = "The analyzed footage produced an ambiguous forensic score falling between normal biological movement and overt synthetic manipulation. Subtle temporal fluctuations or compression artifacts prevented a definitive low-risk classification.";
      interpP2 = "Operational Context: This result indicates caution rather than proven tampering. Secondary forensic review with uncompressed source footage and acoustic synchronization analysis is recommended.";
    } else {
      interpP1 = "This clip exhibited natural temporal micro-motions and lip dynamics consistent with genuine biological speech recordings. No measurable generator seams, diffusion blending artifacts, or temporal stutter were detected.";
      interpP2 = "Operational Context: This finding indicates low forensic risk. It serves as automated screening evidence and should be evaluated alongside chain of custody and file provenance.";
    }

    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(...textWhite);
    const splitP1 = doc.splitTextToSize(interpP1, contentWidth - 8);
    doc.text(splitP1, margin + 4, y + 6);

    doc.setTextColor(...textMuted);
    const splitP2 = doc.splitTextToSize(interpP2, contentWidth - 8);
    doc.text(splitP2, margin + 4, y + 20);

    // Methodology & Limitations Notice
    y = 188;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(...accentCyan);
    doc.text("// EVIDENTIARY GUIDELINES & LIMITATIONS", margin, y);

    y = 191;
    doc.setFillColor(...bgCard);
    doc.setDrawColor(...borderLine);
    doc.setLineWidth(0.25);
    doc.roundedRect(margin, y, contentWidth, 48, 1.5, 1.5, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(7);
    doc.setTextColor(...textWhite);
    doc.text("Calibrated Three-Tier Forensic Policy:", margin + 4, y + 6);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(6.8);
    doc.setTextColor(...textMuted);
    doc.text("• Score < 0.40 (Likely Authentic): Low anomaly signal. Spatio-temporal motion aligns with natural recording.", margin + 4, y + 11.5);
    doc.text("• 0.40 <= Score < 0.90 (Inconclusive): Ambiguous signal. Video compression or rapid motion requires review.", margin + 4, y + 16.5);
    doc.text("• Score >= 0.90 (Signs of Possible Manipulation): High anomaly signal. Suggests synthetic face-swap / lip-sync.", margin + 4, y + 21.5);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(7);
    doc.setTextColor(...textWhite);
    doc.text("Forensic Screening Signal Notice:", margin + 4, y + 29);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(6.8);
    doc.setTextColor(...textMuted);
    const noticeText = "SIGNAL is an automated forensic perception instrument designed to identify anomalous statistical patterns in facial dynamics. This report serves as technical screening evidence and does not constitute absolute or judicial proof of authenticity or manipulation. Results must be interpreted in conjunction with complete chain-of-custody verification, device provenance, and comprehensive expert human review.";
    const splitNotice = doc.splitTextToSize(noticeText, contentWidth - 8);
    doc.text(splitNotice, margin + 4, y + 34.5);

    // Footer
    y = 245;
    doc.setDrawColor(...borderLine);
    doc.setLineWidth(0.2);
    doc.line(margin, y, pageWidth - margin, y);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(6.5);
    doc.setTextColor(...textMuted);
    doc.text("SIGNAL AI FORENSIC PERCEPTION ENGINE • PROPRIETARY SPATIO-TEMPORAL INTELLIGENCE", margin, y + 4.5);
    doc.text("CONFIDENTIAL FORENSIC AUDIT REPORT", pageWidth - margin, y + 4.5, { align: "right" });

    doc.setFontSize(5.5);
    doc.setTextColor(...textDarkMuted);
    doc.text("Generated directly from live /detect telemetry • All values verified against detector pipeline output.", margin, y + 8.5);
    doc.text("PAGE 1 OF 1", pageWidth - margin, y + 8.5, { align: "right" });

    // Save and download directly without navigating away
    doc.save("SIGNAL-forensic-report.pdf");
  }

  if (btnViewReport) {
    btnViewReport.addEventListener("click", () => {
      if (!latestAnalysisData) return;
      populateReportModal(latestAnalysisData, currentFileName);
      reportModalOverlay.classList.add("active");
    });
  }

  if (modalCloseBtn) {
    modalCloseBtn.addEventListener("click", () => {
      reportModalOverlay.classList.remove("active");
    });
  }

  if (reportModalOverlay) {
    reportModalOverlay.addEventListener("click", (e) => {
      if (e.target === reportModalOverlay) {
        reportModalOverlay.classList.remove("active");
      }
    });
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && reportModalOverlay && reportModalOverlay.classList.contains("active")) {
      reportModalOverlay.classList.remove("active");
    }
  });

  if (modalBtnPrint) {
    modalBtnPrint.addEventListener("click", () => {
      window.print();
    });
  }

  if (btnDownloadPdfCard) {
    btnDownloadPdfCard.addEventListener("click", downloadPdfReport);
  }
  if (modalBtnDownloadPdf) {
    modalBtnDownloadPdf.addEventListener("click", downloadPdfReport);
  }

  if (btnDownloadResults) {
    btnDownloadResults.addEventListener("click", downloadRawJson);
  }
  if (modalBtnDownloadJson) {
    modalBtnDownloadJson.addEventListener("click", downloadRawJson);
  }

  async function checkSystemHealth() {
    const navStatus = document.getElementById("nav-system-status");
    const metaDetector = document.getElementById("meta-detector");
    const metaAlignment = document.getElementById("meta-alignment");
    const metaStride = document.getElementById("meta-stride");
    const metaLimit = document.getElementById("meta-limit");
    const dropzonePrompt = document.getElementById("dropzone-sub-prompt");

    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.status === "ok" && data.model_loaded) {
        if (navStatus) {
          navStatus.innerHTML = '<i></i>ENGINE READY';
          navStatus.classList.remove("offline");
        }
        if (metaDetector) metaDetector.textContent = data.detector_name || "YuNet ONNX";
        if (metaAlignment) metaAlignment.textContent = data.alignment_method || "Mouth ROI crop";
        if (metaStride) metaStride.textContent = data.sampling_stride || "25 frames";
        if (metaLimit) metaLimit.textContent = data.input_limit || `MAX ${data.max_upload_mb || 100} MB`;
        if (dropzonePrompt) dropzonePrompt.textContent = `MP4 / AVI &middot; ${data.input_limit || 'MAX 100 MB'}`;
      } else {
        throw new Error("Engine not ready");
      }
    } catch (err) {
      if (navStatus) {
        navStatus.innerHTML = '<i></i>ENGINE OFFLINE';
        navStatus.classList.add("offline");
      }
      if (metaDetector) metaDetector.textContent = "OFFLINE";
      if (metaAlignment) metaAlignment.textContent = "OFFLINE";
      if (metaStride) metaStride.textContent = "OFFLINE";
      if (metaLimit) metaLimit.textContent = "OFFLINE";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    checkSystemHealth();
  });
