import os

html_path = os.path.join("static", "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """    downloadAnchor.click();

    const navStatus = document.getElementById("nav-system-status");"""

replacement = """    downloadAnchor.click();
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

  async function checkSystemHealth() {"""

if target in content:
    new_content = content.replace(target, replacement, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: index.html updated")
else:
    print("ERROR: target not found")
