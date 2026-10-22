import os

html_path = os.path.join("static", "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """  footer{position:relative; z-index:2; border-top:1px solid var(--line); padding:36px 48px; display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:12px;}


</head>"""

replacement = """  footer{position:relative; z-index:2; border-top:1px solid var(--line); padding:36px 48px; display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:12px;}

  @media (max-width:980px){
    .hero-grid{grid-template-columns:1fr;}
    .panel-col{margin-top:30px; margin-left:auto; margin-right:auto; max-width:520px; width:100%;}
    .panel{transform:none;}
    .stats-band{grid-template-columns:repeat(2,1fr);}
    .bento{grid-template-columns:1fr;}
    .method-card.lg{grid-row:auto;}
    nav{padding:18px 24px;}
    .navlinks{display:none;}
    .track{flex-wrap:wrap; gap:28px;}
    .node{width:45%;}
    .track-line{display:none;}
    .archive-panel{grid-template-columns:1fr;}
  }

  /* ---------------- print styles ---------------- */
  @media print {
    body, html {
      background: #ffffff !important;
      color: #000000 !important;
      cursor: default !important;
    }
    nav, .hero, .ticker, #stats, #workflow, #methodology, #architecture, #archive, #results-section, footer, .cdot, .cring, #noise, .scanlines, .blob, .vignette, .modal-close-btn, .report-modal-footer {
      display: none !important;
    }
    .report-modal-overlay {
      position: static !important;
      background: none !important;
      padding: 0 !important;
      display: block !important;
      opacity: 1 !important;
      pointer-events: auto !important;
    }
    .report-modal {
      box-shadow: none !important;
      border: 1px solid #ccc !important;
      max-height: none !important;
      max-width: 100% !important;
      background: #ffffff !important;
      color: #000000 !important;
      transform: none !important;
    }
    .report-modal-header {
      background: #f4f6f8 !important;
      color: #000000 !important;
      border-bottom: 2px solid #333 !important;
    }
    .report-modal-brand b {
      color: #000000 !important;
    }
    .report-meta-grid, .report-metric-card, .report-explanation-box, .report-guidance-box {
      background: #f8fafb !important;
      border: 1px solid #ddd !important;
      color: #111 !important;
    }
    .report-meta-value, .report-metric-value, #modal-verdict-summary, #modal-explanation-p1, #modal-explanation-p2 {
      color: #000 !important;
    }
    .report-verdict-banner.real {
      background: #e8fbf0 !important;
      border-left: 6px solid #1a8f4c !important;
    }
    .report-verdict-banner.real .report-banner-tier {
      color: #1a8f4c !important;
    }
    .report-verdict-banner.inconclusive {
      background: #fff8eb !important;
      border-left: 6px solid #c27d00 !important;
    }
    .report-verdict-banner.inconclusive .report-banner-tier {
      color: #c27d00 !important;
    }
    .report-verdict-banner.fake {
      background: #fdebee !important;
      border-left: 6px solid #d11a2a !important;
    }
    .report-verdict-banner.fake .report-banner-tier {
      color: #d11a2a !important;
    }
  }
</style>
</head>"""

if target in content:
    content = content.replace(target, replacement, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Style tag closed and media queries restored.")
else:
    print("ERROR: Target string not found.")
