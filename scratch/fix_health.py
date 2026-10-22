import os

html_path = os.path.join("static", "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """  if (modalBtnDownloadJson) {
    modalBtnDownloadJson.addEventListener("click", downloadRawJson);
  }


    const metaAlignment = document.getElementById("meta-alignment");"""

replacement = """  if (modalBtnDownloadJson) {
    modalBtnDownloadJson.addEventListener("click", downloadRawJson);
  }

  async function checkSystemHealth() {
    const navStatus = document.getElementById("nav-system-status");
    const metaDetector = document.getElementById("meta-detector");
    const metaAlignment = document.getElementById("meta-alignment");"""

if target in content:
    content = content.replace(target, replacement, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("TARGET NOT FOUND")
