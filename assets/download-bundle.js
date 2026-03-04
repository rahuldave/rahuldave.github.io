/**
 * Download Bundle + Run in Browser Buttons
 *
 * Gates on /posts/<slug>/ URLs. Fetches zip HEAD + bundles.json in parallel,
 * then injects buttons in deterministic order:
 *   [Run in Browser] [Download and Run] ... [Download md] [Summarize]
 *
 * Uses the same IIFE pattern as llm-explain.js (runs after DOM is ready
 * since it's loaded via include-after-body).
 */
(function () {
  "use strict";

  // Only run on post pages: /posts/<slug>/ or /posts/<slug>/index.html
  var match = window.location.pathname.match(/^\/posts\/([^/]+)\/?/);
  if (!match) return;

  var slug = match[1];
  var zipUrl = "/posts/" + slug + "/" + slug + ".zip";

  // Fetch both in parallel, then inject buttons in order
  var zipCheck = fetch(zipUrl, { method: "HEAD" })
    .then(function (res) {
      if (!res.ok) return null;
      return parseInt(res.headers.get("content-length") || "0", 10);
    })
    .catch(function () { return null; });

  var manifestCheck = fetch("/bundles.json")
    .then(function (res) {
      if (!res.ok) return null;
      return res.json();
    })
    .catch(function () { return null; });

  Promise.all([zipCheck, manifestCheck]).then(function (results) {
    var sizeBytes = results[0];
    var manifest = results[1];

    var summarizeWrap = document.querySelector(".llm-summarize-wrap");

    // "Run in Browser" button (leftmost) — only for Pyodide-compatible posts
    if (manifest && manifest[slug] && manifest[slug].pyodide_compatible && summarizeWrap) {
      var runBtn = document.createElement("a");
      runBtn.href = "/lab/loader.html?zip=" + manifest[slug].zip;
      runBtn.className = "download-bundle-btn run-in-browser-btn";
      runBtn.innerHTML =
        '<span class="download-bundle-icon">\u25B6</span>' +
        '<span class="download-bundle-label">Run in Browser</span>';
      runBtn.title = "Open in JupyterLite (runs in your browser, no install needed)";
      runBtn.target = "_blank";
      summarizeWrap.insertBefore(runBtn, summarizeWrap.firstChild);
    }

    // "Download and Run" button (second from left)
    if (sizeBytes !== null) {
      var sizeStr = formatSize(sizeBytes);
      var dlBtn = document.createElement("a");
      dlBtn.href = zipUrl;
      dlBtn.download = "";
      dlBtn.className = "download-bundle-btn";
      dlBtn.innerHTML =
        '<span class="download-bundle-icon">\u2913</span>' +
        '<span class="download-bundle-label">Download and Run</span>' +
        (sizeStr
          ? '<span class="download-bundle-size">' + sizeStr + "</span>"
          : "");
      dlBtn.title = "Download zip bundle. Run with: uvx juv run index.ipynb";

      if (summarizeWrap) {
        // Insert after Run in Browser (if present) but before Download md / Summarize
        var runBtn = summarizeWrap.querySelector(".run-in-browser-btn");
        if (runBtn) {
          summarizeWrap.insertBefore(dlBtn, runBtn.nextSibling);
        } else {
          summarizeWrap.insertBefore(dlBtn, summarizeWrap.firstChild);
        }
      } else {
        var wrap = document.createElement("div");
        wrap.className = "download-bundle-wrap";
        wrap.appendChild(dlBtn);

        var article =
          document.querySelector("#quarto-document-content") ||
          document.querySelector("main.content") ||
          document.querySelector("main");
        if (article) {
          var titleBlock = article.querySelector(".quarto-title-block");
          if (titleBlock) {
            titleBlock.parentNode.insertBefore(wrap, titleBlock.nextSibling);
          } else {
            article.insertBefore(wrap, article.firstChild);
          }
        }
      }
    }
  });

  function formatSize(bytes) {
    if (!bytes || bytes <= 0) return "";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }
})();
