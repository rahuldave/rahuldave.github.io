/**
 * Download Bundle Button — injects a "Download & Run" button on notebook posts.
 * Run in Browser Button — injects a "Run in Browser" button for Pyodide-compatible posts.
 *
 * Gates on /posts/<slug>/ URLs, HEAD-checks for <slug>.zip,
 * and injects download/run buttons near the "Summarize this article" button.
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

  // HEAD-check if the zip exists
  fetch(zipUrl, { method: "HEAD" })
    .then(function (res) {
      if (!res.ok) return;

      var sizeBytes = parseInt(res.headers.get("content-length") || "0", 10);
      var sizeStr = formatSize(sizeBytes);

      injectDownloadButton(zipUrl, sizeStr);
    })
    .catch(function () {
      // zip doesn't exist, do nothing
    });

  // Check bundles.json for Pyodide compatibility and inject "Run in Browser" button
  fetch("/bundles.json")
    .then(function (res) {
      if (!res.ok) return;
      return res.json();
    })
    .then(function (manifest) {
      if (!manifest) return;
      var info = manifest[slug];
      if (info && info.pyodide_compatible) {
        injectRunInBrowserButton(info.zip);
      }
    })
    .catch(function () {
      // bundles.json doesn't exist or parse error, do nothing
    });

  function formatSize(bytes) {
    if (!bytes || bytes <= 0) return "";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  function injectDownloadButton(url, sizeStr) {
    // Find the summarize button wrapper to place our button next to it
    var summarizeWrap = document.querySelector(".llm-summarize-wrap");

    // Create our button
    var btn = document.createElement("a");
    btn.href = url;
    btn.download = "";
    btn.className = "download-bundle-btn";
    btn.innerHTML =
      '<span class="download-bundle-icon">\u2913</span>' +
      '<span class="download-bundle-label">Download and Run</span>' +
      (sizeStr
        ? '<span class="download-bundle-size">' + sizeStr + "</span>"
        : "");

    // Tooltip with run instructions
    btn.title = "Download zip bundle. Run with: uvx juv run index.ipynb";

    if (summarizeWrap) {
      // Insert into the same wrapper as the summarize button (same line, flush right)
      summarizeWrap.insertBefore(btn, summarizeWrap.firstChild);
    } else {
      // Fallback: create own wrapper at top of article content
      var wrap = document.createElement("div");
      wrap.className = "download-bundle-wrap";
      wrap.appendChild(btn);

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

  function injectRunInBrowserButton(zipPath) {
    var summarizeWrap = document.querySelector(".llm-summarize-wrap");
    if (!summarizeWrap) return;

    var btn = document.createElement("a");
    btn.href = "/lab/loader.html?zip=" + zipPath;
    btn.className = "download-bundle-btn run-in-browser-btn";
    btn.innerHTML =
      '<span class="download-bundle-icon">\u25B6</span>' +
      '<span class="download-bundle-label">Run in Browser</span>';
    btn.title = "Open in JupyterLite (runs in your browser, no install needed)";

    // Insert as first child so it appears leftmost
    summarizeWrap.insertBefore(btn, summarizeWrap.firstChild);
  }
})();
