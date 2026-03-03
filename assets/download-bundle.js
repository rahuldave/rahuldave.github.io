/**
 * Download Bundle Button — injects a "Download & Run" button on notebook posts.
 *
 * Gates on /posts/<slug>/ URLs, HEAD-checks for <slug>.zip,
 * and injects a download button near the "Summarize this article" button.
 *
 * Uses the same IIFE pattern as llm-explain.js (runs after DOM is ready
 * since it's loaded via include-after-body).
 */
(function () {
  "use strict";

  // Only run on post pages: /posts/<slug>/ or /posts/<slug>/index.html
  const match = window.location.pathname.match(/^\/posts\/([^/]+)\/?/);
  if (!match) return;

  const slug = match[1];
  const zipUrl = `/posts/${slug}/${slug}.zip`;

  // HEAD-check if the zip exists
  fetch(zipUrl, { method: "HEAD" })
    .then(function (res) {
      if (!res.ok) return;

      const sizeBytes = parseInt(res.headers.get("content-length") || "0", 10);
      const sizeStr = formatSize(sizeBytes);

      injectButton(zipUrl, sizeStr);
    })
    .catch(function () {
      // zip doesn't exist, do nothing
    });

  function formatSize(bytes) {
    if (!bytes || bytes <= 0) return "";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  function injectButton(url, sizeStr) {
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
})();
