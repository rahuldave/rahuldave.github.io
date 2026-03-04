(function () {
  "use strict";

  // Only run on post pages
  var match = window.location.pathname.match(/^\/posts\/([^/]+)\/?/);
  if (!match) return;

  var params = new URLSearchParams(window.location.search);
  var pathId = params.get("path");
  if (!pathId) return;

  var stepNum = parseInt(params.get("step"), 10) || 0;

  fetch("/assets/learning-paths.json")
    .then(function (res) { return res.ok ? res.json() : null; })
    .catch(function () { return null; })
    .then(function (manifest) {
      if (!manifest || !manifest[pathId]) return;

      var path = manifest[pathId];
      var steps = path.steps;
      var parts = path.parts;
      var total = steps.length;

      // Find current step (by step param or by slug match)
      var currentSlug = match[1];
      var idx = stepNum > 0 ? stepNum - 1 : -1;
      if (idx < 0 || idx >= total || steps[idx].slug !== currentSlug) {
        for (var i = 0; i < total; i++) {
          if (steps[i].slug === currentSlug) {
            idx = i;
            break;
          }
        }
      }
      if (idx < 0) return;

      // Find which part this step belongs to
      var partInfo = findPart(parts, idx);
      var pathUrl = "/learning-paths/" + pathId + ".html";

      // ── Top banner ──
      var banner = document.createElement("div");
      banner.className = "lp-banner";

      var label = document.createElement("span");
      label.className = "lp-banner-label";
      var labelText = "Part " + partInfo.partNum + ": " + escapeHtml(partInfo.partTitle);
      labelText += " \u2014 Step " + (partInfo.stepInPart) + " of " + partInfo.partSize;
      labelText += " in <a href=\"" + pathUrl + "\">" + escapeHtml(path.title) + "</a>";
      label.innerHTML = labelText;
      banner.appendChild(label);

      var nav = document.createElement("span");
      nav.className = "lp-banner-nav";
      if (idx > 0) {
        nav.innerHTML += "<a href=\"" + stepUrl(steps[idx - 1], pathId, idx) + "\">\u2190 Prev</a>";
      }
      if (idx < total - 1) {
        nav.innerHTML += "<a href=\"" + stepUrl(steps[idx + 1], pathId, idx + 2) + "\">Next \u2192</a>";
      }
      banner.appendChild(nav);

      // Insert after .quarto-title-block
      var titleBlock = document.querySelector(".quarto-title-block");
      if (titleBlock && titleBlock.parentNode) {
        titleBlock.parentNode.insertBefore(banner, titleBlock.nextSibling);
      }

      // ── Bottom nav ──
      var bottomNav = document.createElement("div");
      bottomNav.className = "lp-nav";

      if (idx > 0) {
        var prev = steps[idx - 1];
        var prevPart = findPart(parts, idx - 1);
        bottomNav.innerHTML += '<a class="lp-nav-card lp-nav-prev" href="' + stepUrl(prev, pathId, idx) + '">' +
          '<span class="lp-nav-direction">\u2190 Previous</span>' +
          '<span class="lp-nav-part">Part ' + prevPart.partNum + ': ' + escapeHtml(prevPart.partTitle) + '</span>' +
          '<span class="lp-nav-title">' + escapeHtml(prev.title) + '</span></a>';
      }

      if (idx < total - 1) {
        var next = steps[idx + 1];
        var nextPart = findPart(parts, idx + 1);
        bottomNav.innerHTML += '<a class="lp-nav-card lp-nav-next" href="' + stepUrl(next, pathId, idx + 2) + '">' +
          '<span class="lp-nav-direction">Next \u2192</span>' +
          '<span class="lp-nav-part">Part ' + nextPart.partNum + ': ' + escapeHtml(nextPart.partTitle) + '</span>' +
          '<span class="lp-nav-title">' + escapeHtml(next.title) + '</span></a>';
      }

      // Insert before #discuss-links or at end of article
      var discuss = document.getElementById("discuss-links");
      var article = document.querySelector("#quarto-document-content") || document.querySelector("main.content") || document.querySelector("main");
      if (discuss && discuss.parentNode) {
        discuss.parentNode.insertBefore(bottomNav, discuss);
      } else if (article) {
        article.appendChild(bottomNav);
      }
    });

  function findPart(parts, globalIdx) {
    var offset = 0;
    for (var p = 0; p < parts.length; p++) {
      var partSteps = parts[p].steps.length;
      if (globalIdx < offset + partSteps) {
        return {
          partNum: p + 1,
          partTitle: parts[p].title,
          partSize: partSteps,
          stepInPart: globalIdx - offset + 1
        };
      }
      offset += partSteps;
    }
    // Fallback
    return { partNum: 1, partTitle: "", partSize: 1, stepInPart: 1 };
  }

  function stepUrl(step, pathId, stepNum) {
    return step.url + "?path=" + encodeURIComponent(pathId) + "&step=" + stepNum;
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }
})();
