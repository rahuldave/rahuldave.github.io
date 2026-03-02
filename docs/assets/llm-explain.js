(function() {
  // Gate: only activate on individual post pages
  var p = window.location.pathname;
  if (!p.startsWith('/posts/')) return;
  var listingPaths = ['/posts/', '/posts', '/posts/index.html'];
  if (listingPaths.indexOf(p) >= 0) return;

  // Extract post slug from URL
  var match = p.match(/\/posts\/([^/.]+)/);
  if (!match) return;
  var slug = match[1];
  var baseUrl = '/posts/' + slug + '/';

  // State
  var cellsData = null;
  var contentMd = null;
  var apiKey = localStorage.getItem('claude-api-key');
  var MODEL = 'claude-sonnet-4-20250514';
  var SYSTEM_PROMPT = 'You are a helpful teaching assistant for a data science and machine learning course. ' +
    'Explain concepts clearly and concisely. Reference specific code examples when relevant. ' +
    'Use LaTeX notation (with $...$ for inline and $$...$$ for display) for mathematical expressions. ' +
    'Keep responses focused and educational.';

  // --- Initialization ---

  fetch(baseUrl + 'cells.json')
    .then(function(r) { return r.ok ? r.json() : null; })
    .then(function(data) {
      if (!data) return;
      cellsData = data;
      injectButtons();
    })
    .catch(function() { /* No cells.json — feature disabled for this post */ });

  // --- Button Injection ---

  function injectButtons() {
    injectSummarizeButton();
    injectCellButtons();
  }

  function injectSummarizeButton() {
    var anchor = document.querySelector('#title-block-header') ||
                 document.querySelector('.quarto-title-block') ||
                 document.querySelector('header');
    if (!anchor) return;

    var wrap = document.createElement('div');
    wrap.className = 'llm-summarize-wrap';

    var btn = document.createElement('button');
    btn.className = 'llm-btn llm-btn-summarize';
    btn.textContent = 'Summarize this page';
    btn.addEventListener('click', function() { handleAction('summarize', null); });

    wrap.appendChild(btn);
    anchor.parentNode.insertBefore(wrap, anchor.nextSibling);
  }

  function injectCellButtons() {
    var sourceType = cellsData ? cellsData.source_type : 'ipynb';

    // --- Notebook posts (ipynb): buttons after Quarto's div.cell elements ---
    var notebookCells = document.querySelectorAll('div.cell[id^="cell-"]');
    for (var i = 0; i < notebookCells.length; i++) {
      var cell = notebookCells[i];
      var cellIndex = parseInt(cell.id.replace('cell-', ''), 10);
      if (isNaN(cellIndex)) continue;

      var container = document.createElement('div');
      container.className = 'llm-buttons';

      container.appendChild(makeLlmBtn('Explain up to here', 'explain-upto', cellIndex));
      container.appendChild(makeLlmBtn('Explain this code', 'explain-code', cellIndex));

      cell.parentNode.insertBefore(container, cell.nextSibling);
    }

    // --- Non-notebook posts (md/qmd): buttons after headings and code blocks ---
    if (sourceType !== 'ipynb' || notebookCells.length === 0) {
      // Build a lookup: heading slug -> cell index from cells.json
      var slugToCellIndex = {};
      if (cellsData && cellsData.cells) {
        for (var c = 0; c < cellsData.cells.length; c++) {
          var meta = cellsData.cells[c];
          if (meta.headings) {
            for (var h = 0; h < meta.headings.length; h++) {
              slugToCellIndex[meta.headings[h].slug] = meta.cell;
            }
          }
        }
      }

      // "Explain up to here" after h2/h3 headings
      // Quarto puts the id on the parent <section>, and data-anchor-id on the <h>
      var headings = document.querySelectorAll('h2[data-anchor-id], h3[data-anchor-id], h2[id], h3[id]');
      var seen = new Set();
      for (var i = 0; i < headings.length; i++) {
        var heading = headings[i];
        var headingSlug = heading.getAttribute('data-anchor-id') || heading.id;
        if (!headingSlug || headingSlug === 'toc-title') continue;
        if (seen.has(headingSlug)) continue; // deduplicate
        seen.add(headingSlug);

        var cellIdx = slugToCellIndex[headingSlug];
        if (cellIdx === undefined) cellIdx = i; // fallback to DOM order

        var wrap = document.createElement('div');
        wrap.className = 'llm-buttons';
        wrap.appendChild(makeLlmBtn('Explain up to here', 'explain-upto', cellIdx));

        // Insert after the heading's parent section, or after the heading itself
        var section = heading.closest('section');
        var target = section || heading;
        if (target.nextSibling) {
          target.parentNode.insertBefore(wrap, target.nextSibling);
        } else {
          target.parentNode.appendChild(wrap);
        }
      }

      // "Explain this code" after standalone code blocks (not inside div.cell)
      var codeBlocks = document.querySelectorAll('div.sourceCode');
      for (var i = 0; i < codeBlocks.length; i++) {
        var block = codeBlocks[i];
        // Skip code blocks that are inside Quarto notebook cells
        if (block.closest('div.cell')) continue;

        // Find the matching cell index from cells.json
        // Use position in DOM relative to headings to estimate cell index
        var codeIdx = findCodeCellIndex(block, slugToCellIndex);

        var wrap = document.createElement('div');
        wrap.className = 'llm-buttons';
        wrap.appendChild(makeLlmBtn('Explain up to here', 'explain-upto', codeIdx));
        wrap.appendChild(makeLlmBtn('Explain this code', 'explain-code', codeIdx));

        block.parentNode.insertBefore(wrap, block.nextSibling);
      }
    }
  }

  function findCodeCellIndex(codeBlock, slugToCellIndex) {
    // Walk backwards from the code block to find the nearest heading
    // then look up its cell index and add 1 (the code cell follows the heading)
    if (!cellsData || !cellsData.cells) return 0;

    // Find code cells in cells.json
    var codeCells = [];
    for (var i = 0; i < cellsData.cells.length; i++) {
      if (cellsData.cells[i].type === 'code') {
        codeCells.push(cellsData.cells[i].cell);
      }
    }

    // Find which code block this is in DOM order (among standalone code blocks)
    var allCodeBlocks = document.querySelectorAll('div.sourceCode:not(.cell-code)');
    for (var i = 0; i < allCodeBlocks.length; i++) {
      if (allCodeBlocks[i] === codeBlock && i < codeCells.length) {
        return codeCells[i];
      }
    }
    return codeCells.length > 0 ? codeCells[0] : 0;
  }

  function makeLlmBtn(text, action, cellIndex) {
    var btn = document.createElement('button');
    btn.className = 'llm-btn';
    btn.textContent = text;
    btn.setAttribute('data-action', action);
    btn.setAttribute('data-cell', cellIndex);
    btn.addEventListener('click', makeHandler(action, cellIndex));
    return btn;
  }

  function makeHandler(action, cellIndex) {
    return function() { handleAction(action, cellIndex); };
  }

  // --- Action Handler ---

  function handleAction(action, cellIndex) {
    apiKey = localStorage.getItem('claude-api-key');
    if (!apiKey) {
      showApiKeyModal(function() { handleAction(action, cellIndex); });
      return;
    }

    if (contentMd) {
      executeAction(action, cellIndex);
    } else {
      fetch(baseUrl + 'content.md')
        .then(function(r) {
          if (!r.ok) throw new Error('Could not load content');
          return r.text();
        })
        .then(function(text) {
          contentMd = text;
          executeAction(action, cellIndex);
        })
        .catch(function(err) {
          showChatModal();
          var el = document.querySelector('.llm-chat-response');
          if (el) {
            el.classList.remove('llm-loading');
            el.innerHTML = '<p class="llm-error">Error loading content: ' + escapeHtml(err.message) + '</p>';
          }
        });
    }
  }

  function executeAction(action, cellIndex) {
    var context = sliceContent(action, cellIndex);
    var userPrompt = buildUserPrompt(action, context);
    showChatModal();
    streamResponse(userPrompt);
  }

  // --- Content Slicing ---

  function sliceContent(action, cellIndex) {
    if (action === 'summarize') {
      return contentMd;
    }

    var lines = contentMd.split('\n');

    if (action === 'explain-upto') {
      // Include everything from start through the target cell
      var result = [];
      var foundTarget = false;
      for (var i = 0; i < lines.length; i++) {
        var m = lines[i].match(/<!-- cell:(\d+)/);
        if (m) {
          var idx = parseInt(m[1], 10);
          if (idx > cellIndex && foundTarget) break;
          if (idx === cellIndex) foundTarget = true;
        }
        result.push(lines[i]);
      }
      return result.join('\n');
    }

    if (action === 'explain-code') {
      // Extract the specific code cell + preceding context
      var cellStart = -1;
      var cellEnd = lines.length;

      for (var i = 0; i < lines.length; i++) {
        var m = lines[i].match(/<!-- cell:(\d+)/);
        if (m) {
          var idx = parseInt(m[1], 10);
          if (idx === cellIndex) cellStart = i;
          else if (idx > cellIndex && cellStart >= 0) {
            cellEnd = i;
            break;
          }
        }
      }

      if (cellStart < 0) return contentMd; // fallback

      // Get preceding context (limited to ~3000 chars for focus)
      var preceding = lines.slice(0, cellStart).join('\n');
      if (preceding.length > 3000) {
        preceding = '...\n' + preceding.slice(-3000);
      }

      var codeSection = lines.slice(cellStart, cellEnd).join('\n');
      return preceding + '\n\n' + codeSection;
    }

    return contentMd;
  }

  // --- Prompt Building ---

  function buildUserPrompt(action, context) {
    var title = (cellsData && cellsData.title) ? cellsData.title : slug;

    if (action === 'summarize') {
      return 'Summarize the key concepts from this lecture note titled "' + title +
        '" in 3-5 bullet points. Be concise but thorough.\n\n' + context;
    }

    if (action === 'explain-upto') {
      return 'The student has read up to this point in the lecture note "' + title +
        '". Explain the key concepts covered so far in clear, accessible language. ' +
        'Reference specific code examples where relevant.\n\n' + context;
    }

    if (action === 'explain-code') {
      return 'Explain what this code does and why, in the context of the surrounding lecture material. ' +
        'Be specific about what each part does and the reasoning behind it.\n\n' + context;
    }

    return context;
  }

  // --- API Key Modal ---

  function showApiKeyModal(callback) {
    var overlay = document.createElement('div');
    overlay.className = 'llm-modal-overlay';

    var modal = document.createElement('div');
    modal.className = 'llm-modal llm-key-modal';
    modal.innerHTML =
      '<h3>Enter your Claude API key</h3>' +
      '<p>Your key stays in your browser (localStorage) and is sent only to api.anthropic.com. ' +
      'It is never sent to this website\'s server.</p>' +
      '<input type="password" id="llm-api-key-input" placeholder="sk-ant-..." autocomplete="off">' +
      '<div class="llm-modal-actions">' +
        '<button class="llm-btn-cancel">Cancel</button>' +
        '<button class="llm-btn-save">Save &amp; Continue</button>' +
      '</div>';

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    var input = document.getElementById('llm-api-key-input');
    setTimeout(function() { input.focus(); }, 50);

    modal.querySelector('.llm-btn-cancel').addEventListener('click', function() {
      overlay.remove();
    });

    modal.querySelector('.llm-btn-save').addEventListener('click', function() {
      var key = input.value.trim();
      if (key) {
        localStorage.setItem('claude-api-key', key);
        apiKey = key;
        overlay.remove();
        if (callback) callback();
      }
    });

    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') modal.querySelector('.llm-btn-save').click();
    });

    overlay.addEventListener('click', function(e) {
      if (e.target === overlay) overlay.remove();
    });
  }

  // --- Chat Modal ---

  function showChatModal() {
    var existing = document.querySelector('.llm-chat-overlay');
    if (existing) existing.remove();

    var overlay = document.createElement('div');
    overlay.className = 'llm-modal-overlay llm-chat-overlay';

    var modal = document.createElement('div');
    modal.className = 'llm-modal llm-chat-modal';
    modal.innerHTML =
      '<div class="llm-chat-header">' +
        '<span>Claude Explanation</span>' +
        '<div class="llm-chat-actions">' +
          '<button class="llm-btn-icon llm-btn-settings" title="API key settings">\u2699</button>' +
          '<button class="llm-btn-icon llm-btn-copy" title="Copy response">\u2398</button>' +
          '<button class="llm-btn-icon llm-btn-close" title="Close">\u2715</button>' +
        '</div>' +
      '</div>' +
      '<div class="llm-chat-body">' +
        '<div class="llm-chat-response llm-loading">Thinking\u2026</div>' +
      '</div>';

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    modal.querySelector('.llm-btn-close').addEventListener('click', function() {
      overlay.remove();
    });

    modal.querySelector('.llm-btn-settings').addEventListener('click', function() {
      overlay.remove();
      showApiKeyModal(null);
    });

    modal.querySelector('.llm-btn-copy').addEventListener('click', function() {
      var text = modal.querySelector('.llm-chat-response').textContent;
      navigator.clipboard.writeText(text).then(function() {
        var toast = document.createElement('div');
        toast.className = 'llm-copied-toast';
        toast.textContent = 'Copied!';
        document.body.appendChild(toast);
        setTimeout(function() { toast.remove(); }, 1500);
      });
    });

    overlay.addEventListener('click', function(e) {
      if (e.target === overlay) overlay.remove();
    });

    var escHandler = function(e) {
      if (e.key === 'Escape') {
        overlay.remove();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);
  }

  // --- Streaming API Call ---

  function streamResponse(userPrompt) {
    var responseEl = document.querySelector('.llm-chat-response');
    if (!responseEl) return;

    responseEl.textContent = '';
    responseEl.classList.remove('llm-loading');
    responseEl.classList.add('llm-streaming');

    var fullText = '';

    fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true'
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 2048,
        stream: true,
        system: SYSTEM_PROMPT,
        messages: [{ role: 'user', content: userPrompt }]
      })
    })
    .then(function(response) {
      if (response.status === 401) {
        responseEl.classList.remove('llm-streaming');
        responseEl.innerHTML = '<p class="llm-error">Invalid API key. Click the gear icon to update it.</p>';
        localStorage.removeItem('claude-api-key');
        apiKey = null;
        return;
      }
      if (!response.ok) {
        return response.text().then(function(body) {
          throw new Error('API error ' + response.status + ': ' + body.slice(0, 200));
        });
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      function processChunk() {
        reader.read().then(function(result) {
          if (result.done) {
            finishResponse(fullText, responseEl);
            return;
          }

          buffer += decoder.decode(result.value, { stream: true });
          var lines = buffer.split('\n');
          buffer = lines.pop(); // keep incomplete line

          for (var i = 0; i < lines.length; i++) {
            var line = lines[i];
            if (line.startsWith('data: ')) {
              var data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                var parsed = JSON.parse(data);
                if (parsed.type === 'content_block_delta' &&
                    parsed.delta && parsed.delta.text) {
                  fullText += parsed.delta.text;
                  responseEl.textContent = fullText;
                  responseEl.scrollTop = responseEl.scrollHeight;
                }
              } catch (e) { /* skip unparseable lines */ }
            }
          }

          processChunk();
        }).catch(function(err) {
          finishResponse(fullText || 'Stream interrupted: ' + err.message, responseEl);
        });
      }

      processChunk();
    })
    .catch(function(err) {
      responseEl.classList.remove('llm-streaming');
      responseEl.innerHTML = '<p class="llm-error">' + escapeHtml(err.message) + '</p>';
    });
  }

  function finishResponse(text, el) {
    el.classList.remove('llm-streaming');
    el.innerHTML = renderMarkdown(text);

    // Re-typeset math if MathJax is available
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([el]).catch(function() {});
    } else if (window.MathJax && window.MathJax.typeset) {
      try { window.MathJax.typeset([el]); } catch (e) {}
    }
  }

  // --- Markdown Rendering ---

  function renderMarkdown(text) {
    if (!text) return '';

    // Protect LaTeX display blocks ($$...$$)
    var displayLatex = [];
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, function(m) {
      displayLatex.push(m);
      return '\x00DLATEX' + (displayLatex.length - 1) + '\x00';
    });

    // Protect LaTeX inline ($...$) — avoid matching currency like $5
    var inlineLatex = [];
    text = text.replace(/\$([^\$\n]+?)\$/g, function(m) {
      inlineLatex.push(m);
      return '\x00ILATEX' + (inlineLatex.length - 1) + '\x00';
    });

    // Protect fenced code blocks
    var codeBlocks = [];
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, function(m, lang, code) {
      codeBlocks.push(
        '<pre><code' + (lang ? ' class="language-' + lang + '"' : '') + '>' +
        escapeHtml(code) + '</code></pre>'
      );
      return '\x00CBLOCK' + (codeBlocks.length - 1) + '\x00';
    });

    // Protect inline code
    var inlineCodes = [];
    text = text.replace(/`([^`\n]+)`/g, function(m, code) {
      inlineCodes.push('<code>' + escapeHtml(code) + '</code>');
      return '\x00ICODE' + (inlineCodes.length - 1) + '\x00';
    });

    // HTML-escape the remaining text
    text = escapeHtml(text);

    // Bold and italic
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headers
    text = text.replace(/^#### (.+)$/gm, '<h5>$1</h5>');
    text = text.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    text = text.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    text = text.replace(/^# (.+)$/gm, '<h2>$1</h2>');

    // Unordered list items
    text = text.replace(/^[-*] (.+)$/gm, '<li>$1</li>');

    // Numbered list items
    text = text.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Paragraphs: split on double newlines
    var blocks = text.split(/\n\n+/);
    var html = blocks.map(function(block) {
      block = block.trim();
      if (!block) return '';
      // Don't wrap already-wrapped elements
      if (/^<(h[2-5]|pre|li|ul|ol|\x00)/.test(block)) {
        // Wrap loose <li> in <ul>
        if (block.indexOf('<li>') >= 0 && block.indexOf('<ul>') < 0 && block.indexOf('<ol>') < 0) {
          return '<ul>' + block + '</ul>';
        }
        return block;
      }
      return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
    }).join('\n');

    // Restore inline code
    for (var i = 0; i < inlineCodes.length; i++) {
      html = html.replace('\x00ICODE' + i + '\x00', inlineCodes[i]);
    }

    // Restore code blocks
    for (var i = 0; i < codeBlocks.length; i++) {
      html = html.replace('\x00CBLOCK' + i + '\x00', codeBlocks[i]);
    }

    // Restore LaTeX
    for (var i = 0; i < inlineLatex.length; i++) {
      html = html.replace('\x00ILATEX' + i + '\x00', inlineLatex[i]);
    }
    for (var i = 0; i < displayLatex.length; i++) {
      html = html.replace('\x00DLATEX' + i + '\x00', displayLatex[i]);
    }

    return html;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

})();
