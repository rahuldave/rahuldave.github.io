# Stamp files live in _site/ so a clean render blows them away naturally
RENDER_STAMP   := _site/.stamp.render
LLM_STAMP      := _site/.stamp.llm-context
BUNDLES_STAMP  := _site/.stamp.bundles

# ── Source file globs ────────────────────────────────────────────────
QUARTO_SOURCES := $(shell find posts til collections -name '*.ipynb' -o -name '*.qmd' -o -name '*.md' 2>/dev/null)
QUARTO_SOURCES += $(wildcard *.qmd *.md)
QUARTO_CONFIG  := _quarto.yml $(wildcard styles/*.scss _filters/*.lua includes/*.html assets/*.js)

NOTEBOOK_SOURCES := $(shell find posts -name 'index.ipynb' 2>/dev/null)
DATA_SOURCES     := $(shell find posts -path '*/data/*' -o -path '*/assets/*' 2>/dev/null)

# ── Phony targets (always run) ──────────────────────────────────────
.PHONY: preview build deploy test-bundles clean

## Live preview with hot reload
preview:
	quarto preview

## Build: render + llm-context + bundles (only re-runs stale stages)
build: $(BUNDLES_STAMP)

## Deploy whatever is in _site/ to gh-pages
deploy: build
	@echo "Deploying to gh-pages..."
	@rm -rf /tmp/gh-pages-deploy
	@git worktree add /tmp/gh-pages-deploy gh-pages
	@rsync -av --delete --exclude='.git' --exclude='.stamp.*' _site/ /tmp/gh-pages-deploy/
	@cd /tmp/gh-pages-deploy && git add -A && git commit -m "Deploy site" --allow-empty && git push origin gh-pages
	@git worktree remove /tmp/gh-pages-deploy
	@echo "Done! Site deployed to gh-pages branch."

## Test generated bundles via juv exec
test-bundles: $(BUNDLES_STAMP)
	python3 _scripts/test_bundles.py --report _site/test-report.json

## Blow away all stamps (forces full rebuild on next make build)
clean:
	rm -f _site/.stamp.*

# ── Stamped targets (skip if sources haven't changed) ───────────────

## Render the full site to _site/
$(RENDER_STAMP): $(QUARTO_SOURCES) $(QUARTO_CONFIG)
	quarto render
	rm -f _site/CLAUDE.html
	@touch $@

## Generate LLM context files (_content.md + cells.json) in _site/
$(LLM_STAMP): $(RENDER_STAMP) _scripts/generate_llm_context.py $(NOTEBOOK_SOURCES)
	python3 _scripts/generate_llm_context.py
	@touch $@

## Generate downloadable zip bundles in _site/posts/*/
$(BUNDLES_STAMP): $(LLM_STAMP) _scripts/generate_bundles.py $(NOTEBOOK_SOURCES) $(DATA_SOURCES)
	python3 _scripts/generate_bundles.py
	@touch $@
