.PHONY: preview render build llm-context bundles test-bundles deploy

## Live preview with hot reload
preview:
	quarto preview

## Render the full site to _site/, remove non-site files
render:
	quarto render
	rm -f _site/CLAUDE.html

## Generate LLM context files (_content.md + cells.json) in _site/
llm-context:
	python3 _scripts/generate_llm_context.py

## Generate downloadable zip bundles in _site/posts/*/
bundles:
	python3 _scripts/generate_bundles.py

## Test generated bundles via juv exec
test-bundles:
	python3 _scripts/test_bundles.py --report _site/test-report.json

## Build: render site + generate LLM context + bundles
build: render llm-context bundles

## Deploy _site/ to gh-pages branch via git worktree
deploy: build
	@echo "Deploying to gh-pages..."
	@rm -rf /tmp/gh-pages-deploy
	@git worktree add /tmp/gh-pages-deploy gh-pages
	@rsync -av --delete --exclude='.git' _site/ /tmp/gh-pages-deploy/
	@cd /tmp/gh-pages-deploy && git add -A && git commit -m "Deploy site" --allow-empty && git push origin gh-pages
	@git worktree remove /tmp/gh-pages-deploy
	@echo "Done! Site deployed to gh-pages branch."
