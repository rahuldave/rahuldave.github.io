.PHONY: preview render build llm-context

## Live preview with hot reload
preview:
	quarto preview

## Render the full site to _site/
render:
	quarto render

## Generate LLM context files (content.md + cells.json) in _site/
llm-context:
	python3 _scripts/generate_llm_context.py

## Render and sync to docs/ for GitHub Pages (only copies changed files)
build: render llm-context
	rsync -av _site/ docs/
