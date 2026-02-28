.PHONY: preview render build

## Live preview with hot reload
preview:
	quarto preview

## Render the full site to _site/
render:
	quarto render

## Render and sync to docs/ for GitHub Pages (only copies changed files)
build: render
	rsync -av _site/ docs/
