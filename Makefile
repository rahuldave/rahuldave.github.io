.PHONY: preview render publish

preview:
	quarto preview

render:
	quarto render

publish: render
	cp -a _site/* docs/
	git add .
	git commit
	git push
