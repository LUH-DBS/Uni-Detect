install:
	conda env create --file environment.yml

uninstall:
	conda remove --name Uni-Detect --all

.PHONY: install, uninstall