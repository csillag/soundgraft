.PHONY: install install-dev clean build release

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

clean:
	rm -rf dist/ build/ src/*.egg-info

build: clean
	python -m build

release: build
	twine upload dist/*
