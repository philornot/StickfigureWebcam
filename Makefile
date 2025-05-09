# Makefile - Simplifies common development tasks
# Run 'make help' to see available commands

.PHONY: clean clean-build clean-pyc help test lint type-check dev-setup format run

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "test - run tests quickly with the default Python"
	@echo "lint - check style with flake8"
	@echo "type-check - check types with mypy"
	@echo "dev-setup - set up development environment"
	@echo "format - format code with black and isort"
	@echo "run - run the application"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

test:
	python -m pytest

lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

type-check:
	mypy src tests

dev-setup:
	pip install -e ".[dev]"
	pre-commit install

format:
	black src tests
	isort src tests

run:
	python src/main.py