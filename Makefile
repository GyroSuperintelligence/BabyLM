.PHONY: help install test lint format clean run bootstrap

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run flake8 linter"
	@echo "  format      Format code with black and isort"
	@echo "  clean       Remove Python and test artifacts"
	@echo "  run         Run main.py"
	@echo "  bootstrap   Initialize S2 structure and epigenome"

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest scripts/tests/test_gyrosi.py

lint:
	flake8 .
	mypy .

format:
	black .
	isort .

clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf s2_information/agents/
	rm -rf s2_information/agency/g1_information/
	rm -rf s2_information/agency/g4_information/
	rm -rf s2_information/agency/g5_information/

run:
	python main.py

bootstrap:
	python scripts/genesis.py
