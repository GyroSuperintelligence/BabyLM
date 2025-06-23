.PHONY: help install test lint format clean run dev export-example import-example

help:
	@echo "Available commands:"
	@echo "  install         Install dependencies"
	@echo "  test           Run tests"
	@echo "  lint           Run linting"
	@echo "  format         Format code"
	@echo "  clean          Clean build artifacts"
	@echo "  run            Run the application"
	@echo "  dev            Run in development mode"
	@echo "  export-example Export example knowledge"
	@echo "  import-example Import example knowledge"

install:
	pip install -r requirements.txt
	pre-commit install

test:
	python -m pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/

run:
	python src/main.py

dev:
	python scripts/dev.py

export-example:
	python -m gyro_tools.gyro_knowledge_manager export --knowledge-id example --output examples/example_knowledge.gyro

import-example:
	python -m gyro_tools.gyro_knowledge_manager import --input examples/example_knowledge.gyro --new-session
