
# Makefile for Fetal Ultrasound Classification

# Variables
PYTHON = venv/bin/python
PIP = venv/bin/pip
SRC_DIR = src
TEST_DIR = tests

.PHONY: help install train test clean

help:
	@echo "Available targets:"
	@echo "  install  : Create venv and install dependencies"
	@echo "  train    : Run the training script"
	@echo "  test     : Run unit tests"
	@echo "  clean    : Remove generated files and caches"

install:
	python3 -m venv venv
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m src.train

test:
	$(PYTHON) -m pytest $(TEST_DIR)

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Done."
