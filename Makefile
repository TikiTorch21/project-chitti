# Makefile for ML Course (Powered by uv)
# This file automates common tasks for the project.
# It uses 'uv' for all environment and package management.

# By default, running 'make' will display the help message.
.DEFAULT_GOAL := help

# Define the shell for consistency.
SHELL := /bin/bash

# Declare targets that are not files.
.PHONY: help setup-and-check install-dev format lint test clean

# --- User-Facing Commands ---

help:
	@echo "Commands:"
	@echo "  make install-dev      - Create a .venv and install all dependencies from pyproject.toml."
	@echo "  make format           - Auto-format all Python code with black and ruff."
	@echo "  make lint             - Check the code for style issues and errors with ruff."
	@echo "  make test             - Run all unit tests with pytest."
	@echo "  make setup-and-check  - A handy shortcut to run install, lint, and test in one go."
	@echo "  make clean            - Remove the virtual environment and temporary files."

# A convenient meta-target for first-time setup or CI checks.
setup-and-check: install-dev lint test
	@echo "✅ Environment is set up and all checks passed."

# --- Core Workflow Targets ---

# Creates the virtual environment and installs dependencies.
install-dev:
	@echo "› Creating virtual environment with uv..."
	uv venv
	@echo "› Installing dependencies from pyproject.toml..."
	uv pip install -e ".[dev]"
	@echo "✅ Dependencies installed successfully into ./.venv"
	@echo "💡 To work interactively, activate the environment: source .venv/bin/activate"

# Formats code using the tools from the virtual environment.
# 'uv run' executes a command within the managed environment without needing to activate it first.
format:
	@echo "› Formatting code with black and ruff..."
	uv run black .
	uv run ruff check . --fix

# Lints code using the ruff instance from the virtual environment.
lint:
	@echo "› Linting code with ruff..."
	uv run ruff check .

# Runs tests using the pytest instance from the virtual environment.
test:
	@echo "› Running tests with pytest..."
	uv run pytest

# Removes the virtual environment and other build artifacts.
clean:
	@echo "› Cleaning up project..."
	rm -rf .venv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "🧹 Clean-up complete."