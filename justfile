# Project task runner
# Run `just` or `just --list` to see available recipes

set dotenv-load := false
set shell := ["bash", "-cu"]

# Configuration
python_version := "3.12"
source_dir := "src"
test_dir := "tests/"

# Display available recipes
default:
    @just --list

# ----------------------------
# Environment Management
# ----------------------------

# Set up Python version, venv, and install dependencies
init:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔧 Installing uv if missing..."
    if ! command -v uv >/dev/null 2>&1; then
        echo "📦 Installing uv..."
        python3 -m pip install --user --upgrade uv
    else
        echo "✅ uv is already installed"
    fi
    echo "🐍 Setting up Python {{ python_version }} environment..."
    uv python install {{ python_version }}
    uv venv --python {{ python_version }} .venv
    echo "📦 Installing project dependencies..."
    uv sync --extra dev
    . .venv/bin/activate && uv pip install -e .
    echo "🔗 Setting up pre-commit hooks..."
    if [ -f .pre-commit-config.yaml ]; then
        uv run pre-commit install
        echo "✅ Pre-commit hooks installed"
    else
        echo "⚠️  No .pre-commit-config.yaml found, skipping pre-commit setup"
    fi
    echo "🎉 Environment setup complete!"

# Sync project dependencies
sync:
    @echo "Syncing project dependencies..."
    uv sync --extra dev
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Clean Python caches and tooling artifacts
clean-project:
    @echo "Cleaning project caches..."
    find . -type d \( -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' -o -name '__pycache__' \) -exec rm -rf {} +
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Remove the virtual environment folder
clean-env:
    @echo "Deleting virtual environment..."
    rm -rf .venv
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# ----------------------------
# Code Quality
# ----------------------------

# Format codebase using ruff
format:
    @echo "Formatting code with ruff..."
    uv run ruff format
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Lint code using ruff and autofix issues
lint:
    @echo "Running lint checks with ruff..."
    uv run ruff check . --fix
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Perform static type checks using mypy
type-check:
    @echo "Running type checks with mypy..."
    uv run --extra dev mypy {{ source_dir }}
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# ----------------------------
# Tests
# ----------------------------

# Run standard tests with coverage report (excludes integration)
test:
    @echo "Running tests with pytest..."
    uv run python -m pytest -m "not integration" -vv -s {{ test_dir }} \
        --cov={{ source_dir }} \
        --cov-config=pyproject.toml \
        --cov-fail-under=70 \
        --cov-report=term-missing
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Run integration tests with pytest
test-integration:
    @echo "Running INTEGRATION tests with pytest..."
    uv run python -m pytest -m integration -vv --verbose -s {{ test_dir }}
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Run all tests including integration tests
test-all:
    @echo "Running ALL tests with pytest..."
    uv run python -m pytest -vv -s {{ test_dir }} \
        --cov={{ source_dir }} \
        --cov-config=pyproject.toml \
        --cov-fail-under=70 \
        --cov-report=term-missing
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# ----------------------------
# Branch Validation
# ----------------------------

# Run formatting, linting, type checks, and tests
validate-branch: format lint type-check test
    @echo "🎉 Branch validation successful - ready for PR!"
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# ----------------------------
# Examples
# ----------------------------

# Process documents with loading and chunking demonstration
process-documents data_path="aiee_toolset/test_data":
    @echo "🚀 Running document processing demonstration..."
    @echo "📁 Data path: {{ data_path }}"
    uv run python -m {{ source_dir }}.data_loading.example --data-path {{ data_path }}
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Run deep research pipeline on a topic
deep-research topic:
    @echo "🔬 Running deep research on: {{ topic }}..."
    uv run python -m {{ source_dir }}.agents.pydantic.research --topic "{{ topic }}"
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Run LlamaIndex ReAct agent with OpenAI GPT-4
llamaindex-react model="openai:gpt-4":
    @echo "🧠 Running ReAct agent with {{ model }}..."
    uv run python -m {{ source_dir }}.agents.llamaindex.simple_react --model "{{ model }}"
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'

# Run ReAct agent demo
react-demo:
    @echo "🤖 Running ReAct agent demo..."
    uv run python -m {{ source_dir }}.agents.react_demo
    @printf '\033[0;32m--------------------------------------------------\033[0m\n'
