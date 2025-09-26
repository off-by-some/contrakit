#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project root (one level up from scripts/)
cd "$SCRIPT_DIR/.."

# Verify we're in the right place
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Script must be in scripts/ subdirectory of project root."
    exit 1
fi


poetry run pytest -q
poetry build
poetry publish -r testpypi