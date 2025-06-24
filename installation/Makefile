# Makefile for Corpus CLI
.PHONY: install setup clean test help dev

# Default Python interpreter
PYTHON := python3
VENV := env
VENV_BIN := $(VENV)/bin
CORPUS := $(VENV_BIN)/corpus

# Detect OS
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
    CORPUS := $(VENV_BIN)/corpus.exe
    ACTIVATE := $(VENV_BIN)/activate.bat
else
    ACTIVATE := source $(VENV_BIN)/activate
endif

help: ## Show this help message
	@echo 'Corpus CLI Installation'
	@echo ''
	@echo 'Usage:'
	@echo '  make install    Install Corpus CLI (recommended)'
	@echo '  make setup      Run configuration wizard after install'
	@echo '  make dev        Install with development dependencies'
	@echo '  make test       Run tests'
	@echo '  make clean      Remove virtual environment'
	@echo '  make help       Show this message'

install: $(VENV) ## Install Corpus CLI
	@echo "Installing Corpus CLI..."
	@$(VENV_BIN)/pip install --upgrade pip wheel setuptools
	@$(VENV_BIN)/pip install -r requirements.txt
	@$(VENV_BIN)/pip install -e .
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment:"
	@echo "   $$ $(ACTIVATE)"
	@echo ""
	@echo "2. Configure Corpus:"
	@echo "   $$ corpus config setup"
	@echo ""
	@echo "3. Start using Corpus:"
	@echo "   $$ corpus --help"

$(VENV): ## Create virtual environment
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)

setup: ## Run configuration wizard
	@if [ -f "$(CORPUS)" ]; then \
		$(CORPUS) config setup; \
	else \
		echo "Error: Corpus not installed. Run 'make install' first."; \
		exit 1; \
	fi

dev: $(VENV) ## Install with development dependencies
	@echo "Installing Corpus CLI with dev dependencies..."
	@$(VENV_BIN)/pip install --upgrade pip wheel setuptools
	@$(VENV_BIN)/pip install -r requirements.txt
	@$(VENV_BIN)/pip install -e ".[dev]"
	@echo "✓ Development installation complete!"

test: ## Run tests
	@if [ -f "$(VENV_BIN)/pytest" ]; then \
		$(VENV_BIN)/pytest tests/; \
	else \
		echo "Error: pytest not installed. Run 'make dev' first."; \
		exit 1; \
	fi

clean: ## Remove virtual environment
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@rm -rf *.egg-info
	@rm -rf build dist
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "✓ Cleanup complete!"

.DEFAULT_GOAL := help