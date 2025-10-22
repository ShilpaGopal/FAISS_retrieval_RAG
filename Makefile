# Variables
# Default to python3; override with `make PYTHON=/path/to/python`
PYTHON ?= python3.10
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip

.PHONY: help venv clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create the local Python virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

clean: ## Remove the virtual environment folder
	rm -rf $(VENV_DIR)
