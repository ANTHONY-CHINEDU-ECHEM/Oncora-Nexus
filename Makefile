# Oncora Nexus build pipeline. Run `make all` once, then `make serve`.
PY ?= python3

.PHONY: help install data index train evaluate demo all test lint web serve docker clean

help:
	@echo "make install    install Python and web dependencies"
	@echo "make data       download every public dataset (about 90 MB, plus a 690 MB streamed matrix)"
	@echo "make index      build the hybrid knowledge index"
	@echo "make train      train survival, subtype and trial parsing components"
	@echo "make evaluate   train the reranker and run the full evaluation suite"
	@echo "make demo       export the in browser demo bundle"
	@echo "make all        data, index, train, evaluate, demo and web"
	@echo "make serve      run the API and interface on http://localhost:8000"
	@echo "make docker     build and start the stack with a local Ollama model"

install:
	$(PY) -m pip install -r requirements_dev.txt
	cd frontend && npm ci

data:
	$(PY) scripts/download_data.py

index:
	$(PY) scripts/build_index.py

train:
	$(PY) scripts/train_models.py

evaluate:
	$(PY) scripts/run_evaluation.py

demo:
	$(PY) scripts/export_demo.py

web:
	cd frontend && npm run build

all: data index train evaluate demo web

test:
	$(PY) -m pytest

lint:
	ruff check oncora scripts tests
	cd frontend && npm run typecheck

serve:
	uvicorn oncora.api.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose -f deploy/compose.yml up --build

clean:
	rm -rf data/index data/models data/processed reports/*.json frontend/dist
