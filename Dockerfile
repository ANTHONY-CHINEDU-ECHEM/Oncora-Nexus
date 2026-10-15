# Hugging Face Spaces (Docker SDK) image: Oncora Nexus with an embedded Ollama.
# A small model keeps CPU inference practical on the free tier; upgrade the
# Space hardware and set ONCORA_LLM_MODEL=llama3.1:8b for higher quality.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ONCORA_LLM_MODEL=qwen2.5:3b \
    ONCORA_OLLAMA_URL=http://127.0.0.1:11434 \
    ONCORA_LLM_TIMEOUT=240 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    HOME=/home/user
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://ollama.com/install.sh | sh
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt ./
RUN pip install -r requirements.txt
COPY --chown=user oncora ./oncora
COPY --chown=user scripts ./scripts
USER user
RUN mkdir -p reports/figures \
 && python scripts/download_data.py \
 && python scripts/build_index.py \
 && python scripts/train_models.py \
 && python scripts/run_evaluation.py --skip-nli --mode extractive \
 && rm -rf data/raw/pubmed_rct/train.7z data/raw/pubmed_rct/train.txt
COPY --from=web --chown=user /web/dist ./frontend/dist
COPY --chown=user deploy/hf_space/start.sh ./start.sh
EXPOSE 7860
CMD ["sh", "start.sh"]
