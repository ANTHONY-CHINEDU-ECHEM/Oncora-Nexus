# Deployment

## Local workstation

```bash
make install && make all
ollama pull llama3.1:8b
make serve
```

The service starts in about 5 seconds and holds roughly 1.5 GB of memory. Without Ollama it runs in extractive mode and switches to language model synthesis automatically when a model becomes available (checked every 30 seconds).

## Docker Compose with a local model

```bash
docker compose -f deploy/compose.yml up --build
```

Three services: `ollama`, a one shot `model_pull`, and `api`, which builds the interface, downloads data, builds the index and trains every model during the image build. Set `ONCORA_LLM_MODEL=qwen2.5:3b` for a smaller model, or uncomment the GPU block for NVIDIA acceleration.

## Hugging Face Spaces

```bash
bash deploy/hf_space/prepare_space.sh ../oncora-space
cd ../oncora-space && git init && git remote add origin https://huggingface.co/spaces/<user>/<space> && git add . && git commit -m "Oncora Nexus" && git push origin main
```

The Space image embeds Ollama and pulls `qwen2.5:3b` at start up; the interface is usable in extractive mode while the model downloads. Upgrade the hardware and set `ONCORA_LLM_MODEL` for larger models.

## Static browser demo (GitHub Pages)

`frontend/public/demo` holds the exported bundle, so a plain `npm run build` produces a fully working static site in `frontend/dist`. The workflow `.github/workflows/pages.yml` publishes it on every push that changes the frontend. Regenerate the bundle after retraining with `python scripts/export_demo.py`.

## Hospital network

* Run the API and Ollama on an internal host; set `ONCORA_CORS_ORIGINS` to the internal domain.
* Place the service behind the trust's reverse proxy and single sign on; the API itself is stateless apart from the audit database (`ONCORA_AUDIT_DB`).
* Schedule `scripts/fetch_live_trials.py --country "United Kingdom"` nightly, followed by `scripts/build_index.py` and `scripts/train_models.py`.
* Replace the knowledge cards with locally approved protocols; the pathway rules reference cards by identifier.
