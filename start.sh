#!/bin/sh
# Start Ollama, pull the configured model in the background, then serve the app.
# The interface works immediately in extractive mode and switches to language
# model synthesis automatically once the model is available.
ollama serve > /tmp/ollama.log 2>&1 &
( sleep 4 && ollama pull "${ONCORA_LLM_MODEL}" > /tmp/ollama-pull.log 2>&1 ) &
exec uvicorn oncora.api.main:app --host 0.0.0.0 --port 7860 --workers 1
