#!/bin/bash
# ============================================================
# Ollama Entrypoint
# ============================================================
# Arranca el servidor Ollama y descarga el modelo automaticamente.
# El modelo se cachea en el volumen persistente, asi que solo
# se descarga la primera vez.
# ============================================================

set -e

# Arrancar Ollama en background
ollama serve &
OLLAMA_PID=$!

# Esperar a que Ollama este listo
echo "[entrypoint] Esperando a que Ollama arranque..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done
echo "[entrypoint] Ollama listo."

# Descargar el modelo si no existe
MODEL="${OLLAMA_MODEL:-qwen2:1.5b}"
echo "[entrypoint] Verificando modelo: $MODEL"

if ollama list | grep -q "$(echo $MODEL | cut -d: -f1)"; then
    echo "[entrypoint] Modelo $MODEL ya disponible."
else
    echo "[entrypoint] Descargando modelo $MODEL..."
    ollama pull "$MODEL"
    echo "[entrypoint] Modelo $MODEL descargado."
fi

# Mantener Ollama en foreground
echo "[entrypoint] Ollama sirviendo modelo $MODEL"
wait $OLLAMA_PID
