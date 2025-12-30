# jetson-engine-builder
contenedor que expone una API, le subís un .onnx, corre trtexec dentro de la Jetson, y te devuelve el .engine + logs. Además, una UI mínima en Gradio para subir el archivo, ver el log en vivo y descargar el engine.

Jetson está en L4T R32.7.6 (JetPack 4.x). 

No intentar “instalar TensorRT dentro del contenedor”

Montar TensorRT del host (/usr/src/tensorrt) y opcionalmente libs necesarias, para que el engine se genere con la misma versión real que usa tu Jetson.


✅ FastAPI con jobs (no bloquea): POST /api/jobs → te devuelve job_id

✅ SSE para logs en vivo: GET /api/jobs/{id}/stream (o UI lo muestra)

✅ Descarga: GET /api/jobs/{id}/download (zip con .engine + log + metadata.json)

✅ Gradio UI minimalista: upload ONNX, opciones, log live, botón descargar

✅ Aviso fuerte: engine específico de ESA Jetson / JetPack / TensorRT

✅ Límites (tamaño ONNX) + limpieza automática por TTL

✅ Workspace configurable (default 512) + soporta shapes dinámicos opcionales

✅ Auto-detección ONNX

Cuando subís el .onnx, te muestra:

Input detectado

Shapes sugeridos (ej: images:1x3x640x640)
Y podés activar “Usar shapes sugeridos (auto)” para que rellene min/opt/max.

✅ Progreso por etapas

Se actualiza desde el log:

Parse ONNX

Build network

Tactics / kernel selection

Serialize engine

Done

# Levantar el servicio
mkdir -p ~/jetson-engine-builder/app
cd ~/jetson-engine-builder
# pegá los 4 archivos
docker compose up -d --build
docker logs -f jetson_engine_builder



# Usar la UI
Abrís en el navegador:

UI Gradio: http://JETSON_IP:8000/

Health: http://JETSON_IP:8000/api/health

# 🧪 Usar la API (modo pro)
Crear job
curl -X POST "http://JETSON_IP:8000/api/jobs" \
  -F "file=@/ruta/model.onnx" \
  -F "fp16=true" \
  -F "workspace=512"


Te devuelve job_id.

# Ver estado
curl "http://JETSON_IP:8000/api/jobs/<JOB_ID>"

Ver log
curl "http://JETSON_IP:8000/api/jobs/<JOB_ID>/log"

# Log en vivo (SSE)
curl -N "http://JETSON_IP:8000/api/jobs/<JOB_ID>/stream"

# Descargar ZIP (engine+log+metadata)
curl -L "http://JETSON_IP:8000/api/jobs/<JOB_ID>/download" -o result.zip



## Notas importantes (para que quede “pro”)

El .engine es específico de:

Jetson / arquitectura

versión de TensorRT

versión de JetPack / L4T

Si cambiás JetPack o TensorRT, regenerá engines.

workspace: en Nano 4GB, típicamente 256–1024 (si te quedás sin memoria, bajá).