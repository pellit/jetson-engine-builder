import os
import re
import json
import time
import uuid
import shutil
import zipfile
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import gradio as gr
import onnx

APP_NAME = "Jetson ONNX → TensorRT Engine Builder"
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()

UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
for d in (UPLOADS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "24"))
MAX_CONCURRENT_JOBS = int(
    os.getenv("MAX_CONCURRENT_JOBS", "1"))  # en Nano conviene 1

jobs: Dict[str, Dict[str, Any]] = {}
job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# ---------------- Utils ----------------


def now_ts() -> float:
    return time.time()


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def job_paths(job_id: str) -> Dict[str, Path]:
    base = JOBS_DIR / job_id
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "onnx": base / "model.onnx",
        "engine": base / "model.engine",
        "log": base / "build.log",
        "meta": base / "metadata.json",
        "zip": base / "result.zip",
    }


def set_job_state(job_id: str, **kwargs) -> None:
    jobs.setdefault(job_id, {})
    jobs[job_id].update(kwargs)


def get_job(job_id: str) -> Dict[str, Any]:
    if job_id not in jobs:
        base = JOBS_DIR / job_id
        if not base.exists():
            raise KeyError(job_id)
        paths = job_paths(job_id)
        status = "unknown"
        if paths["zip"].exists():
            status = "done"
        elif paths["log"].exists():
            status = "running_or_failed"
        jobs[job_id] = {"id": job_id, "status": status,
                        "created_at": base.stat().st_mtime}
    return jobs[job_id]


def append_log(job_id: str, text: str) -> None:
    paths = job_paths(job_id)
    with open(paths["log"], "a", encoding="utf-8") as f:
        f.write(text)


def write_meta(job_id: str, meta: Dict[str, Any]) -> None:
    paths = job_paths(job_id)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_zip(job_id: str) -> Path:
    paths = job_paths(job_id)
    zpath = paths["zip"]
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if paths["engine"].exists():
            z.write(paths["engine"], arcname="model.engine")
        if paths["log"].exists():
            z.write(paths["log"], arcname="build.log")
        if paths["meta"].exists():
            z.write(paths["meta"], arcname="metadata.json")
    return zpath


def find_trtexec() -> Optional[str]:
    p = shutil.which("trtexec")
    if p:
        return p
    candidate = "/usr/src/tensorrt/bin/trtexec"
    if Path(candidate).exists():
        return candidate
    return None


def tensorrt_info() -> Dict[str, str]:
    trtexec = find_trtexec()
    if not trtexec:
        return {"trtexec": "NOT_FOUND", "tensorrt": "UNKNOWN"}
    try:
        proc = subprocess.run([trtexec, "--version"],
                              capture_output=True, text=True)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"TensorRT\s+v(\d+)", out)
        trt = m.group(1) if m else "UNKNOWN"
        return {"trtexec": trtexec, "tensorrt": trt}
    except Exception:
        return {"trtexec": trtexec, "tensorrt": "UNKNOWN"}

# ---------------- ONNX Inspection (auto input name + shape) ----------------


def _dim_to_int(d) -> Optional[int]:
    # ONNX dim can be dim_value (int) or dim_param (str) or missing
    if hasattr(d, "dim_value") and d.dim_value:
        return int(d.dim_value)
    return None


def inspect_onnx_inputs(onnx_path: Path) -> Dict[str, Any]:
    """
    Devuelve:
      - primary_input_name
      - primary_input_shape (list[int|None]) incluyendo batch si está
      - suggested_shapes_str: 'name:1x3x640x640' si se puede sugerir
      - all_inputs: lista de {name, shape}
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    inputs = []
    for inp in graph.input:
        name = inp.name
        # shape
        shape = []
        t = inp.type.tensor_type
        if t.HasField("shape"):
            for dim in t.shape.dim:
                shape.append(_dim_to_int(dim))
        inputs.append({"name": name, "shape": shape})

    # Heurística: elegir input con 4 dims y canal=3 si existe, sino el primero
    primary = inputs[0] if inputs else {"name": "images", "shape": []}
    for x in inputs:
        shp = x["shape"]
        if len(shp) == 4 and (shp[1] == 3 or shp[1] is None):
            primary = x
            break

    # Sugerencia shapes: si hay dims None, ponemos defaults típicos
    suggested = ""
    if primary.get("name"):
        shp = primary.get("shape", [])
        if len(shp) == 4:
            b, c, h, w = shp
            b = b or 1
            c = c or 3
            # Si no hay H/W, sugerimos 640 (por defecto YOLOv8)
            h = h or 640
            w = w or 640
            suggested = f"{primary['name']}:{b}x{c}x{h}x{w}"

    return {
        "primary_input_name": primary.get("name", ""),
        "primary_input_shape": primary.get("shape", []),
        "suggested_shapes_str": suggested,
        "all_inputs": inputs,
    }

# ---------------- Progress from log (approx stages) ----------------


PROGRESS_RULES = [
    # (regex, percent, stage)
    (re.compile(r"Parsing\s+model|Parsing\s+ONNX|ONNX", re.IGNORECASE), 10, "Parse ONNX"),
    (re.compile(r"Building\s+engine|Building\s+an\s+engine|Builder",
     re.IGNORECASE), 25, "Build network"),
    (re.compile(r"tactic|tactics|Timing|kernel selection|Profiling",
     re.IGNORECASE), 55, "Tactics / kernel selection"),
    (re.compile(r"Serializ|serialize|Saving\s+engine|saveEngine",
     re.IGNORECASE), 80, "Serialize engine"),
    (re.compile(r"Engine built|Build\s+time|DONE|✅", re.IGNORECASE), 95, "Finalize"),
]


def update_progress_from_line(job_id: str, line: str) -> None:
    job = jobs.get(job_id, {})
    cur_p = int(job.get("progress", 0))
    cur_stage = job.get("stage", "Starting")

    for rx, pct, stage in PROGRESS_RULES:
        if rx.search(line):
            # nunca bajar progreso
            if pct > cur_p:
                cur_p = pct
                cur_stage = stage
            break

    set_job_state(job_id, progress=cur_p, stage=cur_stage)

# ---------------- Worker ----------------


async def run_trtexec_job(job_id: str, fp16: bool, workspace: int,
                          min_shapes: str, opt_shapes: str, max_shapes: str,
                          onnx_inspect: Dict[str, Any]) -> None:
    async with job_semaphore:
        trt = tensorrt_info()
        trtexec = trt.get("trtexec")
        if not trtexec or trtexec == "NOT_FOUND":
            set_job_state(job_id, status="failed",
                          error="trtexec no encontrado en el contenedor", progress=0, stage="Failed")
            append_log(job_id, "\nERROR: trtexec no encontrado.\n")
            return

        paths = job_paths(job_id)

        cmd = [
            trtexec,
            f"--onnx={str(paths['onnx'])}",
            f"--saveEngine={str(paths['engine'])}",
            f"--workspace={int(workspace)}",
        ]
        if fp16:
            cmd.append("--fp16")

        shapes_used = False
        if any([min_shapes.strip(), opt_shapes.strip(), max_shapes.strip()]):
            if not opt_shapes.strip():
                set_job_state(
                    job_id, status="failed", error="opt_shapes requerido si usás shapes dinámicos", progress=0, stage="Failed")
                append_log(
                    job_id, "\nERROR: opt_shapes requerido si usás shapes dinámicos.\n")
                return
            shapes_used = True
            if min_shapes.strip():
                cmd.append(f"--minShapes={min_shapes.strip()}")
            cmd.append(f"--optShapes={opt_shapes.strip()}")
            if max_shapes.strip():
                cmd.append(f"--maxShapes={max_shapes.strip()}")

        meta = {
            "app": APP_NAME,
            "job_id": job_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engine_specificity_notice": (
                "Este archivo .engine es ESPECÍFICO de la Jetson/JetPack/L4T/TensorRT donde se generó. "
                "Si cambia JetPack o TensorRT, regenerar."
            ),
            "tensorrt": trt,
            "onnx_inspection": onnx_inspect,
            "params": {
                "fp16": fp16,
                "workspace_mib": int(workspace),
                "shapes_used": shapes_used,
                "min_shapes": min_shapes.strip(),
                "opt_shapes": opt_shapes.strip(),
                "max_shapes": max_shapes.strip(),
            },
            "command": " ".join(cmd),
        }
        write_meta(job_id, meta)

        set_job_state(job_id, status="running", started_at=now_ts(),
                      command=meta["command"], progress=5, stage="Starting")
        append_log(job_id, f"=== {APP_NAME} ===\n")
        append_log(job_id, f"NOTICE: {meta['engine_specificity_notice']}\n\n")
        append_log(job_id, f"TensorRT: {trt}\n")
        append_log(
            job_id, f"Auto input detect: {onnx_inspect.get('primary_input_name')} | suggested: {onnx_inspect.get('suggested_shapes_str')}\n")
        append_log(job_id, f"CMD: {meta['command']}\n\n")
        append_log(job_id, "=== LOG ===\n")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdout is not None
            while True:
                line_b = await proc.stdout.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace")
                append_log(job_id, line)
                update_progress_from_line(job_id, line)

            ret = await proc.wait()
            if ret != 0:
                set_job_state(job_id, status="failed", finished_at=now_ts(
                ), error=f"trtexec exit={ret}", progress=0, stage="Failed")
                append_log(job_id, f"\nERROR: trtexec falló (exit={ret}).\n")
                return

            if not paths["engine"].exists() or paths["engine"].stat().st_size < 1024:
                set_job_state(job_id, status="failed", finished_at=now_ts(
                ), error="engine no generado o inválido", progress=0, stage="Failed")
                append_log(job_id, "\nERROR: engine no generado o inválido.\n")
                return

            build_zip(job_id)
            set_job_state(job_id, status="done",
                          finished_at=now_ts(), progress=100, stage="Done")
            append_log(job_id, "\n✅ DONE: Engine generado y ZIP listo.\n")

        except Exception as e:
            set_job_state(job_id, status="failed", finished_at=now_ts(),
                          error=str(e), progress=0, stage="Failed")
            append_log(job_id, f"\nEXCEPTION: {e}\n")


async def cleanup_loop():
    while True:
        try:
            ttl = JOB_TTL_HOURS * 3600
            cutoff = now_ts() - ttl
            for job_dir in JOBS_DIR.iterdir():
                if not job_dir.is_dir():
                    continue
                mtime = job_dir.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    jobs.pop(job_dir.name, None)
        except Exception:
            pass
        await asyncio.sleep(600)

# ---------------- FastAPI ----------------

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(cleanup_loop())


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "tensorrt": tensorrt_info(),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "job_ttl_hours": JOB_TTL_HOURS, "max_concurrent_jobs": MAX_CONCURRENT_JOBS},
        "notice": "El .engine es específico de esta Jetson/JetPack/L4T/TensorRT.",
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    fp16: bool = Form(True),
    workspace: int = Form(512),
    min_shapes: str = Form(""),
    opt_shapes: str = Form(""),
    max_shapes: str = Form(""),
):
    if not file.filename.lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Subí un archivo .onnx")
    if workspace < 16 or workspace > 4096:
        raise HTTPException(
            status_code=400, detail="workspace fuera de rango (16..4096 MiB)")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413, detail=f"Archivo demasiado grande ({size_mb:.1f}MB). Límite: {MAX_UPLOAD_MB}MB")

    job_id = uuid.uuid4().hex[:12]
    paths = job_paths(job_id)

    with open(paths["onnx"], "wb") as f:
        f.write(contents)

    try:
        onnx_inspect = inspect_onnx_inputs(paths["onnx"])
    except Exception as e:
        onnx_inspect = {"error": f"inspect_onnx_inputs failed: {e}",
                        "primary_input_name": "", "suggested_shapes_str": "", "all_inputs": []}

    set_job_state(job_id, id=job_id, status="queued", created_at=now_ts(), filename=safe_name(file.filename),
                  progress=0, stage="Queued", onnx_inspect=onnx_inspect)

    asyncio.create_task(run_trtexec_job(
        job_id, fp16, workspace, min_shapes, opt_shapes, max_shapes, onnx_inspect))

    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "onnx_inspection": onnx_inspect,
        "notice": "El .engine resultante es específico de esta Jetson/JetPack/L4T/TensorRT.",
        "endpoints": {
            "status": f"/api/jobs/{job_id}",
            "stream": f"/api/jobs/{job_id}/stream",
            "download": f"/api/jobs/{job_id}/download",
            "log": f"/api/jobs/{job_id}/log",
        },
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    paths = job_paths(job_id)
    return {
        "ok": True,
        "job": job,
        "files": {
            "engine_exists": paths["engine"].exists(),
            "zip_exists": paths["zip"].exists(),
            "log_exists": paths["log"].exists(),
        },
        "notice": "El .engine es específico de esta Jetson/JetPack/L4T/TensorRT.",
    }


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str):
    try:
        _ = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    paths = job_paths(job_id)
    if not paths["log"].exists():
        raise HTTPException(status_code=404, detail="Log no disponible aún")
    return FileResponse(str(paths["log"]), media_type="text/plain", filename=f"{job_id}.log")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    try:
        job = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    paths = job_paths(job_id)
    if not paths["zip"].exists():
        if paths["engine"].exists() and paths["engine"].stat().st_size > 1024:
            build_zip(job_id)
        else:
            raise HTTPException(
                status_code=409, detail=f"Resultado no listo. Estado: {job.get('status')}")

    return FileResponse(
        str(paths["zip"]),
        media_type="application/zip",
        filename=f"{job_id}_engine.zip",
        headers={
            "X-Engine-Notice": "Engine específico de esta Jetson/JetPack/L4T/TensorRT"},
    )


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str):
    try:
        _ = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    paths = job_paths(job_id)

    async def event_gen():
        last_size = 0
        while True:
            if paths["log"].exists():
                cur_size = paths["log"].stat().st_size
                if cur_size > last_size:
                    with open(paths["log"], "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_size)
                        chunk = f.read()
                        last_size = cur_size
                    yield f"data: {chunk.replace(chr(10), '\\ndata: ')}\n\n"

            job = get_job(job_id)
            if job.get("status") in ("done", "failed"):
                await asyncio.sleep(0.3)
                if paths["log"].exists() and paths["log"].stat().st_size == last_size:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ---------------- Gradio UI ----------------


def ui_inspect(onnx_file) -> Tuple[str, str, str]:
    """
    Devuelve: (input_name, suggested_shapes, all_inputs_text)
    """
    if onnx_file is None:
        return "", "", ""
    p = Path(onnx_file)
    if p.suffix.lower() != ".onnx":
        return "", "", "Archivo inválido (debe ser .onnx)"
    try:
        info = inspect_onnx_inputs(p)
        all_inputs = info.get("all_inputs", [])
        all_txt = "\n".join(
            [f"- {x['name']}: {x['shape']}" for x in all_inputs]) or "(no inputs detectados)"
        return info.get("primary_input_name", ""), info.get("suggested_shapes_str", ""), all_txt
    except Exception as e:
        return "", "", f"Error inspeccionando ONNX: {e}"


def ui_submit(onnx_file, fp16=True, workspace=512, use_suggested=False, min_shapes="", opt_shapes="", max_shapes="", suggested_shapes=""):
    if onnx_file is None:
        return "Subí un .onnx", "", 0, "Queued", None, ""

    src = Path(onnx_file)
    if src.suffix.lower() != ".onnx":
        return "Archivo inválido (debe ser .onnx)", "", 0, "Queued", None, ""

    job_id = uuid.uuid4().hex[:12]
    paths = job_paths(job_id)
    shutil.copy(src, paths["onnx"])

    try:
        onnx_inspect = inspect_onnx_inputs(paths["onnx"])
    except Exception as e:
        onnx_inspect = {"error": str(
            e), "primary_input_name": "", "suggested_shapes_str": "", "all_inputs": []}

    # Si el usuario marca "usar sugerido", rellenamos optShapes (y min/max igual para estabilidad)
    if use_suggested and suggested_shapes.strip():
        opt_shapes = suggested_shapes.strip()
        min_shapes = suggested_shapes.strip()
        max_shapes = suggested_shapes.strip()

    set_job_state(job_id, id=job_id, status="queued", created_at=now_ts(), filename=src.name,
                  progress=0, stage="Queued", onnx_inspect=onnx_inspect)

    asyncio.get_event_loop().create_task(
        run_trtexec_job(job_id, bool(fp16), int(workspace),
                        min_shapes, opt_shapes, max_shapes, onnx_inspect)
    )

    return f"Job creado: {job_id}", job_id, 0, "Queued", None, ""


def ui_poll(job_id: str):
    if not job_id:
        return "Sin job", "", 0, "Queued", None

    job = get_job(job_id)
    paths = job_paths(job_id)

    log_text = ""
    if paths["log"].exists():
        log_text = paths["log"].read_text(
            encoding="utf-8", errors="replace")[-25000:]

    status = job.get("status", "unknown")
    prog = int(job.get("progress", 0))
    stage = job.get("stage", "Starting")

    zip_path = None
    if paths["zip"].exists():
        zip_path = str(paths["zip"])
    elif paths["engine"].exists() and paths["engine"].stat().st_size > 1024:
        build_zip(job_id)
        zip_path = str(paths["zip"])

    return f"Estado: {status}", log_text, prog, stage, zip_path


with gr.Blocks(title=APP_NAME) as demo:
    gr.Markdown(
        f"# {APP_NAME}\n"
        f"Subí un **.onnx** y generá un **.engine** usando **TensorRT del host** (Jetson Nano).\n\n"
        f"⚠️ **IMPORTANTE:** el `.engine` es **ESPECÍFICO** de esta **Jetson / JetPack (L4T R32.7.6) / TensorRT**.\n"
    )

    with gr.Row():
        onnx_in = gr.File(label="Modelo ONNX (.onnx)", file_types=[".onnx"])
    with gr.Row():
        detected_input = gr.Textbox(
            label="Input detectado (auto)", interactive=False)
        suggested_shapes = gr.Textbox(
            label="Shapes sugeridos (auto)", interactive=False)

    all_inputs_txt = gr.Textbox(
        label="Inputs encontrados (debug)", lines=6, interactive=False)

    onnx_in.change(fn=ui_inspect, inputs=[onnx_in], outputs=[
                   detected_input, suggested_shapes, all_inputs_txt])

    with gr.Row():
        fp16_in = gr.Checkbox(value=True, label="FP16 (recomendado)")
        ws_in = gr.Slider(64, 2048, value=512, step=64,
                          label="Workspace (MiB)")
        use_suggested_in = gr.Checkbox(
            value=False, label="Usar shapes sugeridos (auto)")

    with gr.Accordion("Shapes dinámicos (manual, opcional)", open=False):
        gr.Markdown(
            "Formato: `input:1x3x640x640` (usa el nombre del input real)")
        min_in = gr.Textbox(label="minShapes",
                            placeholder="images:1x3x640x640")
        opt_in = gr.Textbox(label="optShapes",
                            placeholder="images:1x3x640x640")
        max_in = gr.Textbox(label="maxShapes",
                            placeholder="images:1x3x640x640")

    submit = gr.Button("Crear Job")
    job_id_out = gr.Textbox(label="Job ID")
    status_out = gr.Textbox(label="Estado")
    stage_out = gr.Textbox(label="Etapa (aprox)")
    progress_out = gr.Slider(0, 100, value=0, step=1,
                             label="Progreso (aprox) %")
    log_out = gr.Textbox(label="Log (tail)", lines=18)
    download_out = gr.File(label="Descargar ZIP (engine+log+metadata)")

    submit.click(
        fn=ui_submit,
        inputs=[onnx_in, fp16_in, ws_in, use_suggested_in,
                min_in, opt_in, max_in, suggested_shapes],
        outputs=[status_out, job_id_out, progress_out,
                 stage_out, download_out, log_out],
    )

    poll = gr.Button("Actualizar estado / log")
    poll.click(fn=ui_poll, inputs=[job_id_out], outputs=[
               status_out, log_out, progress_out, stage_out, download_out])

# Montar gradio en FastAPI
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(cleanup_loop())


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "tensorrt": tensorrt_info(),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "job_ttl_hours": JOB_TTL_HOURS, "max_concurrent_jobs": MAX_CONCURRENT_JOBS},
        "notice": "El .engine es específico de esta Jetson/JetPack/L4T/TensorRT.",
    }


@app.post("/api/jobs")
async def create_job_api(
    file: UploadFile = File(...),
    fp16: bool = Form(True),
    workspace: int = Form(512),
    min_shapes: str = Form(""),
    opt_shapes: str = Form(""),
    max_shapes: str = Form(""),
):
    if not file.filename.lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Subí un archivo .onnx")
    if workspace < 16 or workspace > 4096:
        raise HTTPException(
            status_code=400, detail="workspace fuera de rango (16..4096 MiB)")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413, detail=f"Archivo demasiado grande ({size_mb:.1f}MB). Límite: {MAX_UPLOAD_MB}MB")

    job_id = uuid.uuid4().hex[:12]
    paths = job_paths(job_id)

    with open(paths["onnx"], "wb") as f:
        f.write(contents)

    try:
        onnx_inspect = inspect_onnx_inputs(paths["onnx"])
    except Exception as e:
        onnx_inspect = {"error": str(
            e), "primary_input_name": "", "suggested_shapes_str": "", "all_inputs": []}

    set_job_state(job_id, id=job_id, status="queued", created_at=now_ts(), filename=safe_name(file.filename),
                  progress=0, stage="Queued", onnx_inspect=onnx_inspect)

    asyncio.create_task(run_trtexec_job(
        job_id, fp16, workspace, min_shapes, opt_shapes, max_shapes, onnx_inspect))

    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "onnx_inspection": onnx_inspect,
        "notice": "El .engine es específico de esta Jetson/JetPack/L4T/TensorRT.",
        "endpoints": {
            "status": f"/api/jobs/{job_id}",
            "stream": f"/api/jobs/{job_id}/stream",
            "download": f"/api/jobs/{job_id}/download",
            "log": f"/api/jobs/{job_id}/log",
        },
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    paths = job_paths(job_id)
    return {
        "ok": True,
        "job": job,
        "files": {
            "engine_exists": paths["engine"].exists(),
            "zip_exists": paths["zip"].exists(),
            "log_exists": paths["log"].exists(),
        },
        "notice": "El .engine es específico de esta Jetson/JetPack/L4T/TensorRT.",
    }


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str):
    try:
        _ = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    paths = job_paths(job_id)
    if not paths["log"].exists():
        raise HTTPException(status_code=404, detail="Log no disponible aún")
    return FileResponse(str(paths["log"]), media_type="text/plain", filename=f"{job_id}.log")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    try:
        job = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    paths = job_paths(job_id)
    if not paths["zip"].exists():
        if paths["engine"].exists() and paths["engine"].stat().st_size > 1024:
            build_zip(job_id)
        else:
            raise HTTPException(
                status_code=409, detail=f"Resultado no listo. Estado: {job.get('status')}")

    return FileResponse(
        str(paths["zip"]),
        media_type="application/zip",
        filename=f"{job_id}_engine.zip",
        headers={
            "X-Engine-Notice": "Engine específico de esta Jetson/JetPack/L4T/TensorRT"},
    )


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str):
    try:
        _ = get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    paths = job_paths(job_id)

    async def event_gen():
        last_size = 0
        while True:
            if paths["log"].exists():
                cur_size = paths["log"].stat().st_size
                if cur_size > last_size:
                    with open(paths["log"], "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_size)
                        chunk = f.read()
                        last_size = cur_size
                    yield f"data: {chunk.replace(chr(10), '\\ndata: ')}\n\n"

            job = get_job(job_id)
            if job.get("status") in ("done", "failed"):
                await asyncio.sleep(0.3)
                if paths["log"].exists() and paths["log"].stat().st_size == last_size:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
