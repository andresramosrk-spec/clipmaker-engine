
import os
import re
import json
import time
import uuid
import shutil
import subprocess
import tempfile

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from google import genai
from supabase import create_client


app = FastAPI(
    title="ClipMaker AI",
    description="Motor inteligente de procesamiento de videos"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/clipmaker_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_BUCKET = "clipmaker-clips"


def get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en Render"
        )

    return create_client(supabase_url, supabase_key)


def upload_clip_to_supabase(supabase, local_path, job_id, filename):
    storage_path = f"{job_id}/{filename}"

    with open(local_path, "rb") as clip_file:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=clip_file,
            file_options={
                "content-type": "video/mp4",
                "cache-control": "3600"
            }
        )

    return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)

app.mount(
    "/clips",
    StaticFiles(directory=str(JOBS_DIR)),
    name="clips"
)


# ----------------------------------
# ESTADO DEL MOTOR
# ----------------------------------

@app.get("/")
def home():
    return {
        "message": "ClipMaker AI Engine funcionando",
        "version": "3.0",
        "ai": "Gemini",
        "features": [
            "single_clip",
            "multiple_clips",
            "ai_highlights"
        ]
    }


@app.get("/health")
def health():
    return {"status": "online"}


@app.get("/ai-health")
def ai_health():
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="La API Key de Gemini no está configurada"
        )

    return {
        "status": "configured",
        "provider": "Google Gemini"
    }


# ----------------------------------
# FUNCIONES DE VIDEO
# ----------------------------------

def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la duración del video"
        )

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Duración de video no válida"
        )


def make_clip(input_path, output_path, start, duration):
    command = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg no pudo generar el clip: " + result.stderr[-1000:]
        )


def save_uploaded_video(video, destination):
    with open(destination, "wb") as f:
        shutil.copyfileobj(video.file, f)

    size = os.path.getsize(destination)

    if size == 0:
        raise HTTPException(
            status_code=400,
            detail="El archivo de video está vacío"
        )

    if size > 300 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Para esta primera versión utiliza videos de máximo 300 MB"
        )


# ----------------------------------
# GEMINI: DETECCIÓN INTELIGENTE
# ----------------------------------

def analyze_video_with_gemini(
    video_path,
    video_duration,
    quantity,
    clip_duration,
    mode,
    category
):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No está configurada la variable GEMINI_API_KEY en Render"
        )

    client = genai.Client(api_key=api_key)
    uploaded_file = None

    try:
        # Subir video a Gemini
        uploaded_file = client.files.upload(
            file=str(video_path)
        )

        # Esperar a que Gemini procese el archivo
        start_wait = time.monotonic()

        while True:
            state = getattr(uploaded_file, "state", None)
            state_name = getattr(state, "name", str(state)).upper()

            if state_name == "ACTIVE":
                break

            if state_name == "FAILED":
                raise RuntimeError(
                    "Gemini no pudo procesar el archivo de video"
                )

            if time.monotonic() - start_wait > 600:
                raise RuntimeError(
                    "Gemini tardó demasiado procesando el video"
                )

            time.sleep(5)

            uploaded_file = client.files.get(
                name=uploaded_file.name
            )

        if mode == "automatic":
            category_instruction = (
                "Selecciona los momentos más interesantes de cualquier tipo."
            )
        else:
            category_instruction = (
                "El usuario quiere clips de esta categoría: "
                + category
                + ". Prioriza momentos relacionados con esa categoría."
            )

        prompt = f"""
Eres el sistema de detección de momentos destacados de ClipMaker AI.

Analiza el video completo, incluyendo imagen, audio, conversaciones,
reacciones, cambios de escena y acontecimientos importantes.

DURACIÓN TOTAL DEL VIDEO: {video_duration:.2f} segundos.

CANTIDAD SOLICITADA: {quantity} clips.

DURACIÓN EXACTA DE CADA CLIP: {clip_duration} segundos.

MODO: {mode}.

INSTRUCCIONES DE SELECCIÓN:
{category_instruction}

Busca momentos que tengan potencial para convertirse en videos cortos:
- Frases llamativas.
- Reacciones espontáneas.
- Momentos graciosos.
- Información sorprendente.
- Conversaciones interesantes.
- Momentos de tensión o emoción.
- Acciones importantes.

No inventes conversaciones ni acontecimientos.

Debes encontrar exactamente {quantity} momentos diferentes.

Los clips no pueden superponerse entre sí.

Cada momento debe tener suficiente contenido interesante durante
los {clip_duration} segundos completos.

El inicio debe estar entre 0 y {video_duration - clip_duration:.2f} segundos.

Devuelve exclusivamente JSON válido con esta estructura:

{{
  "clips": [
    {{
      "start_seconds": 12.5,
      "title": "Título corto del momento",
      "reason": "Explicación breve de por qué es interesante",
      "score": 95
    }}
  ]
}}

REGLAS:
- Devuelve exactamente {quantity} elementos.
- start_seconds debe ser un número.
- score debe ser un número entero entre 1 y 100.
- Ordena los momentos por puntuación de mayor a menor.
- No agregues explicaciones fuera del JSON.
"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                uploaded_file,
                prompt
            ],
            config={
                "response_mime_type": "application/json"
            }
        )

        if not response.text:
            raise RuntimeError(
                "Gemini no devolvió resultados"
            )

        raw_text = response.text.strip()

        # Limpiar posibles bloques Markdown
        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text
        )

        raw_text = re.sub(
            r"\s*```$",
            "",
            raw_text
        )

        result = json.loads(raw_text)

        clips = result.get("clips", [])

        if not isinstance(clips, list):
            raise RuntimeError(
                "La respuesta de Gemini no tiene una lista de clips válida"
            )

        if len(clips) < quantity:
            raise RuntimeError(
                "Gemini encontró menos momentos de los solicitados. "
                "Prueba con otro video o una cantidad menor."
            )

        # Validar y ordenar los momentos
        valid_clips = []

        for clip in clips:
            start = float(clip["start_seconds"])

            if start < 0:
                continue

            if start + clip_duration > video_duration:
                continue

            valid_clips.append({
                "start_seconds": start,
                "title": str(clip.get("title", "Momento destacado"))[:100],
                "reason": str(clip.get("reason", ""))[:300],
                "score": max(
                    1,
                    min(100, int(clip.get("score", 50)))
                )
            })

        # Primero los momentos con mayor puntuación
        valid_clips.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Evitar clips superpuestos
        selected = []

        for candidate in valid_clips:
            candidate_start = candidate["start_seconds"]
            candidate_end = candidate_start + clip_duration

            overlaps = any(
                candidate_start < item["start_seconds"] + clip_duration
                and candidate_end > item["start_seconds"]
                for item in selected
            )

            if not overlaps:
                selected.append(candidate)

            if len(selected) == quantity:
                break

        if len(selected) < quantity:
            raise RuntimeError(
                "La IA no encontró suficientes momentos diferentes "
                "sin superposición. Prueba con menos clips."
            )

        # Orden cronológico para facilitar la revisión
        selected.sort(
            key=lambda item: item["start_seconds"]
        )

        return selected

    finally:
        # Intentar borrar el archivo remoto de Gemini
        if uploaded_file is not None:
            try:
                client.files.delete(
                    name=uploaded_file.name
                )
            except Exception:
                pass


# ----------------------------------
# PROCESAMIENTO ORIGINAL
# ----------------------------------

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    start: float = 0,
    duration: float = 15
):
    if start < 0 or duration <= 0 or duration > 60:
        raise HTTPException(
            status_code=400,
            detail="Tiempo de recorte no válido"
        )

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.mp4")
    output_path = os.path.join(temp_dir, "clipmaker.mp4")

    try:
        save_uploaded_video(video, input_path)

        make_clip(
            input_path,
            output_path,
            start,
            duration
        )

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="clipmaker_vertical.mp4"
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo procesar el video: {str(error)}"
        )

    finally:
        await video.close()


# ----------------------------------
# GENERACIÓN MÚLTIPLE ORIGINAL
# ----------------------------------

@app.post("/process-multiple")
async def process_multiple(
    video: UploadFile = File(...),
    quantity: int = 5,
    duration: int = 30
):
    if quantity not in [5, 10, 15]:
        raise HTTPException(
            status_code=400,
            detail="La cantidad debe ser 5, 10 o 15 clips"
        )

    if duration not in [15, 30, 45, 60]:
        raise HTTPException(
            status_code=400,
            detail="La duración debe ser 15, 30, 45 o 60 segundos"
        )

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "original_video.mp4"

    try:
        save_uploaded_video(video, input_path)

        video_duration = get_video_duration(input_path)

        if video_duration < duration:
            raise HTTPException(
                status_code=400,
                detail="El video original es más corto que la duración elegida"
            )

        max_start = video_duration - duration
        clips = []

        for index in range(quantity):
            start = max_start * index / (quantity - 1)

            filename = f"clip_{index + 1:02d}.mp4"
            output_path = job_dir / filename

            make_clip(
                input_path,
                output_path,
                start,
                duration
            )

            clips.append({
                "number": index + 1,
                "filename": filename,
                "duration": duration,
                "url": f"/clips/{job_id}/{filename}"
            })

        return {
            "status": "completed",
            "job_id": job_id,
            "total_clips": len(clips),
            "duration": duration,
            "clips": clips
        }

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    except Exception as error:
        shutil.rmtree(job_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"Error generando los clips: {str(error)}"
        )

    finally:
        await video.close()


# ----------------------------------
# NUEVO: GENERACIÓN CON INTELIGENCIA ARTIFICIAL
# ----------------------------------

@app.post("/process-ai")
def process_ai(
    video: UploadFile = File(...),
    quantity: int = 5,
    duration: int = 30,
    mode: str = "automatic",
    category: str = ""
):
    if quantity not in [5, 10, 15]:
        raise HTTPException(
            status_code=400,
            detail="La cantidad debe ser 5, 10 o 15 clips"
        )

    if duration not in [15, 30, 45, 60]:
        raise HTTPException(
            status_code=400,
            detail="La duración debe ser 15, 30, 45 o 60 segundos"
        )

    if mode not in ["automatic", "custom"]:
        raise HTTPException(
            status_code=400,
            detail="El modo debe ser automatic o custom"
        )

    if mode == "custom" and not category.strip():
        raise HTTPException(
            status_code=400,
            detail="Debes indicar una categoría para el modo personalizado"
        )

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "original_video.mp4"

    try:
        save_uploaded_video(video, input_path)

        video_duration = get_video_duration(input_path)

        if video_duration < quantity * duration:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Para crear {quantity} clips diferentes de "
                    f"{duration} segundos, utiliza un video de al menos "
                    f"{quantity * duration} segundos."
                )
            )

        # Analizar el video con Gemini
        selected_clips = analyze_video_with_gemini(
            input_path,
            video_duration,
            quantity,
            duration,
            mode,
            category
        )

        generated_clips = []
        supabase = get_supabase_client()

        # Crear cada MP4 con FFmpeg y guardarlo de forma permanente en Supabase
        for index, moment in enumerate(selected_clips):
            filename = f"clip_{index + 1:02d}.mp4"
            output_path = job_dir / filename

            start = moment["start_seconds"]

            make_clip(
                input_path,
                output_path,
                start,
                duration
            )

            public_url = upload_clip_to_supabase(
                supabase,
                output_path,
                job_id,
                filename
            )

            generated_clips.append({
                "number": index + 1,
                "filename": filename,
                "duration": duration,
                "start_seconds": start,
                "title": moment["title"],
                "reason": moment["reason"],
                "score": moment["score"],
                "url": public_url
            })

        return {
            "status": "completed",
            "engine": "ClipMaker AI",
            "ai_provider": "Google Gemini",
            "mode": mode,
            "category": category if mode == "custom" else None,
            "job_id": job_id,
            "total_clips": len(generated_clips),
            "duration": duration,
            "clips": generated_clips
        }

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    except Exception as error:
        shutil.rmtree(job_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"Error en el procesamiento inteligente: {str(error)}"
        )

    finally:
        video.file.close()
