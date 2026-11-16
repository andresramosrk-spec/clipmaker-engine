import os
import uuid
import shutil
import subprocess
import tempfile

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="ClipMaker Engine",
    description="Motor rápido de generación de clips verticales sin IA"
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

ALLOWED_QUANTITIES = [5, 10, 15, 30, 50]
ALLOWED_DURATIONS = [15, 30, 45, 60]
MAX_UPLOAD_BYTES = 300 * 1024 * 1024


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
        "message": "ClipMaker Engine funcionando",
        "version": "4.0",
        "ai": False,
        "features": [
            "single_clip",
            "multiple_clips",
            "vertical_1080x1920",
            "no_overlapping_segments",
            "5_10_15_30_50_clips"
        ]
    }


@app.get("/health")
def health():
    return {"status": "online"}


@app.get("/ai-health")
def ai_health():
    return {
        "status": "disabled",
        "message": "La generación actual funciona sin inteligencia artificial."
    }


# ----------------------------------
# FUNCIONES DE VIDEO
# ----------------------------------

def get_video_duration(video_path: Path) -> float:
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
            detail="No se pudo leer la duración del video. Comprueba que sea un archivo de video válido."
        )

    try:
        duration = float(result.stdout.strip())
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="La duración del video no es válida."
        )

    if duration <= 0:
        raise HTTPException(
            status_code=400,
            detail="El video no tiene una duración válida."
        )

    return duration


def make_clip(input_path: Path, output_path: Path, start: float, duration: float):
    """
    Genera un clip vertical 1080x1920.
    Se usa ultrafast para priorizar velocidad de procesamiento.
    """

    video_filter = (
        "scale=1080:1920:"
        "force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "setsar=1"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_message = (result.stderr or "Error desconocido de FFmpeg")[-1500:]
        raise RuntimeError(
            "FFmpeg no pudo generar el clip: " + error_message
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg terminó sin generar un archivo válido.")


def save_uploaded_video(video: UploadFile, destination: Path):
    total_size = 0

    try:
        with open(destination, "wb") as output_file:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="El video supera el límite de 300 MB para esta versión."
                    )

                output_file.write(chunk)

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    if total_size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="El archivo de video está vacío."
        )


def validate_multiple_options(quantity: int, duration: int):
    if quantity not in ALLOWED_QUANTITIES:
        raise HTTPException(
            status_code=400,
            detail="La cantidad debe ser 5, 10, 15, 30 o 50 clips."
        )

    if duration not in ALLOWED_DURATIONS:
        raise HTTPException(
            status_code=400,
            detail="La duración debe ser 15, 30, 45 o 60 segundos."
        )


def calculate_clip_segments(
    video_duration: float,
    quantity: int,
    clip_duration: int
):
    """
    Distribuye los segmentos a lo largo del video.
    Cada segmento tiene la duración elegida y no se superpone
    con el anterior. Si sobra tiempo, se reparte como separación
    entre los clips para cubrir el video completo.
    """

    required_seconds = quantity * clip_duration

    if video_duration + 0.05 < required_seconds:
        minimum_minutes = required_seconds / 60
        raise HTTPException(
            status_code=400,
            detail=(
                f"El video es demasiado corto. Para crear {quantity} clips "
                f"de {clip_duration} segundos necesitas al menos "
                f"{required_seconds} segundos "
                f"({minimum_minutes:.1f} minutos) de video."
            )
        )

    available_gap_time = max(0.0, video_duration - required_seconds)

    if quantity > 1:
        gap_between_clips = available_gap_time / (quantity - 1)
    else:
        gap_between_clips = 0.0

    segments = []

    for index in range(quantity):
        start = index * (clip_duration + gap_between_clips)

        # Evitar que errores de redondeo hagan que el último clip
        # se salga del final del video.
        latest_start = max(0.0, video_duration - clip_duration)
        start = min(start, latest_start)

        segments.append({
            "number": index + 1,
            "start": round(start, 3),
            "end": round(start + clip_duration, 3),
            "duration": clip_duration
        })

    return segments


def create_job_directory():
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


# ----------------------------------
# PROCESAMIENTO DE UN SOLO CLIP
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
            detail="El inicio debe ser positivo y la duración debe estar entre 1 y 60 segundos."
        )

    job_dir = Path(tempfile.mkdtemp(prefix="clipmaker_single_"))
    input_path = job_dir / "original_video"
    output_path = job_dir / "clipmaker_vertical.mp4"

    try:
        save_uploaded_video(video, input_path)
        video_duration = get_video_duration(input_path)

        if start + duration > video_duration:
            raise HTTPException(
                status_code=400,
                detail="El recorte solicitado supera la duración del video original."
            )

        make_clip(input_path, output_path, start, duration)

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="clipmaker_vertical.mp4",
            background=None
        )

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    except Exception as error:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo procesar el video: {str(error)}"
        )

    finally:
        await video.close()


# ----------------------------------
# GENERACIÓN MÚLTIPLE SIN IA
# ----------------------------------

@app.post("/process-multiple")
async def process_multiple(
    video: UploadFile = File(...),
    quantity: int = 5,
    duration: int = 30
):
    validate_multiple_options(quantity, duration)

    job_id, job_dir = create_job_directory()
    input_path = job_dir / "original_video.mp4"

    try:
        save_uploaded_video(video, input_path)

        video_duration = get_video_duration(input_path)

        segments = calculate_clip_segments(
            video_duration,
            quantity,
            duration
        )

        clips = []

        for segment in segments:
            index = segment["number"]
            filename = f"clip_{index:02d}.mp4"
            output_path = job_dir / filename

            make_clip(
                input_path,
                output_path,
                segment["start"],
                duration
            )

            clips.append({
                "number": index,
                "filename": filename,
                "duration": duration,
                "start_seconds": segment["start"],
                "end_seconds": segment["end"],
                "url": f"/clips/{job_id}/{filename}"
            })

        return {
            "status": "completed",
            "engine": "ClipMaker Engine",
            "ai_provider": None,
            "ai_used": False,
            "selection_mode": "distributed_non_overlapping_segments",
            "job_id": job_id,
            "source_duration": round(video_duration, 2),
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
# COMPATIBILIDAD TEMPORAL CON EL FRONTEND
# ----------------------------------

@app.post("/process-ai")
async def process_ai(
    video: UploadFile = File(...),
    quantity: int = 5,
    duration: int = 30,
    mode: str = "automatic",
    category: str = ""
):
    """
    Compatibilidad temporal: si una versión antigua del frontend
    todavía llama a /process-ai, genera clips distribuidos sin IA.
    """

    validate_multiple_options(quantity, duration)

    job_id, job_dir = create_job_directory()
    input_path = job_dir / "original_video.mp4"

    try:
        save_uploaded_video(video, input_path)

        video_duration = get_video_duration(input_path)

        segments = calculate_clip_segments(
            video_duration,
            quantity,
            duration
        )

        clips = []

        for segment in segments:
            index = segment["number"]
            filename = f"clip_{index:02d}.mp4"
            output_path = job_dir / filename

            make_clip(
                input_path,
                output_path,
                segment["start"],
                duration
            )

            clips.append({
                "number": index,
                "filename": filename,
                "duration": duration,
                "start_seconds": segment["start"],
                "end_seconds": segment["end"],
                "title": f"Clip {index}",
                "reason": "Fragmento seleccionado automáticamente por distribución temporal, sin IA.",
                "score": None,
                "url": f"/clips/{job_id}/{filename}"
            })

        return {
            "status": "completed",
            "engine": "ClipMaker Engine",
            "ai_provider": None,
            "ai_used": False,
            "message": "La generación se realizó sin inteligencia artificial.",
            "mode": mode,
            "category": category if mode == "custom" else None,
            "job_id": job_id,
            "source_duration": round(video_duration, 2),
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
