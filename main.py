
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
    title="ClipMaker AI",
    description="Motor de procesamiento de videos"
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

app.mount(
    "/clips",
    StaticFiles(directory=str(JOBS_DIR)),
    name="clips"
)


@app.get("/")
def home():
    return {
        "message": "ClipMaker AI Engine funcionando",
        "version": "2.0"
    }


@app.get("/health")
def health():
    return {"status": "online"}


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
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
        "-i", input_path,
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
        raise RuntimeError("FFmpeg no pudo generar el clip")


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
        with open(input_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                f.write(chunk)

        make_clip(input_path, output_path, start, duration)

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="clipmaker_vertical.mp4"
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No se pudo procesar el video"
        )
    finally:
        await video.close()


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

    input_path = job_dir / "original_video"

    try:
        with open(input_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                f.write(chunk)

        video_duration = get_video_duration(str(input_path))

        if video_duration < duration:
            raise HTTPException(
                status_code=400,
                detail="El video original es más corto que la duración elegida"
            )

        max_start = max(0, video_duration - duration)

        clips = []

        for index in range(quantity):
            if quantity == 1:
                start = 0
            else:
                start = max_start * index / (quantity - 1)

            filename = f"clip_{index + 1:02d}.mp4"
            output_path = job_dir / filename

            make_clip(
                str(input_path),
                output_path,
                start,
                duration
            )

            clips.append({
                "number": index + 1,
                "filename": filename,
                "duration": duration,
                "url": f"{str(video_path := '/clips/' + job_id + '/' + filename)}"
            })

        base_url = str(video_url_base := "")

        return {
            "status": "completed",
            "job_id": job_id,
            "total_clips": len(clips),
            "duration": duration,
            "clips": [
                {
                    **clip,
                    "url": f"/clips/{job_id}/{clip['filename']}"
                }
                for clip in clips
            ]
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
