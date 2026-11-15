
import os
import subprocess
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ClipMaker AI",
    description="Motor de procesamiento de videos"
)

# Permitir conexiones desde la aplicación web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "ClipMaker AI Engine funcionando",
        "version": "1.0"
    }

@app.get("/health")
def health():
    return {"status": "online"}

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    start: float = 0,
    duration: float = 15
):
    if start < 0 or duration <= 0 or duration > 120:
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
            output_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Error procesando el video"
            )

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
