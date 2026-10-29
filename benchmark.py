from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np
import soundfile as sf
import librosa
from pesq import pesq
from pystoi import stoi
import mir_eval
import io
from typing import Tuple

app = FastAPI(title="CalmWave Audio Benchmark API")

MAX_FILE_BYTES = 10 * 1024 * 1024
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = 4000


def read_audio_bytes(content: bytes) -> Tuple[np.ndarray, int]:
    """Decode an uploaded audio file and return samples and sample rate."""
    data, sr = sf.read(io.BytesIO(content))
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("Invalid sample rate")
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Audio is empty or contains invalid numeric values")
    return data, int(sr)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert [samples, channels] audio to mono before resampling."""
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def safe_round(value):
    """Return a finite metric rounded to three decimals, or None."""
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return round(numeric, 3)


def align_and_resample(
    ref_audio: np.ndarray,
    deg_audio: np.ndarray,
    sr_ref: int,
    sr_deg: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray, int]:
    # Convert to mono first: soundfile returns multichannel audio as [samples, channels].
    # librosa.resample operates on the last axis by default, so resampling stereo first
    # would otherwise resample across channels rather than across time.
    ref_audio = to_mono(np.asarray(ref_audio, dtype=np.float64))
    deg_audio = to_mono(np.asarray(deg_audio, dtype=np.float64))

    if sr_ref <= 0 or sr_deg <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")

    if sr_ref != target_sr:
        ref_audio = librosa.resample(
            ref_audio, orig_sr=sr_ref, target_sr=target_sr
        )
    if sr_deg != target_sr:
        deg_audio = librosa.resample(
            deg_audio, orig_sr=sr_deg, target_sr=target_sr
        )

    # Compare the common interval after both signals have the same sample rate.
    min_len = min(len(ref_audio), len(deg_audio))
    ref_audio = ref_audio[:min_len]
    deg_audio = deg_audio[:min_len]
    return ref_audio, deg_audio, target_sr


@app.post("/benchmark")
async def benchmark(
    clean_file: UploadFile = File(...),
    denoised_file: UploadFile = File(...),
):
    try:
        # Read at most MAX_FILE_BYTES + 1 to reject large uploads without
        # loading an arbitrarily large file fully into memory.
        clean_content = await clean_file.read(MAX_FILE_BYTES + 1)
        denoised_content = await denoised_file.read(MAX_FILE_BYTES + 1)

        if len(clean_content) > MAX_FILE_BYTES or len(denoised_content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        try:
            ref, sr_ref = read_audio_bytes(clean_content)
            deg, sr_deg = read_audio_bytes(denoised_content)
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Não foi possível ler um dos arquivos como áudio válido.",
            ) from exc

        ref_proc, deg_proc, final_sr = align_and_resample(
            ref, deg, sr_ref, sr_deg, target_sr=TARGET_SAMPLE_RATE
        )

        if len(ref_proc) < MIN_AUDIO_SAMPLES:
            raise HTTPException(
                status_code=400,
                detail="Áudio muito curto para uma avaliação confiável; envie ao menos 0,25 s.",
            )

        # Some metrics can reject particular signals. Report unavailable metrics as null.
        try:
            score_pesq = pesq(final_sr, ref_proc, deg_proc, "wb")
        except Exception:
            score_pesq = None

        try:
            score_stoi = stoi(ref_proc, deg_proc, final_sr, extended=False)
        except Exception:
            score_stoi = None

        try:
            sdr_arr, _, _, _ = mir_eval.separation.bss_eval_sources(
                ref_proc.reshape(1, -1),
                deg_proc.reshape(1, -1),
                compute_permutation=False,
            )
            sdr_db = float(sdr_arr[0])
        except Exception:
            sdr_db = None

        return {
            "pesq": safe_round(score_pesq),
            "stoi": safe_round(score_stoi),
            "sdr": safe_round(sdr_db),
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception:
        # Avoid exposing internal exception details in API responses.
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao calcular as métricas de áudio.",
        )


@app.get("/test")
async def test():
    return {
        "status": "ok",
        "message": "Rota de teste funcionando",
        "api": "CalmWave Audio Benchmark API",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
