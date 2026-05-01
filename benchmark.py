from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np
import soundfile as sf
import librosa
from pesq import pesq
from pystoi import stoi
import mir_eval
import io
import os
from typing import Tuple

app = FastAPI(title="CalmWave Audio Benchmark API")

def read_audio_bytes(content: bytes) -> Tuple[np.ndarray, int]:
    # sf.read aceita file-like
    data, sr = sf.read(io.BytesIO(content))
    return np.asarray(data), int(sr)

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        return np.mean(x, axis=1)
    return x

def align_and_resample(ref_audio, deg_audio, sr_ref, sr_deg, target_sr=16000):
    # Garante float64 (librosa espera floats)
    ref_audio = np.asarray(ref_audio, dtype=np.float64)
    deg_audio = np.asarray(deg_audio, dtype=np.float64)

    # Resample se necessário
    if sr_ref != target_sr:
        ref_audio = librosa.resample(ref_audio, orig_sr=sr_ref, target_sr=target_sr)
    if sr_deg != target_sr:
        deg_audio = librosa.resample(deg_audio, orig_sr=sr_deg, target_sr=target_sr)

    # Mono
    ref_audio = to_mono(ref_audio)
    deg_audio = to_mono(deg_audio)

    # Align by shortest
    min_len = min(len(ref_audio), len(deg_audio))
    ref_audio = ref_audio[:min_len]
    deg_audio = deg_audio[:min_len]

    return ref_audio, deg_audio, target_sr

@app.post("/benchmark")
async def benchmark(clean_file: UploadFile = File(...), denoised_file: UploadFile = File(...)):
    try:
        # leitura
        clean_content = await clean_file.read()
        denoised_content = await denoised_file.read()

        # limites simples (por segurança)
        max_bytes = 10 * 1024 * 1024  # 10 MB
        if len(clean_content) > max_bytes or len(denoised_content) > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")

        ref, sr_ref = read_audio_bytes(clean_content)
        deg, sr_deg = read_audio_bytes(denoised_content)

        # garantir arrays válidos
        if ref.size < 1600 or deg.size < 1600:
            raise HTTPException(status_code=400, detail="Audio muito curto para análise")

        # Resample/alinhamento (uso 16k por padrão para PESQ wideband)
        ref_proc, deg_proc, final_sr = align_and_resample(ref, deg, sr_ref, sr_deg, target_sr=16000)

        # PESQ (pode lançar erro se inputs inválidos)
        try:
            score_pesq = pesq(final_sr, ref_proc, deg_proc, 'wb')
        except Exception as e:
            # fallback ou informar que PESQ falhou
            score_pesq = None

        # STOI
        try:
            score_stoi = stoi(ref_proc, deg_proc, final_sr, extended=False)
        except Exception:
            score_stoi = None

        # SDR (em dB) usando mir_eval
        try:
            # mir_eval espera shape (nsrc, nsamples)
            sdr_arr, _, _, _ = mir_eval.separation.bss_eval_sources(
                ref_proc.reshape(1, -1),
                deg_proc.reshape(1, -1),
                compute_permutation=False
            )
            sdr_db = float(sdr_arr[0])
        except Exception:
            sdr_db = None

        # Monta resposta com chaves simples no root (facilita app)
        result = {
            "pesq": round(float(score_pesq), 3) if score_pesq is not None else None,
            "stoi": round(float(score_stoi), 3) if score_stoi is not None else None,
            "sdr": round(float(sdr_db), 3) if sdr_db is not None else None,
            "status": "success"
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        # log simples
        print("Erro: ", str(e))
        raise HTTPException(status_code=500, detail="Erro interno ao calcular métricas: " + str(e))

@app.get('/test')
async def test():
    return {
        "status": "ok",
        "message": "Rota de teste funcionando",
        "api": "CalmWave Audio Benchmark API"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)