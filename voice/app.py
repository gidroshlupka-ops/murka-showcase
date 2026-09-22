"""RVC voice API: expressive TTS + voice-to-voice with auto pitch.

Weights are not in this repo. Place your own `.pth` / `.index` under `weights/`
and set MODEL_PATH / INDEX_PATH (see .env.example).
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import subprocess
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="RVC Voice API")
_infer_lock = asyncio.Lock()
_rvc = None

MODEL_PATH = os.environ.get("MODEL_PATH", "weights/model.pth")
INDEX_PATH = os.environ.get("INDEX_PATH", "weights/model.index")

RVC_F0_METHOD = os.environ.get("RVC_F0_METHOD", "rmvpe")
RVC_PITCH = int(os.environ.get("RVC_PITCH", "6"))
RVC_INDEX_RATE = float(os.environ.get("RVC_INDEX_RATE", "0.72"))
RVC_PROTECT = float(os.environ.get("RVC_PROTECT", "0.35"))
RVC_TARGET_F0 = float(os.environ.get("RVC_TARGET_F0", "220"))
RVC_F0_FEMALE = float(os.environ.get("RVC_F0_FEMALE", "175"))
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_rvc():
    """Lazy-load so the module can be imported without GPU/weights."""
    global _rvc
    if _rvc is not None:
        return _rvc
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"RVC model not found at {MODEL_PATH}. "
            "Drop your own weights into weights/ — they are not shipped here."
        )
    import torch.serialization
    from rvc_python.infer import RVCInference

    torch.serialization.weights_only = False
    device = _device()
    print("Device:", device)
    _rvc = RVCInference(
        model_path=MODEL_PATH,
        index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
        device=device,
    )
    print(f"Model loaded. f0={RVC_F0_METHOD} pitch={RVC_PITCH} index={RVC_INDEX_RATE}")
    _set_rvc(0)
    return _rvc


def _cleanup(*paths: str) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _set_rvc(pitch: int) -> None:
    """f0up_key=0 keeps source melody; timbre comes from the model, not pitch."""
    rvc = _get_rvc()
    pitch = int(pitch)
    params = dict(
        f0method=RVC_F0_METHOD,
        f0up_key=pitch,
        index_rate=RVC_INDEX_RATE,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=1.0,
        protect=RVC_PROTECT,
    )
    setter = getattr(rvc, "set_params", None)
    if setter:
        try:
            setter(**params)
        except TypeError:
            try:
                setter(f0method=RVC_F0_METHOD, f0up_key=pitch)
            except Exception as e:
                print("set_params skipped:", e)
    for k, v in params.items():
        if hasattr(rvc, k):
            try:
                setattr(rvc, k, v)
            except Exception:
                pass


def _ffmpeg(src: str, dst: str, af: str | None = None, ar: str = "44100") -> None:
    cmd = ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", ar]
    if af:
        cmd += ["-af", af]
    cmd.append(dst)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        err = (proc.stderr or b"")[-400:].decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed: {err}")


def _light_denoise_in(src: str, dst: str) -> None:
    """Input denoise without loudness matching (that flattens intonation)."""
    filters = [
        "highpass=f=80,lowpass=f=12000,afftdn=nr=8:nf=-25",
        "highpass=f=80,anlmdn=s=0.0008:p=0.0015",
        "highpass=f=80,aresample=44100",
    ]
    last = None
    for af in filters:
        try:
            _ffmpeg(src, dst, af=af, ar="44100")
            return
        except Exception as e:
            last = e
    raise last or RuntimeError("denoise failed")


def _median_f0_hz(path: str) -> float | None:
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(path, sr=16000, mono=True)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
        voiced = f0[~np.isnan(f0)]
        if len(voiced) < 8:
            return None
        return float(np.median(voiced))
    except Exception as e:
        print("f0 estimate fail:", e)
        return None


def _auto_pitch(f0: float | None) -> int:
    """Low / male F0 → raise semitones. Already-female F0 → 0."""
    if f0 is None or f0 < 60:
        return 12
    if f0 >= RVC_F0_FEMALE:
        return 0
    semis = int(round(12 * math.log2(RVC_TARGET_F0 / max(f0, 70.0))))
    return max(4, min(12, semis))


def _rvc_infer(inp: str, outp: str, pitch: int = 0) -> None:
    rvc = _get_rvc()
    _set_rvc(pitch)
    try:
        rvc.infer_file(input_path=inp, output_path=outp)
    except TypeError:
        rvc.infer_file(inp, outp)
    if not os.path.exists(outp) or os.path.getsize(outp) < 1000:
        raise RuntimeError("RVC output not created")


def _prep_speech(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("…", ". ").replace("...", ". ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t[:2000]


def _tts_style(phrase: str) -> tuple[str, str]:
    p = phrase.strip()
    if p.endswith("?"):
        return "+6%", "+10Hz"
    if p.endswith("!"):
        return "+8%", "+5Hz"
    if len(p) > 90:
        return "-12%", "+3Hz"
    return "-7%", "+4Hz"


def _split_utterances(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = [p.strip() for p in parts if p.strip()]
    return out or [text.strip()]


async def _edge_expressive(text: str, out_mp3: str) -> None:
    import edge_tts

    voice = EDGE_TTS_VOICE
    parts = _split_utterances(text)
    if len(parts) == 1:
        rate, pitch = _tts_style(parts[0])
        await edge_tts.Communicate(parts[0], voice, rate=rate, pitch=pitch).save(out_mp3)
        return

    tmp_files: list[str] = []
    list_file = out_mp3 + ".txt"
    try:
        for i, part in enumerate(parts):
            tmp = f"{out_mp3}.{i}.mp3"
            rate, pitch = _tts_style(part)
            await edge_tts.Communicate(part, voice, rate=rate, pitch=pitch).save(tmp)
            tmp_files.append(tmp)
        with open(list_file, "w", encoding="utf-8") as f:
            for p in tmp_files:
                f.write(f"file '{os.path.abspath(p)}'\n")
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", out_mp3,
            ],
            capture_output=True,
        )
        if proc.returncode != 0 or not os.path.exists(out_mp3):
            rate, pitch = _tts_style(text)
            await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out_mp3)
    finally:
        _cleanup(list_file, *tmp_files)


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
async def tts(req: TTSRequest):
    text = _prep_speech(req.text or "")
    if not text:
        raise HTTPException(400, "text required")

    task_id = str(uuid.uuid4())
    base_file = f"{task_id}_base.mp3"
    rvc_file = f"{task_id}_out.wav"
    try:
        await _edge_expressive(text, base_file)
        async with _infer_lock:
            await asyncio.to_thread(_rvc_infer, base_file, rvc_file, RVC_PITCH)
        with open(rvc_file, "rb") as f:
            audio = f.read()
        return Response(content=audio, media_type="audio/wav")
    except Exception as e:
        print("tts error:", e)
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        _cleanup(base_file, rvc_file)


@app.post("/convert")
@app.post("/voice2voice")
async def convert(
    file: UploadFile = File(...),
    pitch: str = Form("-999"),
):
    """Voice conversion. pitch=auto/-999 → estimate F0; a number is manual semitones."""
    raw = await file.read()
    if not raw or len(raw) < 1000:
        raise HTTPException(400, "empty audio")
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(400, "file too large")

    task_id = str(uuid.uuid4())
    src_file = f"{task_id}_src"
    clean_file = f"{task_id}_clean.wav"
    rvc_file = f"{task_id}_out.wav"
    try:
        with open(src_file, "wb") as f:
            f.write(raw)
        await asyncio.to_thread(_light_denoise_in, src_file, clean_file)
        raw_pitch = str(pitch or "").strip().lower()
        auto = raw_pitch in ("", "auto", "-999", "none")
        f0 = None
        if auto:
            f0 = await asyncio.to_thread(_median_f0_hz, clean_file)
            used = _auto_pitch(f0)
        else:
            used = max(-12, min(12, int(float(raw_pitch))))
        print(f"v2v auto={auto} f0={f0} pitch={used}")
        async with _infer_lock:
            await asyncio.to_thread(_rvc_infer, clean_file, rvc_file, used)
        if not os.path.exists(rvc_file) or os.path.getsize(rvc_file) < 1000:
            raise RuntimeError("RVC output not created")
        with open(rvc_file, "rb") as f:
            audio = f.read()
        headers = {
            "X-RVC-Pitch": str(used),
            "X-RVC-Auto": "1" if auto else "0",
        }
        if f0:
            headers["X-RVC-F0"] = f"{f0:.1f}"
        return Response(content=audio, media_type="audio/wav", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        print("convert error:", e)
        raise HTTPException(500, f"convert failed: {e}")
    finally:
        _cleanup(src_file, clean_file, rvc_file)


@app.get("/")
async def root():
    return {
        "status": "ready" if os.path.exists(MODEL_PATH) else "weights_missing",
        "device": _device(),
        "model": MODEL_PATH,
        "f0method": RVC_F0_METHOD,
        "pitch": RVC_PITCH,
        "index_rate": RVC_INDEX_RATE,
        "tts_voice": EDGE_TTS_VOICE,
    }
