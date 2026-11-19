"""Turns a sound file into timed text. Runs on this computer, no account needed."""
import difflib
import os
import re
import threading

_model = None
_lock = threading.Lock()
MODEL_NAME = os.environ.get("MERIDIAN_MODEL", "large-v3")
# "translate" writes English whatever mix of Hindi and English was spoken. Use "transcribe" to keep the original words.
TASK = os.environ.get("MERIDIAN_TASK", "translate")
HINTS = "Univest, PLTV, LTV, ROAS, CAC, Meta, Google Ads, trial, subscription, signup, budget, launch, vendor, invoice."


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        return _model


def transcribe(wav_path, speaker, progress=None):
    """Return (segments, language). Each segment: start, end, speaker, text."""
    model = _get_model()
    segs, info = model.transcribe(
        wav_path,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200, "threshold": 0.35},
        beam_size=2,
        task=TASK,
        initial_prompt=HINTS,
        condition_on_previous_text=False,
    )
    out = []
    total = max(info.duration or 0, 1)
    for s in segs:
        if progress:
            progress(min(1.0, s.end / total))
        text = s.text.strip()
        if text:
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "speaker": speaker, "text": text})
    return _remove_repeats(out), info.language


def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _remove_repeats(segs):
    """Speech recognition sometimes says the same line again and again. Keep it once."""
    out = []
    for s in segs:
        if out:
            p = out[-1]
            a, b = _norm(p["text"]), _norm(s["text"])
            if a and b and (a == b or difflib.SequenceMatcher(None, a, b).ratio() > 0.88) and s["start"] - p["end"] < 15:
                p["end"] = max(p["end"], s["end"])
                continue
        out.append(s)
    return out
