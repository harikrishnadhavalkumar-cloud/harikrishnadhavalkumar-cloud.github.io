"""Makes a clean version of a meeting's sound: no echo, less background noise, even volume.

The microphone often picks up the other people from the speakers a moment late. That is the echo.
While the other side is talking we turn the microphone down, so only one copy of their voice remains.
"""
import subprocess
from pathlib import Path


def _ff(*args):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], capture_output=True, text=True)
    return r.returncode == 0


def make_clean(d):
    d = Path(d)
    mic, oth, vid = d / "mic.webm", d / "others.webm", d / "video_fixed.webm"
    has_m = mic.exists() and mic.stat().st_size > 2000
    has_o = oth.exists() and oth.stat().st_size > 2000
    if not (has_m or has_o):
        return False
    out = d / "clean.opus"
    tidy = "dynaudnorm=f=200:g=5,alimiter=limit=0.9"
    if has_m and has_o:
        fc = (
            "[0:a]highpass=f=90,afftdn=nf=-28[m];[1:a]highpass=f=60,asplit=2[o1][o2];"
            "[m][o1]sidechaincompress=threshold=0.015:ratio=20:attack=5:release=350:makeup=1[md];"
            "[md][o2]amix=inputs=2:duration=longest:normalize=0," + tidy + "[out]"
        )
        ok = _ff("-i", mic, "-i", oth, "-filter_complex", fc, "-map", "[out]", "-c:a", "libopus", "-b:a", "48k", out)
    else:
        src = mic if has_m else oth
        ok = _ff("-i", src, "-af", "highpass=f=90,afftdn=nf=-28," + tidy, "-c:a", "libopus", "-b:a", "48k", out)
    if ok and vid.exists():
        final = d / "video_clean.webm"
        _ff("-i", vid, "-i", out, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", final)
        if final.exists() and final.stat().st_size > 2000:
            # keep one video, not three: the cleaned one
            for extra in (vid, d / "video.webm"):
                extra.unlink(missing_ok=True)
    return ok
