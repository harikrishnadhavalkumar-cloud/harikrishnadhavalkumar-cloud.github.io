"""Meridian: records a meeting in the browser, then writes the transcript and notes.

Run:  ./run.sh     then open http://localhost:8790
Everything stays on this computer, in meeting_bot/data/.
"""
import difflib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, redirect, request, send_file, send_from_directory

import clean_audio
import notes
import transcribe

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
KINDS = ("video", "mic", "others")
ID_RE = re.compile(r"^[a-f0-9]{12}$")

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
_job_lock = threading.Lock()  # one transcription at a time keeps the computer calm

# ---------- private link: only people with the key can open Meridian from outside ----------
def _site_key():
    env = os.environ.get("MERIDIAN_KEY")
    if env:
        return env
    kf = DATA / ".key"
    if not kf.exists():
        kf.write_text(secrets.token_urlsafe(18))
    return kf.read_text().strip()


SITE_KEY = _site_key()
# Open link mode: anyone who has the web address can open Meridian. Set MERIDIAN_OPEN=0 to require the secret key again.
SITE_OPEN = os.environ.get("MERIDIAN_OPEN", "1") == "1"
_PROXY_HEADERS = ("Cf-Connecting-Ip", "Cf-Ray", "X-Forwarded-For", "X-Real-Ip", "Forwarded")


def is_local():
    return request.remote_addr in ("127.0.0.1", "::1") and not any(h in request.headers for h in _PROXY_HEADERS)


_OWNER_RE = re.compile(r"^[A-Za-z0-9_-]{6,40}$")


@app.before_request
def gate():
    """Local requests are the host. Everyone else needs the secret key, from the link, a header or a cookie."""
    g.set_cookies = {}
    if is_local():
        g.owner = "host"
        return None
    given = request.args.get("k") or request.headers.get("X-Meridian-Key") or request.cookies.get("mk") or ""
    if not SITE_OPEN and not secrets.compare_digest(given.encode(), SITE_KEY.encode()):
        return Response(
            "This link is missing its secret end part. Please open the FULL link you were sent, "
            "the one that ends with ?k= followed by letters and numbers.", 403)
    owner = request.headers.get("X-Meridian-Owner") or request.args.get("o") or request.cookies.get("mo") or ""
    if not _OWNER_RE.match(owner) or owner == "host":
        owner = secrets.token_hex(8)
    g.owner = owner
    g.set_cookies = {"mo": owner}
    if given and secrets.compare_digest(given.encode(), SITE_KEY.encode()):
        g.set_cookies["mk"] = SITE_KEY
    return None


@app.after_request
def put_cookies(resp):
    for k, v in (getattr(g, "set_cookies", None) or {}).items():
        resp.set_cookie(k, v, max_age=31536000, httponly=True, secure=True, samesite="Lax")
    resp.headers["Cache-Control"] = "no-store"
    if request.path.startswith("/api/"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Meridian-Key, X-Meridian-Owner"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return resp
  # one transcription at a time keeps the computer calm


def mdir(mid):
    if not ID_RE.match(mid):
        abort(404)
    d = DATA / mid
    if not d.is_dir():
        abort(404)
    return d  # everyone with the link sees every meeting


def read_meta(d):
    return json.loads((d / "meta.json").read_text())


def write_meta(d, **kw):
    p = d / "meta.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.update(kw)
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False))
    tmp.replace(p)
    return meta


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    free = shutil.disk_usage(str(DATA)).free
    used = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    if g.owner != "host":
        return jsonify(free_gb=round(free / 1e9, 1), used_mb=0)
    return jsonify(free_gb=round(free / 1e9, 1), used_mb=round(used / 1e6))


@app.get("/api/share")
def share():
    """Only the host (this Mac) sees the link to give to friends."""
    if g.owner != "host":
        abort(404)
    fixed = "https://harikrishnadhavalkumar-cloud.github.io/meridian/"
    kf = DATA / ".inbox_key"
    code = kf.read_text().strip() if kf.exists() else ""
    return jsonify(url=fixed, link=fixed + ("#c=" + code if code else ""))


@app.get("/api/notes/<mid>.txt")
def notes_txt(mid):
    d = mdir(mid)
    meta = read_meta(d)
    nfile = d / "notes.json"
    tfile = d / "transcript.json"
    if not (nfile.exists() and tfile.exists()):
        abort(404)
    n = json.loads(nfile.read_text())
    segs = json.loads(tfile.read_text())["segments"]
    body = build_notes_text(meta, segs, n)
    safe = re.sub(r"[^A-Za-z0-9 ,._-]", "", meta.get("title", "meeting")).strip() or "meeting"
    return Response(body, mimetype="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="%s notes.txt"' % safe})


@app.post("/api/start")
def start():
    body = request.get_json(silent=True) or {}
    mid = uuid.uuid4().hex[:12]
    d = DATA / mid
    d.mkdir()
    write_meta(
        d,
        id=mid,
        title=(body.get("title") or "Untitled meeting")[:120],
        you=(body.get("you") or "You")[:40],
        mode=body.get("mode") if body.get("mode") in ("call", "room") else "call",
        source=str(body.get("source") or "any")[:20],
        created=time.time(),
        owner=g.owner,
        status="recording",
        step="Recording",
    )
    return jsonify(id=mid)


@app.post("/api/chunk/<mid>/<kind>")
def chunk(mid, kind):
    d = mdir(mid)
    if kind not in KINDS:
        abort(400)
    with open(d / (kind + ".webm"), "ab") as f:
        f.write(request.get_data())
    return ("", 204)


@app.post("/api/finish/<mid>")
def finish(mid):
    d = mdir(mid)
    body = request.get_json(silent=True) or {}
    write_meta(d, status="queued", step="Saving the recording", duration=float(body.get("duration") or 0))
    threading.Thread(target=run_pipeline, args=(mid,), daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/reprocess/<mid>")
def reprocess(mid):
    d = mdir(mid)
    write_meta(d, status="queued", step="Starting again")
    threading.Thread(target=run_pipeline, args=(mid, request.args.get("notes_only") == "1"), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/status/<mid>")
def status(mid):
    return jsonify(read_meta(mdir(mid)))


@app.get("/api/voices")
def voices_list():
    import voice
    return jsonify(voice.names())


@app.post("/api/voice/<name>")
def voice_enroll(name):
    import voice
    name = re.sub(r"[^\w .'-]", "", name).strip()[:40]
    if not name:
        abort(400)
    raw = DATA / ("_voice_%s.bin" % uuid.uuid4().hex[:8])
    wav = raw.with_suffix(".wav")
    try:
        raw.write_bytes(request.get_data())
        if not voice.to_wav(raw, wav):
            return jsonify(error="Could not read the recording"), 400
        try:
            clips = voice.enroll(name, wav)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, name=name, clips=clips)
    finally:
        raw.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


@app.delete("/api/voice/<name>")
def voice_delete(name):
    import voice
    voice.delete(name)
    return jsonify(ok=True)


@app.get("/api/meetings")
def meetings():
    out = []
    for d in DATA.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            try:
                m = read_meta(d)
                if True:
                    out.append(m)
            except Exception:
                pass
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return jsonify(out)


@app.get("/api/meeting/<mid>")
def meeting(mid):
    d = mdir(mid)
    out = read_meta(d)
    for name in ("transcript", "notes"):
        p = d / (name + ".json")
        out[name] = json.loads(p.read_text()) if p.exists() else None
    out["has"] = {k: (d / (k + ".webm")).exists() or (d / (k + "_fixed.webm")).exists() for k in KINDS}
    out["has"]["clean"] = (d / "clean.opus").exists()
    return jsonify(out)


@app.get("/api/media/<mid>/<kind>")
def media(mid, kind):
    d = mdir(mid)
    if kind not in KINDS:
        abort(400)
    p = d / (kind + "_clean.webm")
    if not p.exists():
        p = d / (kind + "_fixed.webm")
    if not p.exists():
        p = d / (kind + ".webm")
    if not p.exists():
        abort(404)
    title = re.sub(r"[^A-Za-z0-9 _-]", "", read_meta(d).get("title", "meeting")) or "meeting"
    return send_file(p, mimetype="video/webm" if kind == "video" else "audio/webm", conditional=True,
                     download_name="%s-%s.webm" % (title, kind), as_attachment=request.args.get("download") == "1")


@app.delete("/api/meeting/<mid>")
def delete(mid):
    shutil.rmtree(mdir(mid))
    return jsonify(ok=True)


@app.delete("/api/meeting/<mid>/media")
def delete_media(mid):
    """Free the disk space but keep the transcript and notes."""
    d = mdir(mid)
    freed = 0
    for f in d.iterdir():
        if f.suffix in (".webm", ".wav", ".opus"):
            freed += f.stat().st_size
            f.unlink()
    size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
    write_meta(d, size=size, media_deleted=True)
    return jsonify(freed_mb=round(freed / 1e6), size=size)


@app.patch("/api/meeting/<mid>/actions")
def save_actions(mid):
    d = mdir(mid)
    p = d / "notes.json"
    if not p.exists():
        abort(404)
    n = json.loads(p.read_text())
    n["done"] = [int(x) for x in (request.get_json(silent=True) or {}).get("done", [])]
    p.write_text(json.dumps(n, ensure_ascii=False))
    return jsonify(ok=True)


# ---------- the work after you press Stop ----------

def ffmpeg(*args):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    return r.returncode == 0


def merge(parts):
    """Join the 'You' and 'Others' lines by time. Drop 'You' lines that are just the speakers leaking into the mic."""
    mine = [s for s in parts if s["_kind"] == "mic"]
    others = [s for s in parts if s["_kind"] != "mic"]
    keep = list(others)
    for m in mine:
        echo = False
        for o in others:
            if o["start"] < m["end"] and m["start"] < o["end"]:
                if difflib.SequenceMatcher(None, m["text"].lower(), o["text"].lower()).ratio() > 0.6:
                    echo = True
                    break
        if not echo:
            keep.append(m)
    keep.sort(key=lambda s: s["start"])
    for s in keep:
        s.pop("_kind", None)
    return keep


def notes_dir():
    """Where finished notes are saved as plain text files.
    If Google Drive for desktop is installed, notes go straight into My Drive/Meridian Notes and sync by themselves."""
    env = os.environ.get("MERIDIAN_NOTES_DIR")
    if env:
        return Path(env).expanduser()
    for gd in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*/My Drive")):
        return gd / "Meridian Notes"
    return Path.home() / "Documents" / "Meridian Notes"


def build_notes_text(meta, segments, n):
    def mm(t):
        t = int(t)
        return "%02d:%02d" % (t // 60, t % 60)

    when = time.strftime("%d %b %Y %H:%M", time.localtime(meta.get("created", time.time())))
    dur = int(meta.get("duration") or 0)
    lines = [meta.get("title", "Meeting"), "Recorded by: " + meta.get("you", "You"), "Recorded: " + when,
             "Length: %d min %d s" % (dur // 60, dur % 60), ""]
    lines += ["OVERALL SUMMARY", n.get("overview") or n.get("summary", ""), ""]
    if n.get("main_focus"):
        lines += ["MAIN FOCUS", n["main_focus"], ""]
    if n.get("key_points"):
        lines += ["KEY POINTS"] + ["- " + x for x in n["key_points"]] + [""]
    if n.get("topics"):
        lines += ["TOPICS IN DETAIL"]
        for i, t in enumerate(n["topics"], 1):
            lines += ["%d. %s%s" % (i, t.get("title", ""), (" (%s%% of the talking)" % t["share"]) if t.get("share") is not None else ""), "   " + (t.get("paragraph") or t.get("summary", ""))]
            lines += ["   - " + x for x in t.get("points", [])]
            if t.get("outcome"):
                lines += ["   Outcome: " + t["outcome"]]
            lines += [""]
    if n.get("open_questions"):
        lines += ["OPEN QUESTIONS"] + ["- " + x for x in n["open_questions"]] + [""]
    men = n.get("mentions") or {}
    if men.get("people") or men.get("terms"):
        lines += ["MENTIONED", "People: " + (", ".join(men.get("people", [])) or "none"), "Tools and terms: " + (", ".join(men.get("terms", [])) or "none"), ""]
    for k, v in (n.get("parts") or {}).items():
        lines += ["WHAT %s SAID" % ("THE OTHERS" if k == "others" else v["name"].upper()), v["summary"], ""]
    lines += ["DECISIONS"] + (["- " + x for x in n.get("decisions", [])] or ["- none"])
    lines += ["", "WHO DOES WHAT"] + (
        ["- %s: %s%s" % (a["owner"], a["task"], (" (" + a["due"] + ")") if a.get("due") else "") for a in n.get("actions", [])] or ["- none"]
    )
    lines += ["", "TRANSCRIPT (English)"] + ["[%s] %s: %s" % (mm(s["start"]), s["speaker"], s["text"]) for s in segments]
    return "\n".join(lines)


def export_notes(d, meta, segments, n):
    """Save a text copy of the notes. Each person gets their own folder, named after the name they typed."""
    who = re.sub(r"[^A-Za-z0-9 _-]", "", meta.get("you", "") or "").strip().title() or "Guest"
    if who.lower() == "you":
        who = "Guest"
    out = notes_dir() / who
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9 ,._-]", "", meta.get("title", "Meeting")).strip() or "Meeting"
    stamp = time.strftime("%Y-%m-%d %H.%M", time.localtime(meta.get("created", time.time())))
    path = out / ("%s %s.txt" % (stamp, safe))
    path.write_text(build_notes_text(meta, segments, n), encoding="utf-8")
    return str(path)


def export_brief(d, meta, segments, n):
    """A single HTML file with the designed brief and all its data inside. Opens in any browser, no server needed."""
    tpl = (ROOT / "static" / "brief.html").read_text(encoding="utf-8")
    data = {
        "title": meta.get("title", "Meeting"), "you": meta.get("you", ""), "created": meta.get("created", time.time()),
        "duration": meta.get("duration", 0), "language": meta.get("language"),
        "words": sum(len(x["text"].split()) for x in segments), "notes": n, "transcript": {"segments": segments},
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = tpl.replace("<script>\n(function(){", "<script>window.__MERIDIAN__=" + blob + ";</script>\n<script>\n(function(){", 1)
    who = re.sub(r"[^A-Za-z0-9 _-]", "", meta.get("you", "") or "").strip().title() or "Guest"
    if who.lower() == "you":
        who = "Guest"
    out = notes_dir() / who
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9 ,._-]", "", meta.get("title", "Meeting")).strip() or "Meeting"
    stamp = time.strftime("%Y-%m-%d %H.%M", time.localtime(meta.get("created", time.time())))
    path = out / ("%s %s (brief).html" % (stamp, safe))
    path.write_text(html, encoding="utf-8")
    return str(path)


def export_report(d, meta, segments, n):
    """The executive report (brown deck) as one HTML file with its data inside, plus a PDF when Chrome is installed."""
    tpl = (ROOT / "static" / "report.html").read_text(encoding="utf-8")
    data = {
        "title": meta.get("title", "Meeting"), "you": meta.get("you", ""), "created": meta.get("created", time.time()),
        "duration": meta.get("duration", 0), "language": meta.get("language"),
        "words": sum(len(x["text"].split()) for x in segments), "notes": n, "transcript": {"segments": segments},
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = tpl.replace("<script>\n(function(){", "<script>window.__MERIDIAN__=" + blob + ";</script>\n<script>\n(function(){", 1)
    who = re.sub(r"[^A-Za-z0-9 _-]", "", meta.get("you", "") or "").strip().title() or "Guest"
    if who.lower() == "you":
        who = "Guest"
    out = notes_dir() / who
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9 ,._-]", "", meta.get("title", "Meeting")).strip() or "Meeting"
    stamp = time.strftime("%Y-%m-%d %H.%M", time.localtime(meta.get("created", time.time())))
    path = out / ("%s %s (executive report).html" % (stamp, safe))
    path.write_text(html, encoding="utf-8")
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome.exists():
        try:
            subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                            "--print-to-pdf=" + str(path.with_suffix(".pdf")), "file://" + str(path)],
                           capture_output=True, timeout=90)
        except Exception:
            pass
    return str(path)


def quality_issues(segments, n):
    """The rule: a report must have real written content and a clean transcript."""
    issues = []
    words = sum(len(x["text"].split()) for x in segments)
    if words >= 300:
        if len((n.get("overview") or "").split()) < 80:
            issues.append("the overview is short")
        topics = n.get("topics") or []
        if not topics:
            issues.append("no topics were found")
        elif sum(1 for t in topics if t.get("paragraph")) < max(1, len(topics) // 2):
            issues.append("most topics have no written paragraph")
    if words >= 300 and n.get("engine") == "local-deep":
        secs = n.get("sections") or []
        need = max(2, int(0.7 * (-(-words // 380))))
        if len(secs) < need:
            issues.append("not every part of the meeting has its own section (%d of %d)" % (len(secs), need))
        elif sum(1 for x in secs if len((x.get("summary") or "").split()) >= 20) < 0.8 * len(secs):
            issues.append("many sections are too short")
        if len((n.get("catchup") or "").split()) < 80:
            issues.append("the catch-up briefing is short")
        cov = n.get("coverage") or {}
        if cov.get("terms_total", 0) >= 8 and cov.get("terms_found", 0) < 0.85 * cov["terms_total"]:
            issues.append("some key words from the meeting are missing in the notes (%d of %d)" % (cov.get("terms_found", 0), cov["terms_total"]))
    texts = [s["text"].lower().strip() for s in segments]
    rep = sum(1 for i in range(1, len(texts)) if texts[i] == texts[i - 1])
    if segments and rep / len(segments) > 0.08:
        issues.append("many repeated lines in the transcript")
    return issues


def run_pipeline(mid, notes_only=False):
    d = DATA / mid
    try:
        meta = read_meta(d)
        with _job_lock:
            write_meta(d, status="working", step="Preparing the sound")
            reuse = notes_only and (d / "transcript.json").exists()
            if reuse:
                tj = json.loads((d / "transcript.json").read_text())
                segments, lang = tj["segments"], tj.get("language")
            else:
                if (d / "video.webm").exists():
                    if ffmpeg("-i", str(d / "video.webm"), "-c", "copy", str(d / "video_fixed.webm")):
                        pass
                # which sound files do we have?
                sources = []
                for kind in ("mic", "others"):
                    f = d / (kind + ".webm")
                    if f.exists() and f.stat().st_size > 2000:
                        sources.append(kind)
                if not sources and (d / "video.webm").exists():
                    sources = ["video"]
                if not sources:
                    raise RuntimeError("No sound was recorded")
                write_meta(d, step="Listening to the recording")
                parts, lang = [], None
                voices_found = {}
                for kind in sources:
                    wav = d / (kind + ".wav")
                    src = d / (kind + ".webm")
                    if not ffmpeg("-i", str(src), "-ac", "1", "-ar", "16000", str(wav)):
                        raise RuntimeError("Could not read the %s sound" % kind)
                    if kind == "mic":
                        who = meta["you"] if meta["mode"] == "call" else "Voices"
                    elif kind == "others":
                        who = "Others"
                    else:
                        who = "Voices"
                    label = {"mic": "your voice", "others": "the other voices", "video": "the recording"}[kind]
                    state = {"last": -1}

                    def prog(frac, label=label):
                        pct = int(frac * 100)
                        if pct >= state["last"] + 3:
                            state["last"] = pct
                            write_meta(d, step="Listening to %s (%d%%)" % (label, pct))

                    segs, language = transcribe.transcribe(str(wav), who, prog)
                    lang = lang or language
                    for s in segs:
                        s["_kind"] = kind
                    if not (kind == "mic" and meta.get("mode") == "call"):
                        try:
                            import voice
                            write_meta(d, step="Recognising voices")
                            found = voice.label_segments(segs, str(wav))
                            if found:
                                voices_found.update(found)
                        except Exception:
                            pass
                    parts.extend(segs)
                    wav.unlink(missing_ok=True)
                segments = merge(parts)
                (d / "transcript.json").write_text(json.dumps({"language": lang, "segments": segments}, ensure_ascii=False))
                if voices_found:
                    write_meta(d, voices=sorted(voices_found, key=lambda k: -voices_found[k]))
            write_meta(d, step="Writing the notes")
            n = notes.generate(segments, meta.get("title", ""), meta.get("you", "You"), meta.get("mode", "call"))
            issues = quality_issues(segments, n)
            if issues and str(n.get("engine", "")).startswith("local"):  # not good enough: try once more before anything is shown
                write_meta(d, step="Writing the notes again (first try was thin)")
                n2 = notes.generate(segments, meta.get("title", ""), meta.get("you", "You"), meta.get("mode", "call"))
                if len(quality_issues(segments, n2)) <= len(issues):
                    n, issues = n2, quality_issues(segments, n2)
            write_meta(d, quality_issues=issues)
            (d / "notes.json").write_text(json.dumps(n, ensure_ascii=False))
            try:
                write_meta(d, notes_file=export_notes(d, meta, segments, n))
            except Exception:
                pass
            try:
                write_meta(d, brief_file=export_brief(d, read_meta(d), segments, n))
            except Exception:
                pass
            try:
                write_meta(d, report_file=export_report(d, read_meta(d), segments, n), headline=n.get("headline", ""), kind=n.get("kind", ""))
            except Exception:
                pass
            write_meta(d, step="Cleaning the sound")
            try:
                clean_audio.make_clean(d)
            except Exception:
                pass
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            write_meta(d, status="done", step="Ready", language=lang, words=sum(len(s["text"].split()) for s in segments), size=size, error=None)
    except Exception as e:
        write_meta(d, status="error", step="Something went wrong", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("MERIDIAN_PORT", "8790"))
    try:
        from waitress import serve
    except ImportError:
        app.run(host="127.0.0.1", port=port, threaded=True)
    else:
        serve(app, host="127.0.0.1", port=port, threads=8, max_request_body_size=512 * 1024 * 1024, channel_timeout=600)
