"""Cloud helper: take one waiting recording from the Drive mailbox, write the first version of the notes, send them back."""
import base64, json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MERIDIAN_NOTES_DIR", "/tmp/meridian-notes")
import server  # the same pipeline as on the Mac
def box(o, t=180):
    o["key"] = os.environ["MERIDIAN_ADMIN"]
    req = urllib.request.Request(os.environ["MERIDIAN_URL"], data=json.dumps(o).encode(), headers={"Content-Type": "text/plain"})
    return json.load(urllib.request.urlopen(req, timeout=t))
wait = 0 if os.environ.get("FORCE") == "true" else int(os.environ.get("MIN_AGE_MIN", "25"))
now = time.time() * 1000
todo = [p for p in box({"a": "pending"}).get("pending", []) if not str(p.get("title", "")).startswith("VOICE::") and now - (p.get("finished") or 0) > wait * 60000]
for p in todo[:1]:
    mid = p["id"]
    if not box({"a": "claim", "id": mid, "by": "cloud"}).get("granted"): continue
    try:
        info = box({"a": "meta", "id": mid}); meta, fin = info.get("meta") or {}, info.get("finish") or {}
        d = server.DATA / mid; d.mkdir(parents=True, exist_ok=True); got = 0
        for kind in ("mic", "others"):
            n, chunks = 0, []
            while True:
                r = box({"a": "parts", "id": mid, "kind": kind, "from": n, "count": 25})
                chunks += r.get("data", [])
                if len(r.get("data", [])) < 25: break
                n += 25
            if chunks:
                with open(d / (kind + ".webm"), "wb") as out:
                    for c in chunks: out.write(base64.b64decode(c))
                got += len(chunks)
        if not got: raise RuntimeError("No sound was found in this recording")
        m = {"id": mid, "title": meta.get("title") or "Recording", "you": meta.get("you") or "Guest", "mode": meta.get("mode") or "room",
             "source": "cloud", "created": (meta.get("started") or time.time() * 1000) / 1000, "owner": "host",
             "status": "queued", "step": "Starting", "duration": fin.get("duration") or 0}
        server.write_meta(d, **m)
        last = [""]
        import threading
        def tick():
            while not stop.is_set():
                try:
                    step = server.read_meta(d).get("step", "")
                    if step != last[0]: box({"a": "status", "id": mid, "step": step + " (first draft in the cloud)", "state": "processing"}, 60); last[0] = step
                except Exception: pass
                stop.wait(30)
        stop = threading.Event(); threading.Thread(target=tick, daemon=True).start()
        server.run_pipeline(mid)
        stop.set()
        meta2 = server.read_meta(d)
        clean = lambda x: {k: v for k, v in x.items() if not k.endswith("_file") and k not in ("owner", "quality_issues")}
        summ = clean(meta2)
        if meta2.get("status") == "done":
            full = clean(server.app.test_client().get("/api/meeting/" + mid).get_json()); full.pop("has", None)
            box({"a": "put", "id": mid, "name": "result.json", "content": json.dumps(full, ensure_ascii=False), "summary": summ, "by": "cloud"}, 300)
            print("done", mid)
        else:
            summ["status"] = "error"; box({"a": "put", "id": mid, "name": "result.json", "content": json.dumps(summ), "summary": summ, "by": "cloud"})
            print("error", mid, meta2.get("error"))
    except Exception as e:
        print("failed", mid, repr(e))
        try: box({"a": "status", "id": mid, "state": "waiting", "step": "Saved safely. Will be retried"}, 60)
        except Exception: pass
        sys.exit(1)
