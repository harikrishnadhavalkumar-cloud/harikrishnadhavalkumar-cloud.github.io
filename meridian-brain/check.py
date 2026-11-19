"""Light check, runs every 20 minutes: is there a finished recording that nobody has turned into notes yet?"""
import json, os, sys, time, urllib.request
def box(o):
    o["key"] = os.environ["MERIDIAN_ADMIN"]
    req = urllib.request.Request(os.environ["MERIDIAN_URL"], data=json.dumps(o).encode(), headers={"Content-Type": "text/plain"})
    return json.load(urllib.request.urlopen(req, timeout=90))
wait = 0 if os.environ.get("FORCE") == "true" else int(os.environ.get("MIN_AGE_MIN", "25"))
now = time.time() * 1000
try:
    work = [p for p in box({"a": "pending"}).get("pending", []) if not str(p.get("title", "")).startswith("VOICE::") and now - (p.get("finished") or 0) > wait * 60000]
except Exception as e:
    print("mailbox problem", e); work = []
print("waiting recordings:", [(p["id"], p["title"]) for p in work])
open(os.environ["GITHUB_OUTPUT"], "a").write("work=%s\n" % ("true" if work else "false"))
