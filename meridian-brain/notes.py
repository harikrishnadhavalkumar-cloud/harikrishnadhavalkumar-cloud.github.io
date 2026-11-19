"""Turns a transcript into summary, decisions, action items and key moments.

Smart notes use the Claude API when ANTHROPIC_API_KEY is set (in the environment or in
meeting_bot/.env). Without a key, a simple built-in method is used instead.
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
MODEL = os.environ.get("MERIDIAN_NOTES_MODEL", "claude-sonnet-5-5")


def _api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def mmss(t):
    t = int(t)
    return "%02d:%02d" % (t // 60, t % 60)


def transcript_text(segments):
    return "\n".join("[%s] %s: %s" % (mmss(s["start"]), s["speaker"], s["text"]) for s in segments)


def _empty_part(name, key, you):
    who = "you" if key == "mine" else "the other people"
    return {"name": name, "summary": "No speech from %s was found in this recording." % who, "points": [], "actions": []}


def _split(segments, you):
    mine = [s for s in segments if s["speaker"] == you]
    others = [s for s in segments if s["speaker"] == "Others"]
    return mine, others


def generate(segments, title="", you="You", mode="call"):
    if not segments:
        return {"engine": "none", "summary": "No speech was found in this recording.", "decisions": [], "actions": [], "moments": []}
    split = mode == "call"
    key = _api_key()
    if key:
        try:
            return _claude_notes(key, segments, title, you, split)
        except Exception as e:  # fall back, but say so
            out = _basic_notes(segments, you, split)
            out["note"] = "Smart notes failed (%s). Basic notes shown." % type(e).__name__
            return out
    if _local_ready():
        try:
            return _local_notes(segments, you, split)
        except Exception as e:
            out = _basic_notes(segments, you, split)
            out["note"] = "The free AI had a problem (%s). Basic notes shown." % type(e).__name__
            return out
    out = _basic_notes(segments, you, split)
    out["note"] = "Basic notes. The free AI (Ollama) was not running."
    return out


def _clean_actions(items):
    return [
        {"owner": str(a.get("owner", "Unassigned")), "task": str(a.get("task", "")), "due": str(a.get("due", ""))}
        for a in items or []
    ]


def _clean_part(d, name):
    d = d or {}
    return {
        "name": name,
        "summary": str(d.get("summary", "")),
        "points": [str(x) for x in d.get("points", [])],
        "actions": _clean_actions(d.get("actions")),
    }


def _claude_notes(key, segments, title, you="You", split=True):
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    text = transcript_text(segments)
    mine, others = _split(segments, you)
    if split and not (mine and others):
        out = _basic_notes(segments, you, True)
        out["note"] = "Only one side spoke, so basic notes are shown."
        return out
    shape = (
        '{"summary": "overall summary, 3 to 5 sentences", "decisions": ["..."], '
        '"actions": [{"owner": "name or Unassigned", "task": "...", "due": "date or empty"}], '
        '"moments": [{"t": seconds_as_number, "label": "short label"}]'
    )
    if split:
        part = '{"summary": "2 to 4 sentences", "points": ["key point"], "actions": [{"owner": "name", "task": "...", "due": ""}]}'
        shape += ', "mine": ' + part + ', "others": ' + part
    shape += "}"
    prompt = (
        "Below is the transcript of a meeting%s. Lines by '%s' are the person who recorded the meeting. "
        "Lines by 'Others' are everyone else.\n"
        "Write notes in the same language and simple words the speakers used.\n"
        "Reply with JSON only, no other text, in exactly this shape:\n%s\n"
        "%s"
        "Only include decisions and actions that were really said. Do not invent anything. "
        "Moments: 3 to 6 important points, t taken from the transcript timestamps.\n\nTranscript:\n%s"
        % (
            (" titled '%s'" % title) if title else "",
            you,
            shape,
            ("'mine' covers only what %s said. 'others' covers only what the others said. 'summary' covers the whole meeting.\n" % you) if split else "",
            text[:60000],
        )
    )
    msg = client.messages.create(model=MODEL, max_tokens=3500, messages=[{"role": "user", "content": prompt}])
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0))
    out = {
        "engine": "claude",
        "summary": str(data.get("summary", "")),
        "decisions": [str(x) for x in data.get("decisions", [])],
        "actions": _clean_actions(data.get("actions")),
        "moments": [{"t": float(x.get("t", 0)), "label": str(x.get("label", ""))} for x in data.get("moments", [])],
    }
    if split:
        out["parts"] = {"mine": _clean_part(data.get("mine"), you), "others": _clean_part(data.get("others"), "Others")}
    return out


LOCAL_MODEL = os.environ.get("MERIDIAN_LOCAL_MODEL", "")
PREFERRED_MODELS = ["qwen2.5:7b", "qwen2.5:3b"]  # the bigger one is sharper; the small one is the fallback
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


def _local_ready():
    """True when the free local AI is running. Picks the best model that is installed."""
    global LOCAL_MODEL
    import urllib.request
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return False
    if LOCAL_MODEL and LOCAL_MODEL in names:
        return True
    for m in PREFERRED_MODELS:
        if m in names:
            LOCAL_MODEL = m
            return True
    return False


def _local_json(system, user):
    import urllib.request
    body = json.dumps({
        "model": LOCAL_MODEL, "stream": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 950, "repeat_penalty": 1.15},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        raw = json.loads(r.read())["message"]["content"]
    return json.loads(raw)


_SYS = (
    "You write meeting notes. Use short, simple words a school student understands. "
    "Do not copy long sentences from the transcript; say it in your own words. "
    "Use only what was really said. Never invent names, numbers, dates, tasks, reactions or thanks. Mention only people who actually spoke. "
    "Keep numbers and money words exactly as spoken: never add a $ sign or a currency that was not said. "
    "Reply with JSON only."
)


def _clip(text, max_words=5200):
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " ..."


def _strs(v):
    out = []
    for x in v or []:
        x = x if isinstance(x, str) else (x.get("text") or x.get("task") or "") if isinstance(x, dict) else str(x)
        x = x.strip()
        if x and "<" not in x and not x.lower().startswith(("up to ", "short key point", "something the group")):
            out.append(x)
    return out


def _acts(v, default_owner):
    out = []
    for a in v or []:
        if isinstance(a, dict) and str(a.get("task", "")).strip():
            owner = str(a.get("owner") or default_owner).strip() or default_owner
            due = str(a.get("due") or "").strip()
            out.append({"owner": owner, "task": str(a["task"]).strip(), "due": "" if due.lower() in ("none", "null", "n/a", "unknown") else due})
    return out


def _chunks(segs, max_words=3000):
    out, cur, n = [], [], 0
    for s in segs:
        w = len(s["text"].split())
        if cur and n + w > max_words:
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(cur)
    return out


def _toks(x):
    return set(re.findall(r"[a-z0-9]+", x.lower()))


def _dedupe_text(items, limit):
    out, seen = [], []
    for x in items:
        w = _toks(x)
        if w and not any(len(w & m) / len(w | m) > 0.6 for m in seen):
            out.append(x)
            seen.append(w)
    return out[:limit]


def _dedupe_actions(acts, limit):
    out, seen = [], []
    for a in acts:
        w = _toks(a["task"])
        if w and not any(len(w & m) / len(w | m) > 0.55 for m in seen):
            out.append(a)
            seen.append(w)
    return out[:limit]


def _join_summaries(texts, who):
    texts = [t for t in texts if t]
    if len(texts) <= 1:
        return texts[0] if texts else ""
    listing = "\n".join("%d) %s" % (i + 1, t) for i, t in enumerate(texts))
    try:
        d = _local_json(
            _SYS,
            ("Short summaries of consecutive parts of %s:\n%s\n\n"
             'Return JSON: {"summary": "<3 to 6 simple sentences covering all the parts>"}. '
             "Replace the <...> with real content.") % (who, listing),
        )
        t = str(d.get("summary", "")).strip()
        if t and "<" not in t:
            return t
    except Exception:
        pass
    return " ".join(texts)[:1200]


def _local_part(segs, name, who_text):
    sums, pts, acts = [], [], []
    for chunk in _chunks(segs):
        text = transcript_text(chunk)
        try:
          d = _local_json(
            _SYS,
            ("Transcript of %s:\n%s\n\nReturn JSON: "
             '{"summary": "<2 to 4 simple sentences>", "points": ["<short key point>"], '
             '"actions": [{"owner": "<who must do it>", "task": "<what to do>", "due": "<when, or empty>"}]}. '
             "Give at most 5 points. Replace every <...> with real content from the transcript. "
             "Only list actions where someone clearly said they will do something or must do something. "
             "If there are none, use an empty list.") % (who_text, text),
          )
        except Exception:
            d = _basic_core(chunk)
            d = {"summary": d["summary"], "points": d["decisions"], "actions": d["actions"]}
        sums.append(str(d.get("summary", "")).strip())
        pts += _strs(d.get("points"))
        acts += _acts(d.get("actions"), name)
    return {"name": name, "summary": _join_summaries(sums, who_text), "points": _dedupe_text(pts, 8), "actions": _dedupe_actions(acts, 12)}


def _local_notes(segments, you, split):
    mine, others = _split(segments, you)
    sums, decs, acts, tops, heads, kinds, keys = [], [], [], [], [], [], []
    for chunk in _chunks(segments, 3600):
        try:
          d = _local_json(
            _SYS,
            ("Transcript of a meeting:\n%s\n\nReturn JSON: "
             '{"summary": "<3 to 5 simple sentences about the whole meeting>", "decisions": ["<something the group agreed or decided>"], '
             '"actions": [{"owner": "<who>", "task": "<what>", "due": "<when, or empty>"}], '
             '"topics": [{"title": "<3 to 6 words>", "summary": "<one short sentence>"}], '
             '"headline": "<one sentence, at most 18 words: the single most important point of the meeting>", '
             '"kind": "<2 to 3 words: what kind of meeting this was>", '
             '"key_points": ["<an important point someone senior must know>"]}. '
             "Give 4 to 7 topics that cover the whole transcript in the order they came up. Give 3 to 5 key points. "
             "Replace every <...> with real content from the transcript. "
             "Only include decisions and actions that were really said. Empty list if none.") % transcript_text(chunk),
          )
        except Exception:
            bc = _basic_core(chunk)
            d = {"summary": bc["summary"], "decisions": bc["decisions"], "actions": []}
        sums.append(str(d.get("summary", "")).strip())
        decs += _strs(d.get("decisions"))
        acts += _acts(d.get("actions"), "Unassigned")
        hl = str(d.get("headline", "")).strip()
        if hl and "<" not in hl:
            heads.append(hl)
        kd = str(d.get("kind", "")).strip()
        if kd and "<" not in kd:
            kinds.append(kd[:40])
        keys += _strs(d.get("key_points"))
        for tp in d.get("topics") or []:
            if isinstance(tp, dict) and str(tp.get("title", "")).strip() and "<" not in str(tp.get("title", "")):
                tops.append({"title": str(tp["title"]).strip()[:60], "summary": str(tp.get("summary", "")).strip() if "<" not in str(tp.get("summary", "")) else ""})
    base = _basic_core(segments)  # key moments with real timestamps
    out = {
        "engine": "local",
        "summary": _join_summaries(sums, "the whole meeting") or base["summary"],
        "decisions": _dedupe_text(decs, 10),
        "actions": _dedupe_actions(acts, 12),
        "moments": base["moments"],
        "topics": tops[:8],
        "headline": heads[0] if heads else "",
        "kind": kinds[0] if kinds else "",
        "key_points": _dedupe_text(keys, 5),
    }
    topics = out["topics"]
    if topics and len(segments) >= 12:
        groups = _assign_topics(segments, topics)
        total_w = max(1, sum(len(x["text"].split()) for x in segments))
        for t, g in zip(topics, groups):
            t["share"] = round(sum(len(x["text"].split()) for x in g) / total_w * 100)
            if len(g) >= 3:
                try:
                    t.update(_local_topic_detail(t["title"], g))
                except Exception:
                    pass
        out["main_focus"] = max(topics, key=lambda t: t.get("share", 0))["title"]
        paras = ["%d) %s: %s" % (i + 1, t["title"], t.get("paragraph") or t.get("summary", "")) for i, t in enumerate(topics)]
        try:
            ov = _local_json(
                _SYS,
                ("Notes on each topic of one meeting, in order:\n%s\n\n"
                 'Return JSON: {"overview": "<two short paragraphs, 6 to 9 sentences in total, giving a complete overview of the whole meeting from start to finish>"}. '
                 "Replace the <...> with real content.") % "\n".join(paras),
            )
            txt = str(ov.get("overview", "")).strip()
            if txt and "<" not in txt:
                out["overview"] = txt
        except Exception:
            pass
    if len(segments) >= 12:
        stories = _narrate(segments)
        if len(stories) >= 2:
            lead = (out.get("headline") or "").rstrip(". ")
            out["overview"] = (("This was a %s. " % out["kind"].lower().rstrip(".")) if out.get("kind") else "") + "\n\n".join(stories)
            out["narrative"] = stories
    out["mentions"] = _mentions(segments)
    qs = []
    for t in topics:
        qs += t.get("questions", [])
    out["open_questions"] = _dedupe_text(qs, 6)
    if split:
        parts = {}
        for key, name, segs, label in (("mine", you, mine, "what %s said" % you), ("others", "Others", others, "what the other people said")):
            parts[key] = _local_part(segs, name, label) if segs else _empty_part(name, key, you)
        out["parts"] = parts
        out["actions"] = _dedupe_actions(parts["mine"]["actions"] + parts["others"]["actions"], 14)
        if not (mine and others):
            only = parts["mine"] if mine else parts["others"]
            out["summary"] = only["summary"] or out["summary"]  # one side spoke: use its own summary, nothing extra
        if mine and others:
            out["summary"] = _join_summaries([parts["mine"]["summary"], parts["others"]["summary"]], "the whole meeting, one part per speaker side") or out["summary"]
    return out


def _narrate(segments):
    """Tell the meeting as a story, part by part, in the order it happened."""
    stories = []
    chunks = _chunks(segments, 700)
    total = len(chunks)
    for i, c in enumerate(chunks[:8]):
        text = _clip(_plain_text(c), 1200)
        try:
            d = _local_json(
                _SYS,
                ('This is part %d of %d of one meeting. These are the spoken words, in order:\n%s\n\n'
                 'Return JSON: {"story": "<5 to 7 simple sentences telling what happened in this part: what was being discussed, what people said about it, and where it ended up>"}. '
                 "Do not copy sentences. Do not invent anything. Replace the <...> with real content.") % (i + 1, total, text),
            )
            st = str(d.get("story", "")).strip()
            if st and "<" not in st and len(st.split()) >= 25:
                stories.append(st)
        except Exception:
            continue
    return stories


_COMMON_CAPS = set("The And But Okay Yes No This That What Where When Who How Why Bro So Now Then Well Also Just Because Please Thank Thanks Hello Right Sure Maybe Actually Basically Like Let Can Could Would Should Will Did Does Have Has Had You Your We Our They Their There Here Today Tomorrow Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())


def _assign_topics(segments, topics):
    def toks(t):
        return set(w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2 and w not in _STOP)
    sets = [(toks(t["title"]), toks(t["title"] + " " + t.get("summary", ""))) for t in topics]
    groups = [[] for _ in topics]
    last = 0
    for seg in segments:
        wt = toks(seg["text"])
        best, bs = -1, 0
        for i, (ti, ts) in enumerate(sets):
            sc = sum((2 if w in ti else 1) for w in wt if w in ts)
            if sc > bs:
                bs, best = sc, i
        if best < 0:
            best = last
        else:
            last = best
        groups[best].append(seg)
    return groups


def _plain_text(segs):
    """The words only. No timestamps, and no speaker tags unless there are real names."""
    named = len({x["speaker"] for x in segs} - {"Voices", "Others"}) > 0
    return "\n".join((("%s: " % x["speaker"]) if named else "") + x["text"] for x in segs)


def _clean_point(x):
    x = re.sub(r"^\[?\d{1,2}:\d{2}\]?\s*", "", x.strip())
    x = re.sub(r"^(Voices|Others)\s*:\s*", "", x)
    return x.strip()


def _local_topic_detail(title, segs):
    text = _clip(_plain_text(segs), 1800)
    prompt = ('Part of a meeting about "%s". These are the spoken words, in order:\n%s\n\n'
              "Now write notes about this topic. Explain it in your own words. Do NOT copy sentences from the words above. "
              'Return JSON: {"paragraph": "<4 to 6 simple sentences explaining what was discussed about this topic and what people said or planned>", '
              '"points": ["<a short specific point, in your own words>"], '
              '"outcome": "<what was decided or concluded, or: No decision was made>", '
              '"questions": ["<a question left open>"]}. '
              "Give 3 to 5 points. Use an empty list for questions if there are none. "
              "Replace every <...> with real content.") % (title, text)
    out = {"paragraph": "", "points": [], "outcome": "", "questions": []}
    for attempt in range(2):
        d = _local_json(_SYS, prompt)
        para = str(d.get("paragraph", "")).strip()
        if para and "<" not in para and len(para.split()) >= 32:
            out["paragraph"] = para
            out["points"] = [x for x in (_clean_point(p) for p in _strs(d.get("points"))) if x and not x.startswith("[") and len(x.split()) >= 3][:6]
            oc = str(d.get("outcome", "")).strip()
            out["outcome"] = oc if oc and "<" not in oc else ""
            out["questions"] = _strs(d.get("questions"))[:3]
            break
        prompt = prompt + "\nIMPORTANT: the paragraph must have at least 4 full sentences and about 60 words."
    return out


def _mentions(segments):
    names, terms = Counter(), Counter()
    for seg in segments:
        words = [re.sub(r"[^A-Za-z]", "", w) for w in seg["text"].split()]
        for i, c in enumerate(words):
            if len(c) >= 2 and c.isupper() and len(c) <= 6:
                terms[c] += 1
            elif i > 0 and re.match(r"^[A-Z][a-z]{2,}$", c) and c not in _COMMON_CAPS:
                prev_cap = bool(re.match(r"^[A-Z][a-z]+$", words[i - 1] or ""))
                next_cap = i + 1 < len(words) and bool(re.match(r"^[A-Z][a-z]+$", words[i + 1] or ""))
                if not prev_cap and not next_cap:
                    names[c] += 1
    skip = {"Indian", "India", "Google", "Play", "Console", "Store", "Android", "Apple", "Density", "Doc", "Docs", "Sheet", "Sheets", "Excel", "Tracker", "Metrics", "Target", "Link", "Group"}
    return {"people": [n for n, k in names.most_common(12) if k >= 3 and n not in skip][:8], "terms": [t for t, k in terms.most_common(10) if k >= 2][:8]}


_STOP = set("the a an and or but to of in on at for is are was were be been it this that i you we they he she with as so do not no yes ok okay just".split())
_ACTION = re.compile(r"\b(i will|i'll|we will|we'll|you will|you'll|need to|have to|has to|should|must|going to|let me|i can|can you|please)\b", re.I)
_DECIDE = re.compile(r"\b(decid\w*|agreed?|final\w*|locked?|confirmed?|approved?|let's go with|we go with|settled)\b", re.I)
_DUE = re.compile(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|this week|by \w+)\b", re.I)


def _basic_notes(segments, you="You", split=False):
    out = _basic_core(segments)
    mine, others = _split(segments, you)
    if split:
        parts = {}
        for key, name, segs in (("mine", you, mine), ("others", "Others", others)):
            if segs:
                c = _basic_core(segs)
                parts[key] = {"name": name, "summary": c["summary"], "points": c["decisions"], "actions": c["actions"]}
            else:
                parts[key] = _empty_part(name, key, you)
        out["parts"] = parts
    return out


def _basic_core(segments):
    sents = []
    for s in segments:
        for part in re.split(r"(?<=[.!?])\s+", s["text"]):
            if len(part.split()) >= 4:
                sents.append((s["start"], s["speaker"], part.strip()))
    words = Counter(w for _, _, p in sents for w in re.findall(r"[a-z']+", p.lower()) if w not in _STOP and len(w) > 2)
    scored = sorted(
        range(len(sents)),
        key=lambda i: -sum(words[w] for w in re.findall(r"[a-z']+", sents[i][2].lower())) / (len(sents[i][2].split()) ** 0.7),
    )[:3]
    summary = " ".join(sents[i][2] for i in sorted(scored)) if scored else segments[0]["text"]
    decisions = [p for _, _, p in sents if _DECIDE.search(p)][:6]
    actions = []
    for t, who, p in sents:
        if _ACTION.search(p) and len(actions) < 8:
            d = _DUE.search(p)
            actions.append({"owner": who, "task": p, "due": d.group(0) if d else ""})
    moments = []
    if sents:
        end = max(t for t, _, _ in sents) + 1
        for w in range(5):
            lo, hi = end * w / 5, end * (w + 1) / 5
            pool = [(t, p) for t, _, p in sents if lo <= t < hi]
            if pool:
                t, p = max(pool, key=lambda x: (1 if (_DECIDE.search(x[1]) or _ACTION.search(x[1])) else 0, min(len(x[1].split()), 14)))
                moments.append({"t": t, "label": p[:60]})
    return {"engine": "basic", "summary": summary, "decisions": decisions, "actions": actions, "moments": moments}
