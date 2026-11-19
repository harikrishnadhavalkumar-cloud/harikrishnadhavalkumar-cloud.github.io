"""Deep notes: read EVERY part of the meeting, keep every topic, deadline, tool, number and project, then check that nothing was dropped.

Written for people who were not in the meeting: they should be able to read the report and know everything.
Uses the free local AI only (Ollama). Safe to fail: the caller keeps the older notes when this raises.
"""
import json
import re
import time
import urllib.request

import glossary

WINDOW_WORDS = 380          # about 3 to 4 minutes of talk per section
MAX_TOPICS = 9
COMMON = set("""The And But Okay Yes No This That What Where When Who How Why Bro So Now Then Well Also Just Because Please Thank Thanks Hello
Right Sure Maybe Actually Basically Like Let Can Could Would Should Will Did Does Have Has Had You Your We Our They Their There Here Today
Tomorrow Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November December
Hindi English Wow Crazy Great Nice Cool Done Hmm Okay Sir Bhai Yaar Look See Tell Give Take Send Call Check Open Close Let's It's I'm""".split())


def _notes():
    import notes
    return notes


def mm(t):
    t = int(max(0, t))
    return "%d:%02d" % (t // 60, t % 60)


def _json(system, user, num_predict=1400):
    n = _notes()
    body = json.dumps({
        "model": n.LOCAL_MODEL, "stream": False, "format": "json", "keep_alive": "10m",
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": num_predict, "repeat_penalty": 1.05},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(n.OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read())["message"]["content"]
    return json.loads(raw)


def _try_json(system, user, num_predict=1400, tries=2):
    last = None
    for _ in range(tries):
        try:
            return _json(system, user, num_predict)
        except Exception as e:  # bad JSON or timeout: try once more
            last = e
    raise last


_ACCESS_WORDS = ("free", "paid", "premium", "trial", "credit", "subscri", "shared", "account", "cost", "price", "buy", "bought", "expire", "member", "login", "access", "cheap", "rupee", "lakh", "pack")


def _tidy(x):
    return glossary.clean_text(x) if isinstance(x, str) else x


def _s(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def _ok(x):
    x = _s(x)
    return x if x and "<" not in x and x.lower() not in ("none", "null", "n/a", "unknown", "empty", "not specified", "not mentioned", "unspecified", "not stated", "tbd", "na", "-") else ""


def _strs(v, limit=12):
    out = []
    for x in v if isinstance(v, list) else []:
        if isinstance(x, dict):
            x = " - ".join(_ok(y) for y in x.values() if _ok(y))
        x = _ok(x)
        if x and x.lower() not in [o.lower() for o in out]:
            out.append(x)
    return out[:limit]


def _dicts(v, keys, limit=12):
    out = []
    for x in v if isinstance(v, list) else []:
        if isinstance(x, dict):
            d = {k: _ok(x.get(k)) for k in keys}
            if d[keys[0]]:
                out.append(d)
    return out[:limit]


def _system(terms):
    return (
        "You are a meticulous meeting note-taker writing for a person who was NOT in the meeting and must understand everything that happened. "
        "Capture EVERY topic, update, request, new project, deadline (and any change to a deadline), tool, access or credit, price, number, name and decision in the text, even small ones. "
        "Be specific: keep the real names, tools, numbers, dates and amounts exactly as spoken. Never write vague filler like 'they discussed various things'. "
        "The speakers mix Hindi and English and the text may contain hearing mistakes: understand the meaning and write clear, professional English. "
        "When a line starts with a person's name, that person said it: use the names to say who said, owns or promised what. "
        "If a part is hard to follow, say so in 'unclear' instead of guessing. Never invent names, numbers, dates or reactions. "
        "Words this team uses: %s. Reply with JSON only." % terms
    )


def _windows(segs):
    out, cur, n = [], [], 0
    for s in segs:
        w = len(s["text"].split())
        if cur and n + w > WINDOW_WORDS:
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        if out and n < WINDOW_WORDS * 0.35:
            out[-1].extend(cur)       # do not leave a tiny last piece alone
        else:
            out.append(cur)
    return out


GENERIC = ("Voices", "Others", "You", "Speaker")


def _text(win):
    out = []
    for s in win:
        sp = s.get("speaker", "")
        out.append("[%s] %s%s" % (mm(s["start"]), (sp + ": ") if sp and sp not in GENERIC else "", s["text"]))
    return "\n".join(out)


def _section(i, total, win, system):
    start, end = win[0]["start"], win[-1].get("end", win[-1]["start"])
    user = (
        "Part %d of %d of one meeting (%s to %s). Spoken words with times:\n%s\n\n"
        'Return JSON with exactly these keys: {"title": "<up to 8 words naming what this part is about>", '
        '"summary": "<3 to 5 specific sentences: what was discussed, who said what, where it ended>", '
        '"bullets": ["<specific fact, 4 to 10 items, nothing important left out>"], '
        '"decisions": ["<anything agreed or decided>"], '
        '"deadlines": [{"item": "<what>", "date": "<when>", "change": "new or moved or confirmed or at risk", "detail": "<for example moved from 13th to 20th>", "owner": "<who>"}], '
        '"tools": [{"name": "<tool or service>", "what": "<what it is for>", "access": "free or paid or trial or credits or shared account or unknown", "cost": "<price if said>", "note": "<anything else said>"}], '
        '"projects": [{"name": "<project or request>", "kind": "new project or ongoing or request or idea", "from": "<who or where it came from>", "detail": "<what it involves>"}], '
        '"numbers": [{"label": "<what the number is>", "value": "<the number exactly as said>", "context": "<why it matters>"}], '
        '"tasks": [{"owner": "<who>", "task": "<what>", "due": "<when or empty>"}], '
        '"unclear": ["<anything you could not follow>"]}. '
        "Use empty lists when there is nothing. Replace every <...> with real content from the words above."
    ) % (i + 1, total, mm(start), mm(end), _text(win))
    sec = {"start": int(start), "end": int(end), "title": "", "summary": "", "bullets": [], "decisions": [], "deadlines": [],
           "tools": [], "projects": [], "numbers": [], "tasks": [], "unclear": []}
    try:
        d = _try_json(system, user)
        sec["title"] = _ok(d.get("title"))[:90]
        sec["summary"] = _ok(d.get("summary"))
        sec["bullets"] = _strs(d.get("bullets"), 10)
        sec["decisions"] = _strs(d.get("decisions"), 6)
        sec["deadlines"] = _dicts(d.get("deadlines"), ["item", "date", "change", "detail", "owner"], 6)
        sec["tools"] = _dicts(d.get("tools"), ["name", "what", "access", "cost", "note"], 8)
        sec["projects"] = _dicts(d.get("projects"), ["name", "kind", "from", "detail"], 6)
        sec["numbers"] = _dicts(d.get("numbers"), ["label", "value", "context"], 8)
        sec["tasks"] = _dicts(d.get("tasks"), ["owner", "task", "due"], 6)
        sec["unclear"] = _strs(d.get("unclear"), 3)
        raw = " ".join(s["text"] for s in win).lower()
        for t in sec["tools"]:
            t["name"] = glossary.canon(re.sub(r"\s+(tool|app|account|software)$", "", t["name"], flags=re.I).strip())
            if t["access"].lower() not in ("free", "paid", "trial", "credits", "shared account") or not any(w in raw for w in _ACCESS_WORDS):
                t["access"] = "unknown"
        sec["title"] = _tidy(sec["title"]); sec["summary"] = _tidy(sec["summary"])
        sec["bullets"] = [_tidy(b) for b in sec["bullets"]]
        sec["unclear"] = [_tidy(b) for b in sec["unclear"]]
    except Exception:
        pass
    if len(sec["summary"].split()) < 12:            # the AI failed: never leave a hole, keep the words themselves
        raw = " ".join(s["text"] for s in win)
        sec["summary"] = " ".join(raw.split()[:70]) + " ..."
        sec["title"] = sec["title"] or "Discussion"
        sec["fallback"] = True
    sec["title"] = sec["title"] or "Discussion"
    who = {}
    for s in win:
        sp = s.get("speaker", "")
        if sp and sp not in GENERIC:
            who[sp] = who.get(sp, 0) + max(1, int(s.get("end", s["start"]) - s["start"]))
    sec["speakers"] = [k for k, _ in sorted(who.items(), key=lambda x: -x[1])][:6]
    return sec


_WORDS = None


def _english(w):
    global _WORDS
    if _WORDS is None:
        try:
            _WORDS = set(x.strip().lower() for x in open("/usr/share/dict/words", encoding="utf-8", errors="ignore") if x.strip() and not x[0].isupper())
        except Exception:
            _WORDS = set()
    return w.lower() in _WORDS


def _terms_in_text(segments):
    """The things a reader would miss if they were dropped: acronyms, names, numbers, dates and the team's own words."""
    text = " ".join(s["text"] for s in segments)
    low = text.lower()
    found = {}
    for t in glossary.known_terms():
        n = low.count(t.lower())
        if n:
            found[t] = max(found.get(t, 0), n + 1)
    for m in re.findall(r"\b[A-Z]{2,6}\b", text):
        found[m] = found.get(m, 0) + 1
    for m in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{3,})\b", text):
        if m not in COMMON and not _english(m):
            found[m] = found.get(m, 0) + 1
    out = []
    for k, v in found.items():
        if v >= 2 or k.lower() in [t.lower() for t in glossary.known_terms()]:
            out.append((k, v))
    nums = {}
    for m in re.findall(r"\b\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?\b", text):
        m2 = m.replace(",", "")
        if len(m2.rstrip("stndrh")) >= 3 or m2.endswith(("st", "nd", "rd", "th")):
            nums[m2] = nums.get(m2, 0) + 1
    out += [(k, v) for k, v in nums.items()]
    out.sort(key=lambda x: -x[1])
    seen, res = set(), []
    for k, v in out:
        if k.lower() not in seen:
            seen.add(k.lower())
            res.append(k)
    return res[:70]


def _flatten(n):
    parts = [n.get("catchup", ""), n.get("headline", ""), n.get("overview", "")]
    for s in n.get("sections", []):
        parts += [s.get("title", ""), s.get("summary", "")] + s.get("bullets", []) + s.get("decisions", [])
        for k in ("deadlines", "tools", "projects", "numbers", "tasks"):
            for d in s.get(k, []):
                parts += list(d.values())
    for t in n.get("topics", []):
        parts += [t.get("title", ""), t.get("paragraph", "")] + t.get("points", [])
    for k in ("deadlines", "tools", "projects", "numbers"):
        for d in n.get(k, []):
            parts += list(d.values())
    for a in n.get("actions", []):
        parts += list(a.values())
    return " ".join(str(p) for p in parts).lower().replace(",", "")


def _coverage(n, segments, terms):
    blob = _flatten(n)
    missing = [t for t in terms if t.lower().replace(",", "") not in blob]
    return missing


def _recover(n, segments, missing, system):
    """For every key word that did not make it into the notes, find what was said about it and add it to the right section."""
    added = 0
    for term in missing[:24]:
        tl = term.lower().replace(",", "")
        idx = next((i for i, s in enumerate(segments) if tl in s["text"].lower().replace(",", "")), None)
        if idx is None:
            continue
        ctx = segments[max(0, idx - 2): idx + 3]
        line = ""
        try:
            d = _try_json(system, ('These words were spoken in a meeting:\n%s\n\nIn ONE specific sentence, say what was said about "%s". '
                                   'Return JSON: {"line": "<one sentence>"}. Replace the <...> with real content.') % (_text(ctx), term), 160, 1)
            line = _ok(d.get("line"))
        except Exception:
            pass
        if not line:
            continue
        t0 = segments[idx]["start"]
        sec = next((s for s in n["sections"] if s["start"] <= t0 <= s["end"] + 1), n["sections"][-1])
        if term.lower() not in line.lower():
            line = "%s: %s" % (term, line)
        sec["bullets"].append("Also mentioned (%s) %s" % (mm(t0), line))
        added += 1
    return added


def _merge_lists(secs, key, ident):
    out, seen = [], {}
    for s in secs:
        for d in s.get(key, []):
            k = ident(d)
            if not k:
                continue
            if k in seen:
                for kk, vv in d.items():
                    if vv and (not seen[k].get(kk) or (kk == "access" and seen[k].get(kk) == "unknown" and vv != "unknown")):
                        seen[k][kk] = vv
                continue
            e = dict(d)
            e["at"] = s["start"]
            seen[k] = e
            out.append(e)
    return out



_DATE = re.compile(r"\b(tomorrow|tonight|today|day after tomorrow|next week|this week|next month|end of (the )?(day|week|month)|eod|eow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2})\b", re.I)
_WORK = re.compile(r"\b(deadline|due|submit|send|share|finish|complete|ready|deliver|launch|close|approve|review|update|publish|go live|build|report|move|shift|postpone|extend|delay|pending|renew|buy|expire|start)", re.I)


def deadline_scan(segments, n, system):
    """Read every line for due dates and date changes, so no deadline depends on the AI noticing it."""
    cands, last = [], -99
    for i, s in enumerate(segments):
        win = " ".join(x["text"] for x in segments[max(0, i - 1): i + 2])
        if _DATE.search(s["text"]) and _WORK.search(win) and s["start"] - last > 25:
            cands.append(i)
            last = s["start"]
    added = 0
    for i in cands[:14]:
        ctx = segments[max(0, i - 2): i + 3]
        try:
            d = _try_json(system, (
                "These words were spoken in a meeting:\n%s\n\nIs someone giving, confirming, promising or CHANGING a deadline, due date or delivery date for a piece of work (for example: we will close it tomorrow; Jatin will send the build tomorrow; moved to the 20th)? "
                'Return JSON: {"is_deadline": true or false, "item": "<the work that is due>", "date": "<when, as said>", "change": "new or moved or confirmed or at risk", '
                '"detail": "<for example moved from 13th to 20th, or empty>", "owner": "<who, or empty>"}. If there is no deadline or date commitment, return empty strings for item and date.') % _text(ctx), 300, 1)
        except Exception:
            continue
        item, date = _ok(d.get("item")), _ok(d.get("date"))
        if not item or not date:
            continue
        w = lambda t: set(x for x in re.findall(r"[a-z0-9]+", t.lower()) if len(x) > 2)
        if any(len(w(item) & w(x.get("item", "") + " " + x.get("detail", ""))) >= 2 for x in n.get("deadlines", [])):
            continue
        ch = _ok(d.get("change")).lower()
        n.setdefault("deadlines", []).append({"item": glossary.clean_text(item), "date": date, "change": ch if ch in ("new", "moved", "confirmed", "at risk") else "new",
                                              "detail": _ok(d.get("detail")), "owner": _ok(d.get("owner")), "at": int(segments[i]["start"])})
        added += 1
    return added


def finalize(n):
    """Rules applied after the AI: every task that has a due date also shows up in the deadline tracker."""
    dls = n.setdefault("deadlines", [])
    words = lambda t: set(w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2)
    for a in n.get("actions", []):
        due = _ok(a.get("due"))
        if not due:
            continue
        tw = words(a.get("task", ""))
        if any(len(tw & words(x.get("item", "") + " " + x.get("detail", ""))) >= 2 for x in dls):
            continue
        at = next((s["start"] for s in n.get("sections", []) if any(t.get("task") == a.get("task") for t in s.get("tasks", []))), 0)
        dls.append({"item": a["task"], "date": due, "change": "new", "detail": "", "owner": a.get("owner", ""), "at": at})
    n["deadlines"] = dls[:20]
    return n


def build(segments, you="You"):
    t0 = time.time()
    n = _notes()
    terms_str = glossary.hotwords()
    system = _system(terms_str)
    wins = _windows(segments)
    secs = [_section(i, len(wins), w, system) for i, w in enumerate(wins)]
    out = {"sections": secs}
    out["deadlines"] = _merge_lists(secs, "deadlines", lambda d: (d["item"][:40] + "|" + d["date"]).lower())
    out["tools"] = _merge_lists(secs, "tools", lambda d: glossary.canon(d["name"]).lower())
    out["projects"] = _merge_lists(secs, "projects", lambda d: d["name"][:50].lower())
    out["numbers"] = _merge_lists(secs, "numbers", lambda d: (d["label"][:40] + "|" + d["value"]).lower())
    out["unclear"] = _strs([u for s in secs for u in s["unclear"]], 10)
    decs = _strs([d for s in secs for d in s["decisions"]], 14)
    out["decisions"] = decs
    acts = []
    for s in secs:
        for t in s["tasks"]:
            acts.append({"owner": t["owner"] or "Team", "task": t["task"], "due": t["due"]})
    out["actions"] = acts[:20]
    out["key_points"] = [s["bullets"][0] for s in secs if s["bullets"]][:12]

    # headline, kind and the catch-up briefing
    digest = "\n".join("%s %s: %s" % (mm(s["start"]), s["title"], s["summary"]) for s in secs)
    extra = []
    for k, label in (("deadlines", "Deadlines"), ("tools", "Tools"), ("projects", "Projects"), ("numbers", "Numbers")):
        if out[k]:
            extra.append("%s: %s" % (label, "; ".join(" ".join(str(v) for v in d.values() if v and not isinstance(v, int)) for d in out[k][:10])))
    try:
        d = _try_json(system, (
            "Notes of a whole meeting, in order:\n%s\n\n%s\n\n"
            'Return JSON: {"headline": "<one sentence up to 16 words saying what the meeting was about>", '
            '"kind": "<2 to 4 words, e.g. Team sync, Planning meeting, Status update>", '
            '"catchup": "<130 to 180 words, 2 short paragraphs, for someone who missed the meeting: what was covered, what was decided, what changed, what is due, which tools or credits were discussed and what happens next. Be specific.>", '
            '"main_focus": "<up to 6 words: the main subject>"}. Replace the <...> with real content.') % (digest, "\n".join(extra)), 900)
        out["headline"] = _ok(d.get("headline"))
        out["kind"] = _ok(d.get("kind"))
        out["catchup"] = _ok(d.get("catchup"))
        out["main_focus"] = _ok(d.get("main_focus"))
    except Exception:
        pass
    if len(out.get("catchup", "").split()) < 110:
        try:
            d2 = _try_json(system, (
                "Notes of a whole meeting, in order:\n%s\n\n%s\n\n"
                'Write a briefing for someone who missed the meeting. It MUST be between 140 and 190 words, in 2 paragraphs. '
                "Cover: what was covered, decisions, what changed, deadlines, tools or credits discussed, and what happens next. Name the real tools, numbers and dates. "
                'Return JSON: {"catchup": "<the briefing>"}. Replace the <...> with real content.') % (digest, "\n".join(extra)), 1100, 1)
            c2 = _ok(d2.get("catchup"))
            if len(c2.split()) > len(out.get("catchup", "").split()):
                out["catchup"] = c2
        except Exception:
            pass
    if len(out.get("catchup", "").split()) < 110:   # still short: add the section summaries so nothing is lost
        extra_txt = " ".join(s["summary"] for s in secs)
        out["catchup"] = (out.get("catchup", "") + " " + extra_txt).strip()
        out["catchup"] = " ".join(out["catchup"].split()[:190])

    # topics: every section belongs to one topic, so nothing falls out
    topics = []
    try:
        listing = "\n".join("%d. %s - %s" % (i + 1, s["title"], " ".join(s["summary"].split()[:25])) for i, s in enumerate(secs))
        d = _try_json(system, (
            "These are the parts of one meeting, in order:\n%s\n\nGroup the parts into 3 to %d topics. EVERY part number must appear in exactly one topic. "
            'Return JSON: {"topics": [{"title": "<topic name, up to 7 words>", "parts": [1, 2]}]}. Replace the <...> with real content.') % (listing, MAX_TOPICS), 700)
        used = set()
        for t in d.get("topics", []):
            ids = [int(x) - 1 for x in t.get("parts", []) if str(x).isdigit() and 0 <= int(x) - 1 < len(secs) and int(x) - 1 not in used]
            if ids and _ok(t.get("title")):
                used.update(ids)
                topics.append({"title": _ok(t["title"])[:70], "ids": sorted(ids)})
        for i in range(len(secs)):
            if i not in used:
                (topics[-1]["ids"].append(i) if topics else topics.append({"title": "Discussion", "ids": [i]}))
    except Exception:
        pass
    if not topics:
        topics = [{"title": s["title"], "ids": [i]} for i, s in enumerate(secs)][:MAX_TOPICS]
    total_w = max(1, sum(len(s["text"].split()) for s in segments))
    final = []
    for t in topics:
        group = [secs[i] for i in sorted(set(t["ids"]))]
        facts = "\n".join("- " + b for s in group for b in ([s["summary"]] + s["bullets"]))
        item = {"title": t["title"], "paragraph": " ".join(s["summary"] for s in group)[:900],
                "points": [b for s in group for b in s["bullets"]][:6], "outcome": "", "questions": [],
                "start": group[0]["start"], "end": group[-1]["end"]}
        w = sum(len(x["text"].split()) for x in segments if group[0]["start"] <= x["start"] <= group[-1]["end"] + 1)
        item["share"] = round(w / total_w * 100)
        try:
            d = _try_json(system, (
                'Facts from one topic ("%s") of a meeting:\n%s\n\n'
                'Return JSON: {"paragraph": "<70 to 120 words, specific, covering every important fact above>", "points": ["<4 to 7 specific points>"], '
                '"outcome": "<what was decided or where it ended, or empty>", "questions": ["<open question, up to 3>"]}. Replace the <...> with real content.') % (t["title"], facts[:3500]), 900)
            para = _ok(d.get("paragraph"))
            if len(para.split()) >= 30:
                item["paragraph"] = para
            pts = _strs(d.get("points"), 7)
            if pts:
                item["points"] = pts
            item["outcome"] = _ok(d.get("outcome"))
            item["questions"] = _strs(d.get("questions"), 3)
        except Exception:
            pass
        final.append(item)
    out["topics"] = final
    out["open_questions"] = _strs([q for t in final for q in t["questions"]], 8)
    if not out.get("main_focus") and final:
        out["main_focus"] = max(final, key=lambda t: t["share"])["title"]

    try:
        deadline_scan(segments, out, system)
    except Exception:
        pass
    finalize(out)

    # the self-check: did every important word survive?
    terms = _terms_in_text(segments)
    missing = _coverage(out, segments, terms)
    first_missing = len(missing)
    if missing:
        _recover(out, segments, missing, system)
        missing = _coverage(out, segments, terms)
    out["coverage"] = {"sections": len(secs), "chunks": len(wins), "terms_total": len(terms), "terms_found": len(terms) - len(missing),
                       "missing": missing[:12], "first_pass_missing": first_missing, "checked": True, "seconds": int(time.time() - t0)}
    out["engine"] = "local-deep"
    return out
