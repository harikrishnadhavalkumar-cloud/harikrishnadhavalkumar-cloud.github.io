"""Vocabulary pack: words the speech tool should expect, and fixes for words it keeps mishearing."""
import re
from pathlib import Path

PATH = Path(__file__).parent / "glossary.txt"


def load():
    terms, fixes = [], []
    try:
        lines = PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "=>" in ln:
            a, b = [x.strip() for x in ln.split("=>", 1)]
            if a and b:
                fixes.append((a, b))
        else:
            terms.append(ln)
    return terms, fixes


def hotwords():
    """A compact word list for the speech tool."""
    terms, fixes = load()
    seen, out = set(), []
    for t in terms + [b for _, b in fixes]:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)[:900]


_cache = None


def _compiled():
    global _cache
    if _cache is None:
        terms, fixes = load()
        pats = []
        for a, b in sorted(fixes, key=lambda x: -len(x[0])):
            pats.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", re.I), b))
        _cache = pats
    return _cache


def fix(text):
    for p, b in _compiled():
        text = p.sub(b, text)
    return text


def known_terms():
    terms, fixes = load()
    return terms + [b for _, b in fixes]


def canon(name):
    """Snap a tool or word to the team's spelling when it is nearly the same (SEMRUS -> SEMrush)."""
    try:
        from rapidfuzz import fuzz, process
    except Exception:
        return name
    terms = known_terms()
    if not name or not terms:
        return name
    low = {t.lower(): t for t in terms}
    if name.lower() in low:
        return low[name.lower()]
    best = process.extractOne(name.lower(), list(low), scorer=fuzz.ratio, score_cutoff=75)
    return low[best[0]] if best else name


def clean_text(text):
    """Fix known mishearings and tidy spellings inside a sentence."""
    text = fix(text)
    terms = [t for t in known_terms() if len(t) >= 5 and t[0].isupper()]
    try:
        from rapidfuzz import fuzz
    except Exception:
        return text
    words = text.split()
    lowmap = {t.lower(): t for t in known_terms() if len(t) >= 4 and " " not in t}
    for i, w in enumerate(words):
        core = w.strip(".,;:!?()\"'")
        if core.lower() in lowmap and core != lowmap[core.lower()]:
            words[i] = w.replace(core, lowmap[core.lower()])
            continue
        if core.lower() in lowmap or "." in core:
            continue
        if len(core) >= 5 and core[0].isalpha():
            for t in terms:
                if core.lower() != t.lower() and fuzz.ratio(core.lower(), t.lower()) >= 84:
                    words[i] = w.replace(core, t)
                    break
    return " ".join(words)
