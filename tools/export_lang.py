"""Export spaCy language data (MIT, Explosion AI) to JSON for spacy-rs."""
import json, os, sys, importlib, spacy
from spacy.symbols import ORTH, NORM
from spacy.lang.norm_exceptions import BASE_NORMS
out = sys.argv[1]
LANGS = sys.argv[2].split(",")
def pat(f):
    if f is None: return None
    return f.__self__.pattern
for lang in LANGS:
    nlp = spacy.blank(lang)
    t = nlp.tokenizer
    rules = {}
    for k, subs in t.rules.items():
        rs = []
        for s in subs:
            d = {"ORTH": s[ORTH] if ORTH in s else s.get("ORTH")}
            n = s.get(NORM, s.get("NORM"))
            if n is not None: d["NORM"] = n
            rs.append(d)
        rules[k] = rs
    tm = t.token_match
    data = {
        "lang": lang,
        "prefix": pat(t.prefix_search),
        "suffix": pat(t.suffix_search),
        "infix": pat(t.infix_finditer),
        "token_match": pat(tm) if tm is not None else None,
        "url_match": pat(t.url_match),
        "rules": dict(sorted(rules.items())),
        "stop_words": sorted(nlp.Defaults.stop_words),
    }
    try:
        la = importlib.import_module(f"spacy.lang.{lang}.lex_attrs")
        nw = getattr(la, "_num_words", None); ow = getattr(la, "_ordinal_words", None)
        if nw is not None: data["num_words"] = sorted(nw)
        if ow is not None: data["ordinal_words"] = sorted(ow)
    except ModuleNotFoundError:
        pass
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{lang}.json"), "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
with open(os.path.join(out, "base_norms.json"), "w", encoding="utf8") as f:
    json.dump(BASE_NORMS, f, ensure_ascii=False, indent=0)
