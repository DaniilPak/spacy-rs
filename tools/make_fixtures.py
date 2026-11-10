"""Generate golden fixtures for spacy-rs with real spaCy."""
import ast, json, os, sys, glob, random
import spacy
OUT = sys.argv[1]
TESTS = os.path.join(os.path.dirname(spacy.__file__), "tests")

def harvest_strings(paths):
    out = []
    for p in paths:
        try:
            tree = ast.parse(open(p, encoding="utf8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if 0 < len(s) < 2000 and "\x00" not in s:
                    out.append(s)
    seen = set(); res = []
    for s in out:
        if s not in seen:
            seen.add(s); res.append(s)
    return res

def tok_attrs(t):
    return [t.text, t.idx, t.whitespace_, t.lower_, t.norm_, t.shape_, t.prefix_, t.suffix_,
            int(t.is_alpha), int(t.is_ascii), int(t.is_digit), int(t.is_lower), int(t.is_upper),
            int(t.is_title), int(t.is_punct), int(t.is_left_punct), int(t.is_right_punct),
            int(t.is_space), int(t.is_bracket), int(t.is_quote), int(t.is_currency),
            int(t.like_url), int(t.like_num), int(t.like_email), int(t.is_stop)]

def dump_tokens(lang, texts, name):
    nlp = spacy.blank(lang)
    docs = []
    for text in texts:
        doc = nlp(text)
        docs.append({"text": text, "tokens": [tok_attrs(t) for t in doc]})
    with open(os.path.join(OUT, name), "w", encoding="utf8") as f:
        json.dump({"lang": lang, "docs": docs}, f, ensure_ascii=False)
    print(name, len(docs), sum(len(d["tokens"]) for d in docs))

if __name__ == "__main__":
    corpus = os.path.join(os.path.dirname(__file__), "..", "tests", "corpus")
    en = [l.rstrip("\n") for l in open(os.path.join(corpus, "en_handwritten.txt"), encoding="utf8")]
    en.append(open(os.path.join(corpus, "en_handwritten.txt"), encoding="utf8").read())
    en += [open(os.path.join(TESTS, "tokenizer", "sun.txt"), encoding="utf8").read()]
    harvested = harvest_strings(glob.glob(os.path.join(TESTS, "tokenizer", "*.py")) +
                                glob.glob(os.path.join(TESTS, "lang", "en", "*.py")) +
                                glob.glob(os.path.join(TESTS, "lang", "test_*.py")) +
                                glob.glob(os.path.join(TESTS, "doc", "*.py")) +
                                glob.glob(os.path.join(TESTS, "matcher", "*.py")))
    dump_tokens("en", en + harvested, "tokens_en.json")
    for lang in ["de", "fr", "es", "it", "pt", "nl", "xx"]:
        texts = harvest_strings(glob.glob(os.path.join(TESTS, "lang", lang, "*.py")))
        texts += harvest_strings(glob.glob(os.path.join(TESTS, "tokenizer", "test_naughty_strings.py")))
        texts += en[:60]
        dump_tokens(lang, texts, f"tokens_{lang}.json")

def fuzz_texts(n, seed):
    rnd = random.Random(seed)
    pieces = ["hello", "World", "can't", "won't", "U.S.", "e.g.", "Dr.", "3.14", "1,000", "$", "€", "%", "(", ")",
              "[", "]", "\"", "'", "''", "``", "...", "…", "-", "--", "—", "/", "\\", ":", ";", "!", "?", ".", ",",
              "😀", "❤️", "🇺🇸", "👍🏽", "http://x.com", "www.a.org/b?c=d", "a@b.com", "#tag", "@user", "10km",
              "5pm", "C++", "x-y", "a/b", "1/2", "2nd", "'s", "’s", "n't", "Über", "naïve", "İstanbul", "ΣΟΦΙΑ",
              "日本語", "中文", "한국어", "مرحبا", "שלום", "\t", "\n", "  ", " ", "°C", "km²", "±5", "~3", "+1",
              "'80s", "O'Neil", "rock'n'roll", "ʼ", "a.m.", "p.m.", ":)", ":-(", "<3", "^_^", "=)", "xD", "&amp;",
              "&", "*", "**", "_", "__init__", "foo_bar", "fooBar", "FOO", "Ph.D.", "1990s", "i.e.", "vs.", "etc.",
              " ", "​", " ", "´", "`", "‘", "’", "“", "”", "„", "«", "»", "‹", "›"]
    out = []
    for _ in range(n):
        k = rnd.randint(1, 12)
        s = ""
        for _ in range(k):
            s += rnd.choice(pieces)
            r = rnd.random()
            if r < 0.5: s += " "
            elif r < 0.55: s += "  "
        out.append(s)
    return out

if __name__ == "__main__":
    dump_tokens("en", fuzz_texts(3000, 1), "tokens_en_fuzz.json")
    dump_tokens("fr", fuzz_texts(800, 2), "tokens_fr_fuzz.json")
    dump_tokens("de", fuzz_texts(800, 3), "tokens_de_fuzz.json")
