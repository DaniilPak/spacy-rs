"""Golden annotations from en_core_web_sm for spacy-rs parity tests."""
import json, os, sys
import spacy
sys.path.insert(0, os.path.dirname(__file__))
OUT = sys.argv[1]
nlp = spacy.load("en_core_web_sm")
corpus = os.path.join(os.path.dirname(__file__), "..", "tests", "corpus")
texts = [l.rstrip("\n") for l in open(os.path.join(corpus, "en_handwritten.txt"), encoding="utf8") if l.strip()]
texts += [p for p in open(os.path.join(corpus, "en_news.txt"), encoding="utf8").read().split("\n\n") if p.strip()]
TESTS = os.path.join(os.path.dirname(spacy.__file__), "tests")
texts.append(open(os.path.join(TESTS, "tokenizer", "sun.txt"), encoding="utf8").read())
import glob
from make_fixtures import harvest_strings, fuzz_texts
harv = harvest_strings(glob.glob(os.path.join(TESTS, "lang", "en", "*.py")) + glob.glob(os.path.join(TESTS, "tokenizer", "*.py"))
                       + glob.glob(os.path.join(TESTS, "doc", "*.py")) + glob.glob(os.path.join(TESTS, "parser", "*.py")) + glob.glob(os.path.join(TESTS, "pipeline", "*.py")))
harv = [h for h in harv if 3 <= len(h) <= 600 and " " in h]
texts += harv
texts += fuzz_texts(400, 7)
docs = []
for doc in nlp.pipe(texts, batch_size=1):
    docs.append({
        "text": doc.text,
        "tokens": [[t.text, t.tag_, t.pos_, str(t.morph), t.lemma_, t.dep_, t.head.i, t.ent_iob_, t.ent_type_, int(bool(t.is_sent_start))] for t in doc],
        "ents": [[e.start, e.end, e.label_] for e in doc.ents],
        "sents": [s.start for s in doc.sents],
        "noun_chunks": [[c.start, c.end] for c in doc.noun_chunks],
    })
d0 = nlp("Apple is looking at buying U.K. startup for $1 billion")
json.dump({"model": nlp.meta["name"] + "-" + nlp.meta["version"], "docs": docs,
           "tensor0": d0.tensor[:3, :8].tolist()}, open(os.path.join(OUT, "model_en_core_web_sm.json"), "w", encoding="utf8"), ensure_ascii=False)
print(len(docs), sum(len(d["tokens"]) for d in docs))
