"""Golden outputs for Matcher / PhraseMatcher / EntityRuler / Sentencizer / lookup lemmatizer."""
import json, os, sys
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.attrs import LOWER
OUT = sys.argv[1]
corpus = os.path.join(os.path.dirname(__file__), "..", "tests", "corpus", "en_handwritten.txt")
texts = [l.rstrip("\n") for l in open(corpus, encoding="utf8") if l.strip()]
texts += [
    "a b b b c a b", "a a a a", "b c b c b c", "the quick brown fox jumps over the lazy dog",
    "Hello, world! Hello world. hello there", "Google Maps and Google Search and google maps",
    "New York City is in New York State; york is not.", "I like apples, apple pie and APPLES.",
    "x y z x y z x", "1 2 3 4 5 6 7 8 9 10", "a b c d e f g", "colour color colr colouur",
]
patterns = {
    "HELLO": [[{"LOWER": "hello"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": {"IN": ["world", "there"]}}]],
    "AB_STAR": [[{"ORTH": "a"}, {"ORTH": "b", "OP": "*"}]],
    "AB_PLUS": [[{"ORTH": "a"}, {"ORTH": "b", "OP": "+"}, {"ORTH": "c", "OP": "?"}]],
    "NOT_A": [[{"ORTH": "b"}, {"ORTH": "a", "OP": "!"}]],
    "QUANT": [[{"ORTH": "a", "OP": "{2}"}], [{"ORTH": "b", "OP": "{1,2}"}, {"ORTH": "c"}], [{"ORTH": "x", "OP": "{,1}"}, {"ORTH": "y"}], [{"ORTH": "z"}, {"ORTH": "x", "OP": "{1,}"}]],
    "NUM": [[{"LIKE_NUM": True}, {"LIKE_NUM": True, "OP": "*"}]],
    "LEN": [[{"LENGTH": {">=": 7}}], [{"LENGTH": 3, "IS_ALPHA": True, "IS_LOWER": False}]],
    "REGEX": [[{"TEXT": {"REGEX": "^[Gg]oogle$"}}, {"LOWER": {"IN": ["maps", "search"]}}], [{"LOWER": {"REGEX": "colou?r"}}]],
    "SHAPE": [[{"SHAPE": "dd"}], [{"SHAPE": "Xxxxx", "IS_STOP": False}, {"SHAPE": "Xxxx"}]],
    "WILD": [[{"LOWER": "the"}, {}, {"LOWER": {"NOT_IN": ["the", "a"]}}]],
    "URL": [[{"LIKE_URL": True}], [{"LIKE_EMAIL": True}]],
    "FUZZY": [[{"LOWER": {"FUZZY": "colour"}}], [{"LOWER": {"FUZZY1": "apples"}}]],
    "PREFIX": [[{"PREFIX": "N", "IS_TITLE": True}, {"PREFIX": "Y"}]],
    "SET": [[{"LOWER": {"IS_SUBSET": ["a", "b"]}}, {"LOWER": {"INTERSECTS": ["c", "d"]}}], [{"NORM": {"IS_SUPERSET": ["e"]}}]],
    "PUNCTRUN": [[{"IS_PUNCT": True}, {"IS_PUNCT": True, "OP": "+"}]],
    "SPACES": [[{"IS_SPACE": True}], [{"SPACY": False, "IS_ALPHA": True}, {"IS_PUNCT": True}]],
}
greedy_keys = {"G_FIRST": ("FIRST", [[{"ORTH": "a"}, {"ORTH": "b", "OP": "*"}]]), "G_LONG": ("LONGEST", [[{"ORTH": {"IN": ["b", "c"]}, "OP": "+"}]]),
               "G_LONG2": ("LONGEST", [[{"LIKE_NUM": True, "OP": "+"}]]), "G_FIRST2": ("FIRST", [[{"IS_ALPHA": True, "OP": "+"}, {"IS_PUNCT": True}]])}
nlp = spacy.blank("en")
out = {"texts": texts, "matcher": {}, "phrase": {}, "entity_ruler": [], "sentencizer": [], "lemma_lookup": []}
for key, pats in patterns.items():
    m = Matcher(nlp.vocab, validate=False); m.add(key, pats)
    out["matcher"][key] = {"patterns": pats, "greedy": None, "matches": [[(s, e) for _, s, e in m(nlp(t))] for t in texts]}
for key, (g, pats) in greedy_keys.items():
    m = Matcher(nlp.vocab, validate=False); m.add(key, pats, greedy=g)
    out["matcher"][key] = {"patterns": pats, "greedy": g, "matches": [[(s, e) for _, s, e in m(nlp(t))] for t in texts]}
# combined matcher (ordering across keys)
m = Matcher(nlp.vocab, validate=False)
for key, pats in patterns.items():
    m.add(key, pats)
out["matcher_all"] = [[(nlp.vocab.strings[k], s, e) for k, s, e in m(nlp(t))] for t in texts]
phrases = {"orth": ["New York", "New York City", "Google", "Google Maps", "apple pie", "hello", "U.S.A.", "can't"],
           "lower": ["new york", "google maps", "apples", "hello world", "york"]}
for attr, plist in phrases.items():
    pm = PhraseMatcher(nlp.vocab, attr=attr.upper() if attr != "orth" else "ORTH")
    for p in plist:
        pm.add(p.upper(), [nlp.make_doc(p)])
    out["phrase"][attr] = {"phrases": plist, "matches": [[(nlp.vocab.strings[k], s, e) for k, s, e in pm(nlp(t))] for t in texts]}
ruler_patterns = [
    {"label": "GPE", "pattern": "New York"}, {"label": "GPE", "pattern": "New York City", "id": "nyc"},
    {"label": "ORG", "pattern": "Google"}, {"label": "PRODUCT", "pattern": [{"LOWER": "google"}, {"LOWER": "maps"}]},
    {"label": "ORG", "pattern": "Apple Inc."}, {"label": "MONEY", "pattern": [{"ORTH": "$"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}, "OP": "?"}]},
    {"label": "PERSON", "pattern": [{"ORTH": {"IN": ["Mr.", "Dr."]}}, {"IS_TITLE": True}]},
    {"label": "GPE", "pattern": "London", "id": "london"}, {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}], "id": "sf"},
]
for overwrite in (False, True):
    nlp2 = spacy.blank("en")
    r = nlp2.add_pipe("entity_ruler", config={"overwrite_ents": overwrite})
    r.add_patterns(ruler_patterns)
    out["entity_ruler"].append({"overwrite": overwrite, "ents": [[(e.start, e.end, e.label_, e.ent_id_) for e in nlp2(t).ents] for t in texts]})
out["ruler_patterns"] = ruler_patterns
nlp3 = spacy.blank("en"); nlp3.add_pipe("sentencizer")
out["sentencizer"] = [[s.start for s in nlp3(t).sents] for t in texts]
nlp4 = spacy.blank("en"); nlp4.add_pipe("lemmatizer", config={"mode": "lookup"}); nlp4.initialize()
out["lemma_lookup"] = [[t.lemma_ for t in nlp4(x)] for x in texts]
nlp5 = spacy.blank("en"); sr = nlp5.add_pipe("span_ruler"); sr.add_patterns(ruler_patterns)
def _sr(t):
    d = nlp5(t)
    return [(s.start, s.end, s.label_, s.id_) for s in d.spans["ruler"]]
out["span_ruler"] = [_sr(t) for t in texts]
json.dump(out, open(os.path.join(OUT, "rules_en.json"), "w", encoding="utf8"), ensure_ascii=False)
print("ok", len(texts))
