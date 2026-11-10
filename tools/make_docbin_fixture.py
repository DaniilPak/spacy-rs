"""DocBin fixtures: spaCy writes, spacy-rs reads (and vice versa via verify mode)."""
import json, sys
import spacy
from spacy.tokens import DocBin, Span
mode = sys.argv[1]
nlp = spacy.load("en_core_web_sm")
if mode == "write":
    texts = ["Apple is looking at buying U.K. startup for $1 billion.", "San Francisco considers banning sidewalk delivery robots",
             "Tabs\tand   spaces here.  ", "Mr. Smith met Dr. Jones at 5 p.m. in St. Louis. It rained."]
    db = DocBin(store_user_data=False)
    exp = []
    for t in texts:
        d = nlp(t)
        d.spans["custom"] = [Span(d, 0, 2, label="FIRST_TWO", kb_id="kb1", span_id="id1")]
        db.add(d)
        exp.append({"text": d.text, "tokens": [[x.text, x.whitespace_, x.tag_, x.pos_, str(x.morph), x.lemma_, x.dep_, x.head.i, x.ent_iob_, x.ent_type_, x.norm_] for x in d],
                    "ents": [[e.start, e.end, e.label_] for e in d.ents], "spans": [[s.start, s.end, s.label_, s.kb_id_, s.id_] for s in d.spans["custom"]],
                    "sents": [s.start for s in d.sents]})
    db.to_disk(sys.argv[2])
    json.dump(exp, open(sys.argv[3], "w"), ensure_ascii=False)
else:
    db = DocBin().from_disk(sys.argv[2])
    exp = json.load(open(sys.argv[3]))
    docs = list(db.get_docs(nlp.vocab))
    assert len(docs) == len(exp), (len(docs), len(exp))
    for d, e in zip(docs, exp):
        got = [[x.text, x.whitespace_, x.tag_, x.pos_, str(x.morph), x.lemma_, x.dep_, x.head.i, x.ent_iob_, x.ent_type_, x.norm_] for x in d]
        assert d.text == e["text"], (d.text, e["text"])
        assert got == e["tokens"], (got, e["tokens"])
        assert [[x.start, x.end, x.label_] for x in d.ents] == e["ents"]
        assert [[s.start, s.end, s.label_, s.kb_id_, s.id_] for s in d.spans["custom"]] == e["spans"]
        assert [s.start for s in d.sents] == e["sents"]
    print("python read rust DocBin OK:", len(docs), "docs")
