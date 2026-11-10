//! DocBin interoperability with spaCy.
use serde_json::{Value, json};

fn rows(doc: &spacy::Doc) -> Value {
    Value::Array(
        doc.iter()
            .map(|x| {
                json!([
                    x.text(),
                    x.whitespace(),
                    x.tag_(),
                    x.pos_(),
                    x.morph_(),
                    x.lemma_(),
                    x.dep_(),
                    x.head().i(),
                    x.ent_iob_(),
                    x.ent_type_(),
                    x.norm_()
                ])
            })
            .collect(),
    )
}

fn check(docs: &[spacy::Doc], exp: &Value) {
    let exp = exp.as_array().unwrap();
    assert_eq!(docs.len(), exp.len());
    for (d, e) in docs.iter().zip(exp) {
        assert_eq!(d.text(), e["text"].as_str().unwrap());
        assert_eq!(rows(d), e["tokens"]);
        let ents: Vec<Value> = d
            .ents()
            .iter()
            .map(|x| json!([x.start(), x.end(), x.label_()]))
            .collect();
        assert_eq!(Value::Array(ents), e["ents"]);
        let spans: Vec<Value> = d.spans["custom"]
            .iter()
            .map(|c| {
                let s = d.span_from(c);
                json!([s.start(), s.end(), s.label_(), s.kb_id_(), s.id_()])
            })
            .collect();
        assert_eq!(Value::Array(spans), e["spans"]);
        let sents: Vec<Value> = d.sents().iter().map(|s| json!(s.start())).collect();
        assert_eq!(Value::Array(sents), e["sents"]);
    }
}

#[test]
fn read_spacy_docbin_and_roundtrip() {
    let dir = env!("CARGO_MANIFEST_DIR");
    if !std::path::Path::new(&format!("{dir}/tests/fixtures/sample.spacy")).exists() {
        eprintln!("skipping: fixtures not found");
        return;
    }
    let bin = spacy::DocBin::from_disk(format!("{dir}/tests/fixtures/sample.spacy")).unwrap();
    let exp: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/tests/fixtures/sample_docbin.json")).unwrap(),
    )
    .unwrap();
    let nlp = spacy::blank("en").unwrap();
    let docs = bin.get_docs(nlp.vocab()).unwrap();
    check(&docs, &exp);
    // round trip through our own writer
    let mut out = spacy::DocBin::default();
    for d in &docs {
        out.add(d);
    }
    let bytes = out.to_bytes();
    if let Ok(p) = std::env::var("SPACY_RS_DOCBIN_OUT") {
        std::fs::write(p, &bytes).unwrap();
    }
    let back = spacy::DocBin::from_bytes(&bytes).unwrap();
    let nlp2 = spacy::blank("en").unwrap();
    let docs2 = back.get_docs(nlp2.vocab()).unwrap();
    check(&docs2, &exp);
}
