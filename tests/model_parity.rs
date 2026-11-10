//! Annotation parity with spaCy's `en_core_web_sm` 3.8.0.
//!
//! Needs the unpacked pipeline at `tests/models/en_core_web_sm` (see
//! `tools/fetch_model.sh`) or `$SPACY_RS_MODEL`; skipped otherwise.
use serde_json::Value;
use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SPACY_RS_MODEL") {
        return Some(PathBuf::from(p));
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/models/en_core_web_sm");
    p.exists().then_some(p)
}

#[test]
fn en_core_web_sm_parity() {
    let Some(path) = model_path() else {
        eprintln!("skipping: en_core_web_sm not found (run tools/fetch_model.sh)");
        return;
    };
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/model_en_core_web_sm.json"
    );
    if !std::path::Path::new(fixture).exists() {
        eprintln!("skipping: {fixture} not found");
        return;
    }
    let nlp = spacy::load(&path).expect("load model");
    let f: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/model_en_core_web_sm.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let names = [
        "text",
        "tag",
        "pos",
        "morph",
        "lemma",
        "dep",
        "head",
        "ent_iob",
        "ent_type",
        "sent_start",
    ];
    let mut total = 0usize;
    let mut wrong = vec![0usize; names.len()];
    let mut ents_ok = 0;
    let mut docs_ok = 0;
    let mut wrong_chunks = 0;
    let mut wrong_sents = 0;
    let docs = f["docs"].as_array().unwrap();
    let mut report = Vec::new();
    for d in docs {
        let doc = nlp.call(d["text"].as_str().unwrap()).unwrap();
        let want = d["tokens"].as_array().unwrap();
        assert_eq!(
            doc.len(),
            want.len(),
            "tokenization differs for {:?}",
            d["text"]
        );
        let mut doc_ok = true;
        for (t, w) in doc.iter().zip(want) {
            let got = serde_json::json!([
                t.text(),
                t.tag_(),
                t.pos_(),
                t.morph_(),
                t.lemma_(),
                t.dep_(),
                t.head().i(),
                t.ent_iob_(),
                t.ent_type_(),
                t.is_sent_start().unwrap_or(false) as u8
            ]);
            total += 1;
            for k in 0..names.len() {
                if got[k] != w[k] {
                    wrong[k] += 1;
                    doc_ok = false;
                    if report.len() < 30 {
                        report.push(format!(
                            "{} of {:?}: want {} got {}",
                            names[k],
                            t.text(),
                            w[k],
                            got[k]
                        ));
                    }
                }
            }
        }
        let got_ents: Vec<Value> = doc
            .ents()
            .iter()
            .map(|e| serde_json::json!([e.start(), e.end(), e.label_()]))
            .collect();
        let chunks: Vec<Value> = doc
            .noun_chunks()
            .unwrap()
            .iter()
            .map(|c| serde_json::json!([c.start(), c.end()]))
            .collect();
        if Value::Array(chunks) != d["noun_chunks"] {
            doc_ok = false;
            wrong_chunks += 1;
        }
        let sents: Vec<Value> = doc
            .sents()
            .iter()
            .map(|s| serde_json::json!(s.start()))
            .collect();
        if Value::Array(sents) != d["sents"] {
            doc_ok = false;
            wrong_sents += 1;
        }
        if Value::Array(got_ents) == d["ents"] {
            ents_ok += 1;
        } else {
            doc_ok = false;
        }
        docs_ok += doc_ok as usize;
    }
    let acc: Vec<String> = names
        .iter()
        .zip(&wrong)
        .map(|(n, w)| format!("{n}: {:.3}%", 100.0 * (total - w) as f64 / total as f64))
        .collect();
    eprintln!(
        "tokens={total} docs={} exact_docs={docs_ok} ent_docs={ents_ok} chunk_mismatch_docs={wrong_chunks} sent_mismatch_docs={wrong_sents}\n{}\n{}",
        docs.len(),
        acc.join("  "),
        report.join("\n")
    );
    assert!(
        wrong_chunks * 100 <= docs.len(),
        "noun chunks differ in {wrong_chunks} docs"
    );
    assert!(
        ents_ok * 100 >= docs.len() * 99,
        "entities differ in {} docs",
        docs.len() - ents_ok
    );
    // Floating-point summation order differs from BLAS, so allow rare flips.
    for (n, w) in names.iter().zip(&wrong) {
        assert!(
            (*w as f64) / (total as f64) < 0.005,
            "{n}: {w} mismatches of {total}"
        );
    }
}
