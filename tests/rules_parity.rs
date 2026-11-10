//! Parity of rule-based components with spaCy 3.8.
use serde_json::Value;
use spacy::matcher::{Greedy, Matcher, PhraseMatcher};
use spacy::pipeline::{EntityRuler, Lemmatizer, LemmatizerMode, RulerPattern, Sentencizer, SpanRuler};
use spacy::{Language, Pipe};

fn fixture() -> Value {
    let p = format!("{}/tests/fixtures/rules_en.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn texts(f: &Value) -> Vec<String> {
    f["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect()
}

fn pairs(v: &Value) -> Vec<(usize, usize)> {
    v.as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)).collect()
}

#[test]
fn matcher_per_key() {
    let f = fixture();
    let texts = texts(&f);
    let nlp = spacy::blank("en").unwrap();
    let mut fails = vec![];
    for (key, spec) in f["matcher"].as_object().unwrap() {
        let mut m = Matcher::new(nlp.vocab());
        let greedy = match spec["greedy"].as_str() {
            Some("FIRST") => Some(Greedy::First),
            Some("LONGEST") => Some(Greedy::Longest),
            _ => None,
        };
        m.add(key, spec["patterns"].as_array().unwrap(), greedy).unwrap();
        for (t, want) in texts.iter().zip(spec["matches"].as_array().unwrap()) {
            let doc = nlp.make_doc(t);
            let got: Vec<(usize, usize)> = m.find(&doc).unwrap().iter().map(|m| (m.start, m.end)).collect();
            if got != pairs(want) {
                fails.push(format!("{key} {t:?}\n want {:?}\n got  {got:?}", pairs(want)));
            }
        }
    }
    assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn matcher_all_keys_order() {
    let f = fixture();
    let texts = texts(&f);
    let nlp = spacy::blank("en").unwrap();
    let mut m = Matcher::new(nlp.vocab());
    for (key, spec) in f["matcher"].as_object().unwrap() {
        if spec["greedy"].is_null() {
            m.add(key, spec["patterns"].as_array().unwrap(), None).unwrap();
        }
    }
    let mut fails = vec![];
    for (t, want) in texts.iter().zip(f["matcher_all"].as_array().unwrap()) {
        let doc = nlp.make_doc(t);
        let got: Vec<(String, usize, usize)> = m
            .find(&doc)
            .unwrap()
            .iter()
            .map(|x| (nlp.vocab().strings.get(x.match_id).unwrap().to_string(), x.start, x.end))
            .collect();
        let want: Vec<(String, usize, usize)> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_u64().unwrap() as usize, p[2].as_u64().unwrap() as usize))
            .collect();
        if got != want {
            fails.push(format!("{t:?}\n want {want:?}\n got  {got:?}"));
        }
    }
    assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn phrase_matcher() {
    let f = fixture();
    let texts = texts(&f);
    let nlp = spacy::blank("en").unwrap();
    for (attr, spec) in f["phrase"].as_object().unwrap() {
        let a = if attr == "lower" { spacy::attrs::LOWER } else { spacy::attrs::ORTH };
        let mut pm = PhraseMatcher::with_attr(nlp.vocab(), a);
        for p in spec["phrases"].as_array().unwrap() {
            let p = p.as_str().unwrap();
            pm.add(&p.to_uppercase(), &[nlp.make_doc(p)]).unwrap();
        }
        for (t, want) in texts.iter().zip(spec["matches"].as_array().unwrap()) {
            let doc = nlp.make_doc(t);
            let got: Vec<(String, usize, usize)> =
                pm.find(&doc).iter().map(|x| (nlp.vocab().strings.get(x.match_id).unwrap().to_string(), x.start, x.end)).collect();
            let want: Vec<(String, usize, usize)> = want
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_u64().unwrap() as usize, p[2].as_u64().unwrap() as usize))
                .collect();
            assert_eq!(got, want, "{attr} {t:?}");
        }
    }
}

fn ruler_patterns(f: &Value) -> Vec<RulerPattern> {
    serde_json::from_value(f["ruler_patterns"].clone()).unwrap()
}

#[test]
fn entity_ruler() {
    let f = fixture();
    let texts = texts(&f);
    for case in f["entity_ruler"].as_array().unwrap() {
        let mut nlp = spacy::blank("en").unwrap();
        let mut ruler = EntityRuler::with_options(&nlp, spacy::attrs::ORTH, case["overwrite"].as_bool().unwrap());
        ruler.add_patterns(&nlp, &ruler_patterns(&f)).unwrap();
        nlp.add_pipe("entity_ruler", ruler);
        for (t, want) in texts.iter().zip(case["ents"].as_array().unwrap()) {
            let doc = nlp.call(t).unwrap();
            let got: Vec<Value> = doc
                .ents()
                .iter()
                .map(|e| serde_json::json!([e.start(), e.end(), e.label_(), e.id_()]))
                .collect();
            assert_eq!(&Value::Array(got), want, "{t:?}");
        }
    }
}

#[test]
fn span_ruler() {
    let f = fixture();
    let texts = texts(&f);
    let mut nlp = spacy::blank("en").unwrap();
    let mut ruler = SpanRuler::new(&nlp);
    ruler.add_patterns(&nlp, &ruler_patterns(&f)).unwrap();
    nlp.add_pipe("span_ruler", ruler);
    for (t, want) in texts.iter().zip(f["span_ruler"].as_array().unwrap()) {
        let doc = nlp.call(t).unwrap();
        let got: Vec<Value> = doc.spans["ruler"]
            .iter()
            .map(|c| {
                let s = doc.span_from(c);
                serde_json::json!([s.start(), s.end(), s.label_(), s.id_()])
            })
            .collect();
        assert_eq!(&Value::Array(got), want, "{t:?}");
    }
}

#[test]
fn sentencizer() {
    let f = fixture();
    let texts = texts(&f);
    let nlp = spacy::blank("en").unwrap();
    let s = Sentencizer::new();
    for (t, want) in texts.iter().zip(f["sentencizer"].as_array().unwrap()) {
        let mut doc = nlp.make_doc(t);
        s.process(&mut doc).unwrap();
        let got: Vec<usize> = doc.sents().iter().map(|s| s.start()).collect();
        let want: Vec<usize> = want.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(got, want, "{t:?}");
    }
}

#[test]
fn lookup_lemmatizer() {
    let f = fixture();
    let texts = texts(&f);
    let mut nlp: Language = spacy::blank("en").unwrap();
    nlp.add_pipe("lemmatizer", Lemmatizer::english(LemmatizerMode::Lookup));
    for (t, want) in texts.iter().zip(f["lemma_lookup"].as_array().unwrap()) {
        let doc = nlp.call(t).unwrap();
        let got: Vec<&str> = doc.iter().map(|t| t.lemma_()).collect();
        let want: Vec<&str> = want.as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        assert_eq!(got, want, "{t:?}");
    }
}
