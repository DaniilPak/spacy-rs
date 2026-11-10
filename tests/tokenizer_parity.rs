//! Exact token/attribute parity with spaCy 3.8 (`spacy.blank(lang)`).
use serde_json::Value;

const FIELDS: [&str; 25] = [
    "text",
    "idx",
    "whitespace_",
    "lower_",
    "norm_",
    "shape_",
    "prefix_",
    "suffix_",
    "is_alpha",
    "is_ascii",
    "is_digit",
    "is_lower",
    "is_upper",
    "is_title",
    "is_punct",
    "is_left_punct",
    "is_right_punct",
    "is_space",
    "is_bracket",
    "is_quote",
    "is_currency",
    "like_url",
    "like_num",
    "like_email",
    "is_stop",
];

fn row(t: &spacy::Token) -> Vec<Value> {
    use serde_json::json;
    vec![
        json!(t.text()),
        json!(t.idx()),
        json!(t.whitespace()),
        json!(t.lower_()),
        json!(t.norm_()),
        json!(t.shape_()),
        json!(t.prefix_()),
        json!(t.suffix_()),
        json!(t.is_alpha() as u8),
        json!(t.is_ascii() as u8),
        json!(t.is_digit() as u8),
        json!(t.is_lower() as u8),
        json!(t.is_upper() as u8),
        json!(t.is_title() as u8),
        json!(t.is_punct() as u8),
        json!(t.is_left_punct() as u8),
        json!(t.is_right_punct() as u8),
        json!(t.is_space() as u8),
        json!(t.is_bracket() as u8),
        json!(t.is_quote() as u8),
        json!(t.is_currency() as u8),
        json!(t.like_url() as u8),
        json!(t.like_num() as u8),
        json!(t.like_email() as u8),
        json!(t.is_stop() as u8),
    ]
}

fn check(lang: &str) {
    let path = format!(
        "{}/tests/fixtures/tokens_{lang}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let nlp = spacy::blank(lang.split('_').next().unwrap()).unwrap();
    let mut failures = Vec::new();
    let docs = data["docs"].as_array().unwrap();
    for d in docs {
        let text = d["text"].as_str().unwrap();
        let doc = nlp.make_doc(text);
        let want = d["tokens"].as_array().unwrap();
        assert!(!want.is_empty() || doc.is_empty());
        let got: Vec<Vec<Value>> = doc.iter().map(|t| row(&t)).collect();
        let want_texts: Vec<&str> = want.iter().map(|r| r[0].as_str().unwrap()).collect();
        let got_texts: Vec<&str> = doc.iter().map(|t| t.text()).collect();
        if want_texts != got_texts {
            failures.push(format!(
                "TOKENS {text:?}\n  want {want_texts:?}\n  got  {got_texts:?}"
            ));
            continue;
        }
        for (w, g) in want.iter().zip(&got) {
            let w = w.as_array().unwrap();
            for (k, (a, b)) in w.iter().zip(g).enumerate() {
                if a != b {
                    failures.push(format!(
                        "ATTR {} of {:?} in {text:?}: want {a} got {b}",
                        FIELDS[k], w[0]
                    ));
                }
            }
        }
    }
    if !failures.is_empty() {
        let n = failures.len();
        let shown: Vec<_> = failures.into_iter().take(40).collect();
        panic!(
            "{lang}: {n} mismatches over {} docs\n{}",
            docs.len(),
            shown.join("\n")
        );
    }
}

#[test]
fn english() {
    check("en");
}
#[test]
fn german() {
    check("de");
}
#[test]
fn french() {
    check("fr");
}
#[test]
fn spanish() {
    check("es");
}
#[test]
fn italian() {
    check("it");
}
#[test]
fn portuguese() {
    check("pt");
}
#[test]
fn dutch() {
    check("nl");
}
#[test]
fn multi_language() {
    check("xx");
}
#[test]
fn english_fuzz() {
    check("en_fuzz");
}
#[test]
fn french_fuzz() {
    check("fr_fuzz");
}
#[test]
fn german_fuzz() {
    check("de_fuzz");
}
