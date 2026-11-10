//! Throughput benchmark: `cargo run --release --example bench -- [model_dir]`.
use std::time::Instant;

fn main() {
    let sun = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/corpus/en_news.txt"
    ))
    .unwrap();
    let paras: Vec<String> = sun.split("\n\n").map(str::to_string).collect();
    let texts: Vec<String> = (0..200).flat_map(|_| paras.clone()).collect();
    let n_chars: usize = texts.iter().map(|t| t.len()).sum();
    let nlp = spacy::blank("en").unwrap();
    let t = Instant::now();
    let mut n_tok = 0;
    for x in &texts {
        n_tok += nlp.make_doc(x).len();
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "tokenizer (1 thread): {} docs, {n_tok} tokens, {:.0} tokens/s, {:.1} MB/s",
        texts.len(),
        n_tok as f64 / dt,
        n_chars as f64 / dt / 1e6
    );
    let t = Instant::now();
    let docs = nlp.tokenizer.pipe(&texts);
    let dt = t.elapsed().as_secs_f64();
    println!(
        "tokenizer (rayon): {:.0} tokens/s",
        docs.iter().map(|d| d.len()).sum::<usize>() as f64 / dt
    );
    if let Some(p) = std::env::args().nth(1) {
        let nlp = spacy::load(p).unwrap();
        let sub = &texts[..1000];
        let t = Instant::now();
        let mut n = 0;
        for x in sub {
            n += nlp.call(x).unwrap().len();
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "en_core_web_sm (1 thread): {} docs, {:.0} tokens/s, {:.0} docs/s",
            sub.len(),
            n as f64 / dt,
            sub.len() as f64 / dt
        );
        let t = Instant::now();
        let docs = nlp.pipe(sub, 64).unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!(
            "en_core_web_sm (rayon): {:.0} tokens/s",
            docs.iter().map(|d| d.len()).sum::<usize>() as f64 / dt
        );
    }
}
