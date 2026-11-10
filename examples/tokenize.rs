fn main() {
    let nlp = spacy::blank("en").unwrap();
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "I can't visit https://example.com, e.g. today!!! 😀 U.S.A. :)".into());
    let doc = nlp.make_doc(&text);
    for t in doc.iter() {
        println!(
            "{:>3} {:<20} {:<12} {:<8} ws={:?} stop={} num={} url={}",
            t.idx(),
            t.text(),
            t.norm_(),
            t.shape_(),
            t.whitespace(),
            t.is_stop(),
            t.like_num(),
            t.like_url()
        );
    }
}
