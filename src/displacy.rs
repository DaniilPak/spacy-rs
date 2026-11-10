//! displaCy-style HTML rendering of named entities (`displacy.render(doc, style="ent")`).

use crate::doc::Doc;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Default entity colours used by displaCy.
pub fn default_color(label: &str) -> &'static str {
    match label {
        "ORG" => "#7aecec",
        "PRODUCT" => "#bfeeb7",
        "GPE" => "#feca74",
        "LOC" => "#ff9561",
        "PERSON" => "#aa9cfc",
        "NORP" => "#c887fb",
        "FAC" => "#9cc9cc",
        "EVENT" => "#ffeb80",
        "LAW" => "#ff8197",
        "LANGUAGE" => "#ff8197",
        "WORK_OF_ART" => "#f0d0ff",
        "DATE" | "TIME" | "MONEY" | "QUANTITY" | "ORDINAL" | "CARDINAL" | "PERCENT" => "#bfe1d9",
        _ => "#ddd",
    }
}

/// Render entities as an HTML fragment.
pub fn render_ents(doc: &Doc) -> String {
    let text = doc.text();
    let mut out =
        String::from("<div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">");
    let mut last = 0usize;
    for e in doc.ents() {
        let a = doc.tokens[e.start()].byte_idx;
        let t = &doc.tokens[e.end() - 1];
        let b = t.byte_idx + t.byte_len;
        out.push_str(&esc(&text[last..a]).replace('\n', "\n<br>"));
        let label = e.label_();
        out.push_str(&format!(
            "<mark class=\"entity\" style=\"background: {}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">{}<span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">{}</span></mark>",
            default_color(label),
            esc(&text[a..b]),
            esc(label)
        ));
        last = b;
    }
    out.push_str(&esc(&text[last..]).replace('\n', "\n<br>"));
    out.push_str("</div>");
    out
}

/// Render a full standalone HTML page for several docs.
pub fn render_page(docs: &[Doc]) -> String {
    let body: Vec<String> = docs.iter().map(render_ents).collect();
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>displaCy</title></head><body style=\"font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 4rem 2rem; direction: ltr\">{}</body></html>",
        body.join("\n")
    )
}
