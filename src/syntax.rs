//! Syntax iterators (`doc.noun_chunks`), ported from `spacy/lang/*/syntax_iterators.py`.

use crate::Error;
use crate::attrs::pos::{NOUN, PRON, PROPN};
use crate::doc::{Doc, Span, SpanC};

/// English base noun phrases (`spacy.lang.en.syntax_iterators.noun_chunks`).
pub fn noun_chunks_en(doc: &Doc) -> Result<Vec<SpanC>, Error> {
    if !doc.has_annotation(crate::attrs::DEP) {
        return Err(Error::Value("[E029] noun_chunks requires the dependency parse, which requires a statistical model to be installed and loaded.".into()));
    }
    let st = &doc.vocab().strings;
    let np_deps: Vec<u64> = [
        "oprd",
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    .iter()
    .map(|l| st.add(l))
    .collect();
    let conj = st.add("conj");
    let np_label = st.add("NP");
    let mut out = Vec::new();
    let mut prev_end: isize = -1;
    for w in doc.iter() {
        let c = w.c();
        if ![NOUN, PROPN, PRON].contains(&c.pos) {
            continue;
        }
        if w.left_edge().i() as isize <= prev_end {
            continue;
        }
        let mut push = |w: &crate::doc::Token<'_>| {
            prev_end = w.i() as isize;
            let mut s = SpanC::new(w.left_edge().i(), w.i() + 1);
            s.label = np_label;
            out.push(s);
        };
        if np_deps.contains(&c.dep) {
            push(&w);
        } else if c.dep == conj {
            let mut head = w.head();
            let mut guard = 0;
            while head.c().dep == conj && head.head().i() < head.i() && guard < doc.len() {
                head = head.head();
                guard += 1;
            }
            if np_deps.contains(&head.c().dep) {
                push(&w);
            }
        }
    }
    Ok(out)
}

impl Doc {
    /// Base noun phrases (`doc.noun_chunks`). Requires a dependency parse.
    /// Supported for English; other languages return an error.
    pub fn noun_chunks(&self) -> Result<Vec<Span<'_>>, Error> {
        match self.vocab().lang() {
            "en" => Ok(noun_chunks_en(self)?
                .iter()
                .map(|c| self.span_from(c))
                .collect()),
            other => Err(Error::Value(format!(
                "[E894] The 'noun_chunks' syntax iterator is not implemented for language '{other}'."
            ))),
        }
    }
}
