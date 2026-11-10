//! `SpanRuler`: add spans to `doc.spans[key]` (and optionally `doc.ents`).

use super::{RulerPattern, filter_spans};
use crate::Error;
use crate::attrs::*;
use crate::doc::{Doc, SpanC};
use crate::language::{Language, Pipe};
use crate::matcher::{Matcher, PhraseMatcher};
use crate::vocab::Vocab;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(crate) fn needs_annotation(attr: u64) -> bool {
    matches!(attr, TAG | POS | MORPH | LEMMA | DEP)
}

/// Rule-based span annotation, like `nlp.add_pipe("span_ruler")`.
#[derive(Clone, Debug)]
pub struct SpanRuler {
    vocab: Arc<Vocab>,
    matcher: Matcher,
    phrase_matcher: PhraseMatcher,
    /// Key in `doc.spans` (None to skip).
    pub spans_key: Option<String>,
    /// Also set `doc.ents` (with the first-longest filter).
    pub annotate_ents: bool,
    /// Overwrite existing spans / entities.
    pub overwrite: bool,
    /// Filter overlapping spans in `doc.spans` with `filter_spans`.
    pub filter_spans: bool,
    label_map: HashMap<u64, (String, String)>,
    patterns: Vec<RulerPattern>,
    phrase_matcher_attr: u64,
}

impl SpanRuler {
    /// A ruler writing to `doc.spans["ruler"]`.
    pub fn new(nlp: &Language) -> Self {
        let vocab = nlp.vocab().clone();
        SpanRuler {
            matcher: Matcher::new(&vocab),
            phrase_matcher: PhraseMatcher::new(&vocab),
            vocab,
            spans_key: Some("ruler".into()),
            annotate_ents: false,
            overwrite: true,
            filter_spans: false,
            label_map: HashMap::new(),
            patterns: Vec::new(),
            phrase_matcher_attr: ORTH,
        }
    }

    /// Set the phrase matcher attribute (call before adding patterns).
    pub fn set_phrase_matcher_attr(&mut self, attr: u64) {
        self.phrase_matcher_attr = attr;
        self.phrase_matcher = PhraseMatcher::with_attr(&self.vocab, attr);
    }

    /// All patterns.
    pub fn patterns(&self) -> &[RulerPattern] {
        &self.patterns
    }

    /// Add patterns.
    pub fn add_patterns(&mut self, nlp: &Language, patterns: &[RulerPattern]) -> Result<(), Error> {
        for p in patterns {
            let id = p.id.clone().unwrap_or_default();
            let key = format!("('{}', '{}')", p.label, id);
            let h = self.vocab.strings.add(&key);
            self.label_map.insert(h, (p.label.clone(), id));
            match &p.pattern {
                serde_json::Value::String(text) => {
                    let doc = if needs_annotation(self.phrase_matcher_attr) {
                        nlp.call(text)?
                    } else {
                        nlp.make_doc(text)
                    };
                    self.phrase_matcher.add(&key, &[doc])?;
                }
                serde_json::Value::Array(_) => {
                    self.matcher
                        .add(&key, std::slice::from_ref(&p.pattern), None)?
                }
                other => return Err(Error::Value(format!("[E097] Invalid pattern: {other}"))),
            }
            self.patterns.push(p.clone());
        }
        Ok(())
    }

    /// Deduplicated, sorted matched spans.
    pub fn find_spans(&self, doc: &Doc) -> Vec<SpanC> {
        let mut m: Vec<(u64, usize, usize)> = if self.matcher.is_empty() {
            Vec::new()
        } else {
            self.matcher
                .find_allow_missing(doc)
                .into_iter()
                .map(|m| (m.match_id, m.start, m.end))
                .collect()
        };
        m.extend(
            self.phrase_matcher
                .find(doc)
                .into_iter()
                .map(|m| (m.match_id, m.start, m.end)),
        );
        let mut seen = HashSet::new();
        let mut out: Vec<SpanC> = Vec::new();
        for (mid, s, e) in m {
            if s == e {
                continue;
            }
            let (label, id) = self.label_map.get(&mid).cloned().unwrap_or_default();
            let mut c = SpanC::new(s, e);
            c.label = self.vocab.strings.add(&label);
            c.id = self.vocab.strings.add(&id);
            if seen.insert(c.clone()) {
                out.push(c);
            }
        }
        let st = &self.vocab.strings;
        out.sort_by(|a, b| {
            (
                a.start,
                a.end,
                st.get_or_empty(a.label),
                st.get_or_empty(a.id),
            )
                .cmp(&(
                    b.start,
                    b.end,
                    st.get_or_empty(b.label),
                    st.get_or_empty(b.id),
                ))
        });
        out
    }
}

impl Pipe for SpanRuler {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        let matches = self.find_spans(doc);
        if let Some(key) = &self.spans_key {
            let mut spans = if self.overwrite {
                Vec::new()
            } else {
                doc.spans.get(key).cloned().unwrap_or_default()
            };
            if self.filter_spans {
                let mut all = spans.clone();
                all.extend(matches.iter().cloned());
                spans = filter_spans(&all);
            } else {
                spans.extend(matches.iter().cloned());
            }
            doc.spans.insert(key.clone(), spans);
        }
        if self.annotate_ents {
            let mut spans = if self.overwrite {
                Vec::new()
            } else {
                doc.ent_spans()
            };
            spans.extend(matches);
            let spans = filter_spans(&spans);
            doc.set_ents(&spans)?;
        }
        Ok(())
    }
}
