use super::Match;
use crate::Error;
use crate::attrs::*;
use crate::doc::{Doc, Span, SpanC, token_attr};
use crate::vocab::Vocab;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default, Clone, Debug)]
struct Node {
    children: HashMap<u64, usize>,
    terminal: Vec<u64>,
}

/// Efficient phrase matching on a token attribute (default `ORTH`), like
/// `spacy.matcher.PhraseMatcher`.
#[derive(Clone)]
pub struct PhraseMatcher {
    vocab: Arc<Vocab>,
    attr: u64,
    nodes: Vec<Node>,
    docs: HashMap<u64, Vec<Vec<u64>>>,
    keys: Vec<u64>,
    callbacks: HashMap<u64, super::OnMatch>,
}

impl std::fmt::Debug for PhraseMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PhraseMatcher(attr={}, {} keys)",
            self.attr,
            self.keys.len()
        )
    }
}

impl PhraseMatcher {
    /// Match on `ORTH`.
    pub fn new(vocab: &Arc<Vocab>) -> Self {
        Self::with_attr(vocab, ORTH)
    }

    /// Match on another attribute, e.g. [`LOWER`] or [`NORM`].
    pub fn with_attr(vocab: &Arc<Vocab>, attr: u64) -> Self {
        let attr = if attr == IDS_TEXT { ORTH } else { attr };
        PhraseMatcher {
            vocab: vocab.clone(),
            attr,
            nodes: vec![Node::default()],
            docs: HashMap::new(),
            keys: Vec::new(),
            callbacks: HashMap::new(),
        }
    }

    /// The attribute matched on.
    pub fn attr(&self) -> u64 {
        self.attr
    }
    /// Number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Whether a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains(&self.vocab.strings.add(key))
    }

    /// Add phrase patterns given as docs.
    pub fn add(&mut self, key: &str, docs: &[Doc]) -> Result<(), Error> {
        let key_id = self.vocab.strings.add(key);
        for d in docs {
            if d.is_empty() {
                continue;
            }
            for (attr, pipe, name) in [
                (TAG, "tagger", "TAG"),
                (POS, "morphologizer or tagger+attribute_ruler", "POS"),
                (MORPH, "morphologizer or tagger+attribute_ruler", "MORPH"),
                (LEMMA, "lemmatizer", "LEMMA"),
                (DEP, "parser", "DEP"),
            ] {
                if self.attr == attr && !d.has_annotation(attr) {
                    return Err(Error::Value(format!(
                        "[E155] The pipeline needs to include a {pipe} in order to use Matcher or PhraseMatcher with the attribute {name}."
                    )));
                }
            }
            let kw: Vec<u64> = d.tokens.iter().map(|t| token_attr(t, self.attr)).collect();
            self.add_keyword(key_id, kw);
        }
        if !self.keys.contains(&key_id) {
            self.keys.push(key_id);
        }
        Ok(())
    }

    /// Add phrase patterns given as attribute-value sequences.
    pub fn add_ids(&mut self, key: &str, phrases: &[Vec<u64>]) {
        let key_id = self.vocab.strings.add(key);
        for p in phrases {
            if !p.is_empty() {
                self.add_keyword(key_id, p.clone());
            }
        }
        if !self.keys.contains(&key_id) {
            self.keys.push(key_id);
        }
    }

    /// Add with an `on_match` callback (run by [`PhraseMatcher::apply`]).
    pub fn add_with_callback(
        &mut self,
        key: &str,
        docs: &[Doc],
        on_match: super::OnMatch,
    ) -> Result<(), Error> {
        self.add(key, docs)?;
        self.callbacks.insert(self.vocab.strings.add(key), on_match);
        Ok(())
    }

    fn add_keyword(&mut self, key: u64, kw: Vec<u64>) {
        let entry = self.docs.entry(key).or_default();
        if entry.contains(&kw) {
            return;
        }
        entry.push(kw.clone());
        let mut cur = 0;
        for t in kw {
            cur = match self.nodes[cur].children.get(&t) {
                Some(&n) => n,
                None => {
                    self.nodes.push(Node::default());
                    let n = self.nodes.len() - 1;
                    self.nodes[cur].children.insert(t, n);
                    n
                }
            };
        }
        if !self.nodes[cur].terminal.contains(&key) {
            self.nodes[cur].terminal.push(key);
        }
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Result<(), Error> {
        let id = self.vocab.strings.add(key);
        if !self.keys.contains(&id) {
            return Err(Error::Value(format!(
                "[E175] Can't remove rule for unknown match pattern ID: {key}"
            )));
        }
        for n in self.nodes.iter_mut() {
            n.terminal.retain(|k| *k != id);
        }
        self.keys.retain(|k| *k != id);
        self.docs.remove(&id);
        self.callbacks.remove(&id);
        Ok(())
    }

    fn find_range(&self, doc: &Doc, start_idx: usize, end_idx: usize) -> Vec<Match> {
        let mut out = Vec::new();
        let vals: Vec<u64> = doc.tokens[start_idx..end_idx]
            .iter()
            .map(|t| token_attr(t, self.attr))
            .collect();
        for s in 0..vals.len() {
            let mut cur = 0;
            for (k, v) in vals[s..].iter().enumerate() {
                match self.nodes[cur].children.get(v) {
                    Some(&n) => {
                        cur = n;
                        for &key in &self.nodes[cur].terminal {
                            out.push(Match {
                                match_id: key,
                                start: start_idx + s,
                                end: start_idx + s + k + 1,
                            });
                        }
                    }
                    None => break,
                }
            }
        }
        out
    }

    /// Find matches in a doc.
    pub fn find(&self, doc: &Doc) -> Vec<Match> {
        self.find_range(doc, 0, doc.len())
    }

    /// Find matches in a span (absolute token offsets, as in spaCy).
    pub fn find_in_span(&self, span: &Span<'_>) -> Vec<Match> {
        self.find_range(span.doc(), span.start(), span.end())
    }

    /// Matches as labelled spans.
    pub fn find_spans<'d>(&self, doc: &'d Doc) -> Vec<Span<'d>> {
        self.find(doc)
            .into_iter()
            .map(|m| {
                let mut c = SpanC::new(m.start, m.end);
                c.label = m.match_id;
                doc.span_from(&c)
            })
            .collect()
    }

    /// Find matches and run callbacks.
    pub fn apply(&self, doc: &mut Doc) -> Vec<Match> {
        let matches = self.find(doc);
        for (i, m) in matches.iter().enumerate() {
            if let Some(cb) = self.callbacks.get(&m.match_id) {
                cb(doc, i, &matches);
            }
        }
        matches
    }
}

const IDS_TEXT: u64 = u64::MAX;
