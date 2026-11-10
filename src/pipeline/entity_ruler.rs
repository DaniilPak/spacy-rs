//! `EntityRuler`: add named entities from token and phrase patterns.

use super::RulerPattern;
use crate::Error;
use crate::attrs::ORTH;
use crate::doc::{Doc, SpanC};
use crate::language::{Language, Pipe};
use crate::matcher::{Matcher, PhraseMatcher};
use crate::vocab::Vocab;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Default separator between label and ID in match keys.
pub const DEFAULT_ENT_ID_SEP: &str = "||";

/// Rule-based named entity recognition, like `nlp.add_pipe("entity_ruler")`.
#[derive(Clone, Debug)]
pub struct EntityRuler {
    vocab: Arc<Vocab>,
    matcher: Matcher,
    phrase_matcher: PhraseMatcher,
    /// Overwrite existing entities.
    pub overwrite: bool,
    ent_id_sep: String,
    ent_ids: HashMap<u64, (String, String)>,
    patterns: Vec<RulerPattern>,
    phrase_matcher_attr: u64,
}

impl EntityRuler {
    /// A ruler for a pipeline's vocab (phrase patterns on `ORTH`).
    pub fn new(nlp: &Language) -> Self {
        Self::with_options(nlp, ORTH, false)
    }

    /// A ruler matching phrases on `phrase_matcher_attr` (e.g. [`crate::attrs::LOWER`]).
    pub fn with_options(nlp: &Language, phrase_matcher_attr: u64, overwrite_ents: bool) -> Self {
        let vocab = nlp.vocab().clone();
        EntityRuler {
            matcher: Matcher::new(&vocab),
            phrase_matcher: PhraseMatcher::with_attr(&vocab, phrase_matcher_attr),
            vocab,
            overwrite: overwrite_ents,
            ent_id_sep: DEFAULT_ENT_ID_SEP.to_string(),
            ent_ids: HashMap::new(),
            patterns: Vec::new(),
            phrase_matcher_attr,
        }
    }

    /// All patterns.
    pub fn patterns(&self) -> &[RulerPattern] {
        &self.patterns
    }

    /// Number of patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }
    /// Whether there are no patterns.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Labels of all patterns (sorted).
    pub fn labels(&self) -> Vec<String> {
        let s: std::collections::BTreeSet<String> = self.patterns.iter().map(|p| p.label.clone()).collect();
        s.into_iter().collect()
    }

    fn key(&self, p: &RulerPattern) -> String {
        match &p.id {
            Some(id) => format!("{}{}{}", p.label, self.ent_id_sep, id),
            None => p.label.clone(),
        }
    }

    /// Add patterns. Phrase patterns are tokenized with `nlp` (the full
    /// pipeline is run if the phrase matcher attribute needs annotations).
    pub fn add_patterns(&mut self, nlp: &Language, patterns: &[RulerPattern]) -> Result<(), Error> {
        for p in patterns {
            let key = self.key(p);
            if let Some(id) = &p.id {
                let h = self.vocab.strings.add(&key);
                self.ent_ids.insert(h, (p.label.clone(), id.clone()));
            }
            match &p.pattern {
                serde_json::Value::String(text) => {
                    let doc = if super::span_ruler::needs_annotation(self.phrase_matcher_attr) {
                        nlp.call(text)?
                    } else {
                        nlp.make_doc(text)
                    };
                    self.phrase_matcher.add(&key, &[doc])?;
                }
                serde_json::Value::Array(_) => {
                    self.matcher.add(&key, std::slice::from_ref(&p.pattern), None)?;
                }
                other => return Err(Error::Value(format!("[E097] Invalid pattern: {other}"))),
            }
            self.patterns.push(p.clone());
        }
        Ok(())
    }

    /// Matches, deduplicated and sorted by length (desc) then start.
    pub fn find_matches(&self, doc: &Doc) -> Result<Vec<(u64, usize, usize)>, Error> {
        let mut m: Vec<(u64, usize, usize)> = if self.matcher.is_empty() {
            Vec::new()
        } else {
            self.matcher.find_allow_missing(doc).into_iter().map(|m| (m.match_id, m.start, m.end)).collect()
        };
        m.extend(self.phrase_matcher.find(doc).into_iter().map(|m| (m.match_id, m.start, m.end)));
        let mut seen = HashSet::new();
        let mut uniq: Vec<(u64, usize, usize)> = m.into_iter().filter(|x| x.1 != x.2 && seen.insert(*x)).collect();
        uniq.sort_by(|a, b| {
            let ka = (a.2 - a.1, -(a.1 as i64));
            let kb = (b.2 - b.1, -(b.1 as i64));
            kb.cmp(&ka)
        });
        Ok(uniq)
    }
}

impl Pipe for EntityRuler {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        let matches = self.find_matches(doc)?;
        let mut entities = doc.ent_spans();
        let mut new_entities = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for (match_id, start, end) in matches {
            if !self.overwrite && doc.tokens[start..end].iter().any(|t| t.ent_type != 0) {
                continue;
            }
            if !seen.contains(&start) && !seen.contains(&(end - 1)) {
                let mut c = SpanC::new(start, end);
                match self.ent_ids.get(&match_id) {
                    Some((label, id)) => {
                        c.label = self.vocab.strings.add(label);
                        c.id = self.vocab.strings.add(id);
                    }
                    None => c.label = match_id,
                }
                new_entities.push(c);
                entities.retain(|e| !(e.start < end && e.end > start));
                seen.extend(start..end);
            }
        }
        entities.extend(new_entities);
        entities.sort_by_key(|e| e.start);
        doc.set_ents(&entities)
    }
}
