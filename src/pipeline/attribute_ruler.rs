//! `AttributeRuler`: set token attributes (POS, MORPH, LEMMA, TAG, ...) from
//! Matcher patterns — used by `en_core_web_sm` to map tags to POS/morphology.

use crate::Error;
use crate::attrs::*;
use crate::doc::Doc;
use crate::language::Pipe;
use crate::matcher::Matcher;
use crate::vocab::Vocab;
use serde_json::Value;
use std::sync::Arc;

/// One attribute-ruler rule.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct AttributeRulerPattern {
    /// Matcher patterns.
    pub patterns: Vec<Value>,
    /// Attributes to set, e.g. `{"POS": "NOUN", "MORPH": "Number=Sing"}`.
    pub attrs: serde_json::Map<String, Value>,
    /// Token index within the match (negative counts from the end).
    #[serde(default)]
    pub index: i64,
}

#[derive(Clone, Debug)]
struct NormAttrs(Vec<(u64, u64)>);

/// Rule-based token attribute assignment.
#[derive(Clone, Debug)]
pub struct AttributeRuler {
    vocab: Arc<Vocab>,
    matcher: Matcher,
    attrs: Vec<NormAttrs>,
    indices: Vec<i64>,
    raw: Vec<AttributeRulerPattern>,
}

impl AttributeRuler {
    /// An empty ruler.
    pub fn new(vocab: &Arc<Vocab>) -> Self {
        AttributeRuler {
            vocab: vocab.clone(),
            matcher: Matcher::new(vocab),
            attrs: Vec::new(),
            indices: Vec::new(),
            raw: Vec::new(),
        }
    }

    /// The rules.
    pub fn patterns(&self) -> &[AttributeRulerPattern] {
        &self.raw
    }

    /// Add one rule (`AttributeRuler.add`).
    pub fn add(&mut self, patterns: &[Value], attrs: &serde_json::Map<String, Value>, index: i64) -> Result<(), Error> {
        let key = self.attrs.len().to_string();
        self.matcher.add(&key, patterns, None)?;
        let mut norm = Vec::new();
        for (k, v) in attrs {
            let a = attr_id(k).ok_or_else(|| Error::Value(format!("unknown attribute {k}")))?;
            let val = match (a, v) {
                (MORPH, Value::String(s)) => self.vocab.strings.add(&crate::morphology::normalize(s)),
                (POS, Value::String(s)) => pos::from_str(&s.to_uppercase()).ok_or_else(|| Error::Value(format!("bad POS {s}")))?,
                (_, Value::String(s)) => self.vocab.strings.add(s),
                (_, Value::Bool(b)) => *b as u64,
                (_, Value::Number(n)) => n.as_u64().unwrap_or(0),
                _ => continue,
            };
            norm.push((a, val));
        }
        self.attrs.push(NormAttrs(norm));
        self.indices.push(index);
        self.raw.push(AttributeRulerPattern {
            patterns: patterns.to_vec(),
            attrs: attrs.clone(),
            index,
        });
        Ok(())
    }

    /// Add many rules.
    pub fn add_patterns(&mut self, patterns: &[AttributeRulerPattern]) -> Result<(), Error> {
        for p in patterns {
            self.add(&p.patterns, &p.attrs, p.index)?;
        }
        Ok(())
    }
}

impl Pipe for AttributeRuler {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        let st = &self.vocab.strings;
        let mut matches: Vec<(usize, u64, usize, usize)> = self
            .matcher
            .find_allow_missing(doc)
            .into_iter()
            .map(|m| (st.get_or_empty(m.match_id).parse::<usize>().unwrap_or(0), m.match_id, m.start, m.end))
            .collect();
        matches.sort();
        for (attr_idx, _, start, end) in matches {
            let len = (end - start) as i64;
            let idx = self.indices[attr_idx];
            let pos = if idx < 0 { len + idx } else { idx };
            if pos < 0 || pos >= len {
                return Err(Error::Value(format!("[E1001] attribute ruler index {idx} out of range for span {start}..{end}")));
            }
            let t = &mut doc.tokens[start + pos as usize];
            for &(a, v) in &self.attrs[attr_idx].0 {
                match a {
                    TAG => t.tag = v,
                    POS => t.pos = v,
                    MORPH => t.morph = v,
                    LEMMA => t.lemma = v,
                    NORM => t.norm = v,
                    DEP => t.dep = v,
                    ENT_TYPE => t.ent_type = v,
                    ENT_IOB => t.ent_iob = v as u8,
                    ENT_ID => t.ent_id = v,
                    ENT_KB_ID => t.ent_kb_id = v,
                    SENT_START => t.sent_start = v as i64 as i8,
                    _ => {}
                }
            }
        }
        Ok(())
    }
}
