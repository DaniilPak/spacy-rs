//! Rule-based pipeline components.

pub mod entity_ruler;
pub mod lemmatizer;
pub mod sentencizer;
pub mod span_ruler;

pub use entity_ruler::EntityRuler;
pub use lemmatizer::{Lemmatizer, LemmatizerMode};
pub use sentencizer::Sentencizer;
pub use span_ruler::SpanRuler;

use crate::doc::SpanC;
use std::collections::HashSet;

/// `spacy.util.filter_spans`: remove overlaps, preferring the (first) longest span.
pub fn filter_spans(spans: &[SpanC]) -> Vec<SpanC> {
    let mut sorted: Vec<&SpanC> = spans.iter().collect();
    // stable sort, descending by (length, -start)
    sorted.sort_by(|a, b| {
        let ka = (a.end - a.start, -(a.start as i64));
        let kb = (b.end - b.start, -(b.start as i64));
        kb.cmp(&ka)
    });
    let mut seen: HashSet<usize> = HashSet::new();
    let mut out = Vec::new();
    for s in sorted {
        if !seen.contains(&s.start) && !seen.contains(&(s.end.saturating_sub(1))) {
            if s.end > s.start {
                seen.extend(s.start..s.end);
            }
            out.push(s.clone());
        }
    }
    out.sort_by_key(|s| s.start);
    out
}

/// A pattern for [`EntityRuler`] / [`SpanRuler`]: a phrase string or token pattern.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RulerPattern {
    /// Entity / span label.
    pub label: String,
    /// A phrase (string) or token pattern (JSON array).
    pub pattern: serde_json::Value,
    /// Optional ID.
    #[serde(default)]
    pub id: Option<String>,
}

impl RulerPattern {
    /// A phrase pattern.
    pub fn phrase(label: &str, text: &str) -> Self {
        RulerPattern {
            label: label.into(),
            pattern: serde_json::Value::String(text.into()),
            id: None,
        }
    }
    /// A token pattern.
    pub fn tokens(label: &str, pattern: serde_json::Value) -> Self {
        RulerPattern {
            label: label.into(),
            pattern,
            id: None,
        }
    }
    /// Set the ID.
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// Parse JSONL (one pattern per line), as written by `ruler.to_disk("patterns.jsonl")`.
pub fn read_jsonl_patterns(s: &str) -> Result<Vec<RulerPattern>, crate::Error> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| crate::Error::Format(format!("bad pattern line: {e}"))))
        .collect()
}
