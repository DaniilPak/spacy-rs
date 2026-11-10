//! Rule-based sentence boundary detection (`spacy.pipeline.Sentencizer`).

use crate::Error;
use crate::doc::Doc;
use crate::language::Pipe;
use std::collections::HashSet;
use std::sync::LazyLock;

static DEFAULT_PUNCT: LazyLock<Vec<String>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../data/sentencizer_punct.json")).expect("punct chars")
});

/// Splits sentences after tokens in `punct_chars`.
#[derive(Clone, Debug)]
pub struct Sentencizer {
    punct_chars: HashSet<String>,
    /// Overwrite existing sentence boundaries.
    pub overwrite: bool,
}

impl Default for Sentencizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Sentencizer {
    /// Default punctuation characters.
    pub fn new() -> Self {
        Sentencizer {
            punct_chars: DEFAULT_PUNCT.iter().cloned().collect(),
            overwrite: false,
        }
    }
    /// Custom punctuation characters.
    pub fn with_punct_chars<S: AsRef<str>>(chars: &[S]) -> Self {
        Sentencizer {
            punct_chars: chars.iter().map(|c| c.as_ref().to_string()).collect(),
            overwrite: false,
        }
    }
    /// The default punctuation characters.
    pub fn default_punct_chars() -> &'static [String] {
        &DEFAULT_PUNCT
    }

    /// Predict sentence starts for a doc.
    pub fn predict(&self, doc: &Doc) -> Vec<bool> {
        let n = doc.len();
        let mut guesses = vec![false; n];
        if n == 0 {
            return guesses;
        }
        let mut start = 0;
        let mut seen_period = false;
        guesses[0] = true;
        for t in doc.iter() {
            let in_punct = self.punct_chars.contains(t.text());
            if seen_period && !t.is_punct() && !in_punct {
                guesses[start] = true;
                start = t.i();
                seen_period = false;
            } else if in_punct {
                seen_period = true;
            }
        }
        if start < n {
            guesses[start] = true;
        }
        guesses
    }
}

impl Pipe for Sentencizer {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        let g = self.predict(doc);
        for (t, s) in doc.tokens.iter_mut().zip(g) {
            if t.sent_start == 0 || self.overwrite {
                t.sent_start = if s { 1 } else { -1 };
            }
        }
        Ok(())
    }
}
