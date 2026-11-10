//! The `tagger` component (`Tagger.v2`: listener + softmax).

use super::nn::Linear;
use crate::Error;
use crate::doc::Doc;
use crate::language::Pipe;

/// Assigns `token.tag_` from the shared `tok2vec` output (`doc.tensor`).
#[derive(Clone, Debug)]
pub struct Tagger {
    /// Softmax layer.
    pub output: Linear,
    /// Tag labels in class order.
    pub labels: Vec<String>,
    /// Overwrite existing tags.
    pub overwrite: bool,
}

impl Tagger {
    /// Class scores per token (logits), `(len, n_labels)`.
    pub fn scores(&self, doc: &Doc) -> Result<Vec<f32>, Error> {
        let (w, t) = doc
            .tensor
            .as_ref()
            .ok_or_else(|| Error::Value("tagger needs doc.tensor; add the tok2vec component first".into()))?;
        if *w != self.output.n_in {
            return Err(Error::Value("tensor width does not match tagger".into()));
        }
        Ok(self.output.forward(t, doc.len()))
    }
}

pub(crate) fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

impl Pipe for Tagger {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        if doc.is_empty() {
            return Ok(());
        }
        let s = self.scores(doc)?;
        let nc = self.output.n_out;
        let vocab = doc.vocab().clone();
        for (i, t) in doc.tokens.iter_mut().enumerate() {
            if t.tag == 0 || self.overwrite {
                let best = argmax(&s[i * nc..(i + 1) * nc]);
                t.tag = vocab.strings.add(&self.labels[best]);
            }
        }
        Ok(())
    }
}
