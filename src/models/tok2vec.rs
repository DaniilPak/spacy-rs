//! spaCy's `Tok2Vec.v2` = `MultiHashEmbed.v2` + `MaxoutWindowEncoder.v2`.

use super::msgpack::{Node, as_str};
use super::nn::{HashEmbed, Maxout, seq2col};
use crate::Error;
use crate::attrs::*;
use crate::doc::{Doc, token_attr};

/// A CNN token-to-vector encoder.
#[derive(Clone, Debug)]
pub struct Tok2Vec {
    attrs: Vec<u64>,
    embeds: Vec<HashEmbed>,
    mixer: Maxout,
    encoders: Vec<Maxout>,
    window: usize,
    pad: usize,
    width: usize,
}

fn attr_from_name(s: &str) -> Result<u64, Error> {
    attr_id(s).ok_or_else(|| Error::Format(format!("unknown feature attribute {s}")))
}

impl Tok2Vec {
    /// Build from the node list of a serialized model, starting at the first
    /// `extract_features` node after `start`.
    pub fn from_nodes(nodes: &[Node]) -> Result<Self, Error> {
        let fe = nodes
            .iter()
            .find(|n| n.name == "extract_features")
            .ok_or_else(|| Error::Format("no extract_features layer".into()))?;
        let attrs: Vec<u64> = fe
            .attr("columns")
            .and_then(|c| c.as_array())
            .ok_or_else(|| Error::Format("extract_features without columns".into()))?
            .iter()
            .map(|c| attr_from_name(as_str(c).unwrap_or("")))
            .collect::<Result<_, _>>()?;
        let mut embeds: Vec<(usize, HashEmbed)> = Vec::new();
        for n in nodes.iter().filter(|n| n.name == "hashembed") {
            let col = n.attr("column").and_then(|c| c.as_u64()).unwrap_or(embeds.len() as u64) as usize;
            embeds.push((col, HashEmbed::from_node(n)?));
        }
        embeds.sort_by_key(|(c, _)| *c);
        let maxouts: Vec<&Node> = nodes.iter().filter(|n| n.name == "maxout").collect();
        let norms: Vec<&Node> = nodes.iter().filter(|n| n.name == "layernorm").collect();
        if maxouts.len() < 2 || maxouts.len() != norms.len() {
            return Err(Error::Format("unexpected tok2vec structure".into()));
        }
        let mixer = Maxout::from_nodes(maxouts[0], Some(norms[0]))?;
        let encoders = maxouts[1..]
            .iter()
            .zip(&norms[1..])
            .map(|(m, l)| Maxout::from_nodes(m, Some(l)))
            .collect::<Result<Vec<_>, _>>()?;
        let window = nodes
            .iter()
            .find(|n| n.name == "expand_window")
            .and_then(|n| n.attr("window_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let pad = nodes
            .iter()
            .find(|n| n.name.starts_with("with_array(residual"))
            .and_then(|n| n.attr("pad"))
            .and_then(|v| v.as_u64())
            .unwrap_or((window * encoders.len()) as u64) as usize;
        let width = embeds.first().map(|e| e.1.width()).unwrap_or(0);
        Ok(Tok2Vec {
            attrs,
            embeds: embeds.into_iter().map(|e| e.1).collect(),
            mixer,
            encoders,
            window,
            pad,
            width,
        })
    }

    /// Output width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Output of the embedding stage only (`MultiHashEmbed`), `(len, width)`.
    pub fn embed(&self, doc: &Doc) -> Vec<f32> {
        let n = doc.len();
        let k = self.embeds.len();
        let w = self.width;
        let mut x = vec![0f32; n * k * w];
        for (i, t) in doc.tokens.iter().enumerate() {
            for (c, (e, &a)) in self.embeds.iter().zip(&self.attrs).enumerate() {
                let id = match a {
                    SPACY => t.spacy as u64,
                    _ => token_attr(t, a),
                };
                let off = i * k * w + c * w;
                e.embed_into(id, &mut x[off..off + w]);
            }
        }
        self.mixer.forward(&x, n)
    }

    /// Token vectors for a doc, row-major `(len, width)`.
    pub fn forward(&self, doc: &Doc) -> Vec<f32> {
        let n = doc.len();
        if n == 0 {
            return Vec::new();
        }
        let w = self.width;
        let h = self.embed(doc);
        // encoder over [pad, doc, pad]
        let total = n + 2 * self.pad;
        let mut y = vec![0f32; total * w];
        y[self.pad * w..(self.pad + n) * w].copy_from_slice(&h);
        for enc in &self.encoders {
            let cols = seq2col(&y, total, w, self.window);
            let d = enc.forward(&cols, total);
            for (a, b) in y.iter_mut().zip(d) {
                *a += b;
            }
        }
        y[self.pad * w..(self.pad + n) * w].to_vec()
    }
}
