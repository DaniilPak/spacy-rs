//! `Language` (the `nlp` object): tokenizer plus a pipeline of components.

use crate::Error;
use crate::doc::Doc;
use crate::lang::{self, LexAttrConfig};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;
use std::sync::Arc;

/// A pipeline component that annotates a doc in place.
pub trait Pipe: Send + Sync {
    /// Process one doc.
    fn process(&self, doc: &mut Doc) -> Result<(), Error>;
    /// Process a batch of docs (override for batched inference).
    fn process_batch(&self, docs: &mut [Doc]) -> Result<(), Error> {
        for d in docs.iter_mut() {
            self.process(d)?;
        }
        Ok(())
    }
}

impl<F> Pipe for F
where
    F: Fn(&mut Doc) -> Result<(), Error> + Send + Sync,
{
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        self(doc)
    }
}

/// A processing pipeline, equivalent to spaCy's `Language` / `nlp`.
pub struct Language {
    vocab: Arc<Vocab>,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
    pipeline: Vec<(String, Box<dyn Pipe>)>,
    lang: String,
}

impl std::fmt::Debug for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Language(lang={}, pipeline={:?})", self.lang, self.pipe_names())
    }
}

/// Create a blank pipeline for a bundled language, like `spacy.blank("en")`.
pub fn blank(lang_code: &str) -> Result<Language, Error> {
    Language::blank(lang_code)
}

impl Language {
    /// Create a blank pipeline (tokenizer only).
    pub fn blank(lang_code: &str) -> Result<Self, Error> {
        let data = lang::load(lang_code).ok_or_else(|| Error::UnknownLanguage(lang_code.to_string()))?;
        let vocab = Arc::new(Vocab::new(lang_code, LexAttrConfig::for_lang(&data)));
        let tokenizer = Tokenizer::for_lang(vocab.clone(), &data)?;
        Ok(Language {
            vocab,
            tokenizer,
            pipeline: Vec::new(),
            lang: lang_code.to_string(),
        })
    }

    /// Assemble a pipeline from parts.
    pub fn from_parts(vocab: Arc<Vocab>, tokenizer: Tokenizer) -> Self {
        let lang = vocab.lang().to_string();
        Language {
            vocab,
            tokenizer,
            pipeline: Vec::new(),
            lang,
        }
    }

    /// Language code.
    pub fn lang(&self) -> &str {
        &self.lang
    }
    /// The shared vocabulary.
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }
    /// Names of the pipeline components in order.
    pub fn pipe_names(&self) -> Vec<&str> {
        self.pipeline.iter().map(|(n, _)| n.as_str()).collect()
    }
    /// Append a component.
    pub fn add_pipe(&mut self, name: impl Into<String>, pipe: impl Pipe + 'static) {
        self.pipeline.push((name.into(), Box::new(pipe)));
    }
    /// Insert a component before another one (or at the end if `before` is missing).
    pub fn add_pipe_before(&mut self, name: impl Into<String>, pipe: impl Pipe + 'static, before: &str) {
        let pos = self.pipeline.iter().position(|(n, _)| n == before).unwrap_or(self.pipeline.len());
        self.pipeline.insert(pos, (name.into(), Box::new(pipe)));
    }
    /// Add a boxed component.
    pub fn add_boxed_pipe(&mut self, name: impl Into<String>, pipe: Box<dyn Pipe>) {
        self.pipeline.push((name.into(), pipe));
    }
    /// Remove a component by name.
    pub fn remove_pipe(&mut self, name: &str) -> Option<Box<dyn Pipe>> {
        let pos = self.pipeline.iter().position(|(n, _)| n == name)?;
        Some(self.pipeline.remove(pos).1)
    }
    /// Whether a component exists.
    pub fn has_pipe(&self, name: &str) -> bool {
        self.pipeline.iter().any(|(n, _)| n == name)
    }
    /// Get a component by name.
    pub fn get_pipe(&self, name: &str) -> Option<&dyn Pipe> {
        self.pipeline.iter().find(|(n, _)| n == name).map(|(_, p)| p.as_ref())
    }

    /// Tokenize only (`nlp.make_doc`).
    pub fn make_doc(&self, text: &str) -> Doc {
        self.tokenizer.tokenize(text)
    }

    /// Run the full pipeline on a text (`nlp(text)`).
    pub fn call(&self, text: &str) -> Result<Doc, Error> {
        let mut doc = self.make_doc(text);
        self.process(&mut doc)?;
        Ok(doc)
    }

    /// Run all components on an existing doc.
    pub fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        for (_, p) in &self.pipeline {
            p.process(doc)?;
        }
        Ok(())
    }

    /// Run the pipeline with some components disabled.
    pub fn call_with_disabled(&self, text: &str, disable: &[&str]) -> Result<Doc, Error> {
        let mut doc = self.make_doc(text);
        for (n, p) in &self.pipeline {
            if !disable.contains(&n.as_str()) {
                p.process(&mut doc)?;
            }
        }
        Ok(doc)
    }

    /// Process many texts (`nlp.pipe`). Texts are split into batches of
    /// `batch_size`; batches run in parallel when the `rayon` feature is on.
    pub fn pipe<S: AsRef<str> + Sync>(&self, texts: &[S], batch_size: usize) -> Result<Vec<Doc>, Error> {
        let bs = batch_size.max(1);
        let run = |chunk: &[S]| -> Result<Vec<Doc>, Error> {
            let mut docs: Vec<Doc> = chunk.iter().map(|t| self.make_doc(t.as_ref())).collect();
            for (_, p) in &self.pipeline {
                p.process_batch(&mut docs)?;
            }
            Ok(docs)
        };
        #[cfg(feature = "rayon")]
        let batches: Vec<Result<Vec<Doc>, Error>> = {
            use rayon::prelude::*;
            texts.par_chunks(bs).map(run).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let batches: Vec<Result<Vec<Doc>, Error>> = texts.chunks(bs).map(run).collect();
        let mut out = Vec::with_capacity(texts.len());
        for b in batches {
            out.extend(b?);
        }
        Ok(out)
    }
}
