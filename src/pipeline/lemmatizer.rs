//! Lookup and rule-based lemmatization (`spacy.pipeline.Lemmatizer`,
//! `spacy.lang.en.lemmatizer.EnglishLemmatizer`).
//!
//! The English tables come from `spacy-lookups-data` (MIT), which derives them
//! from WordNet 3.0 (see `data/lookups/LICENSE-WordNet.txt`).

use crate::Error;
use crate::doc::{Doc, Token};
use crate::language::Pipe;
use crate::uchar::{py_isalpha, py_lower};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, LazyLock, RwLock};

/// Lemmatizer mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LemmatizerMode {
    /// `lemma_lookup` table keyed by token text.
    Lookup,
    /// POS-based rules (`lemma_rules`, `lemma_exc`, `lemma_index`); needs POS tags.
    Rule,
}

/// Lemmatizer tables. Keys are spaCy string IDs (as in `Lookups` tables).
#[derive(Clone, Debug, Default)]
pub struct LemmaTables {
    /// `lemma_lookup`: form → lemma.
    pub lookup: HashMap<u64, String>,
    /// `lemma_rules`: pos → [(old suffix, new suffix)].
    pub rules: HashMap<u64, Vec<(String, String)>>,
    /// `lemma_exc`: pos → form → lemmas.
    pub exc: HashMap<u64, HashMap<String, Vec<String>>>,
    /// `lemma_index`: pos → known lemmas.
    pub index: HashMap<u64, std::collections::HashSet<String>>,
}

fn gunzip_json(bytes: &[u8]) -> Value {
    let mut s = String::new();
    flate2::read::GzDecoder::new(bytes).read_to_string(&mut s).expect("bundled gzip");
    serde_json::from_str(&s).expect("bundled json")
}

fn key_id(k: &str) -> u64 {
    crate::strings::get_string_id(k)
}

impl LemmaTables {
    /// Build tables from JSON values in spacy-lookups-data format (string or numeric keys).
    pub fn from_json(lookup: Option<&Value>, rules: Option<&Value>, exc: Option<&Value>, index: Option<&Value>) -> Self {
        let mut t = LemmaTables::default();
        if let Some(Value::Object(m)) = lookup {
            t.lookup = m.iter().filter_map(|(k, v)| Some((key_id(k), v.as_str()?.to_string()))).collect();
        }
        if let Some(Value::Object(m)) = rules {
            for (pos, rs) in m {
                let v = rs
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|p| {
                                let p = p.as_array()?;
                                Some((p.first()?.as_str()?.to_string(), p.get(1)?.as_str()?.to_string()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                t.rules.insert(key_id(pos), v);
            }
        }
        if let Some(Value::Object(m)) = exc {
            for (pos, forms) in m {
                let mut h = HashMap::new();
                if let Value::Object(f) = forms {
                    for (k, v) in f {
                        let ls: Vec<String> = match v {
                            Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                            Value::String(s) => vec![s.clone()],
                            _ => vec![],
                        };
                        h.insert(k.clone(), ls);
                    }
                }
                t.exc.insert(key_id(pos), h);
            }
        }
        if let Some(Value::Object(m)) = index {
            for (pos, forms) in m {
                let set = forms
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                t.index.insert(key_id(pos), set);
            }
        }
        t
    }

    /// The bundled English tables.
    pub fn english() -> Arc<LemmaTables> {
        static T: LazyLock<Arc<LemmaTables>> = LazyLock::new(|| {
            let lookup = gunzip_json(include_bytes!("../../data/lookups/en_lemma_lookup.json.gz"));
            let rules = gunzip_json(include_bytes!("../../data/lookups/en_lemma_rules.json.gz"));
            let exc = gunzip_json(include_bytes!("../../data/lookups/en_lemma_exc.json.gz"));
            let index = gunzip_json(include_bytes!("../../data/lookups/en_lemma_index.json.gz"));
            Arc::new(LemmaTables::from_json(Some(&lookup), Some(&rules), Some(&exc), Some(&index)))
        });
        T.clone()
    }
}

/// Assigns `token.lemma_`.
pub struct Lemmatizer {
    mode: LemmatizerMode,
    tables: Arc<LemmaTables>,
    /// Overwrite existing lemmas.
    pub overwrite: bool,
    english: bool,
    cache: RwLock<HashMap<(u64, u64, u64), Vec<String>>>,
}

impl std::fmt::Debug for Lemmatizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lemmatizer({:?})", self.mode)
    }
}

impl Lemmatizer {
    /// English lemmatizer with the bundled tables.
    pub fn english(mode: LemmatizerMode) -> Self {
        Self::with_tables(mode, LemmaTables::english(), true)
    }

    /// Lemmatizer with custom tables. `english` enables `EnglishLemmatizer.is_base_form`.
    pub fn with_tables(mode: LemmatizerMode, tables: Arc<LemmaTables>, english: bool) -> Self {
        Lemmatizer {
            mode,
            tables,
            overwrite: false,
            english,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// The mode.
    pub fn mode(&self) -> LemmatizerMode {
        self.mode
    }

    /// `lookup_lemmatize`.
    pub fn lookup_lemmatize(&self, token: &Token<'_>) -> Vec<String> {
        let t = token.text();
        vec![self.tables.lookup.get(&crate::strings::get_string_id(t)).cloned().unwrap_or_else(|| t.to_string())]
    }

    fn is_base_form(&self, token: &Token<'_>) -> bool {
        if !self.english {
            return false;
        }
        let pos = token.pos_().to_lowercase();
        let m = token.morph_();
        let get = |f: &str| -> Option<String> {
            let v = crate::morphology::feats_to_pairs(m);
            v.into_iter().find(|(k, _)| k == f).map(|(_, v)| v)
        };
        let is = |f: &str, v: &str| get(f).as_deref() == Some(v);
        (pos == "noun" && is("Number", "Sing"))
            || (pos == "verb" && is("VerbForm", "Inf"))
            || (pos == "verb" && is("VerbForm", "Fin") && is("Tense", "Pres") && get("Number").is_none())
            || (pos == "adj" && is("Degree", "Pos"))
            || is("VerbForm", "Inf")
            || is("VerbForm", "None")
            || is("Degree", "Pos")
    }

    /// `rule_lemmatize`.
    pub fn rule_lemmatize(&self, token: &Token<'_>) -> Vec<String> {
        let key = (token.orth(), token.c().pos, token.c().morph);
        if let Some(v) = self.cache.read().expect("lock").get(&key) {
            return v.clone();
        }
        let string = token.text();
        let univ_pos = token.pos_().to_lowercase();
        if univ_pos.is_empty() || univ_pos == "eol" || univ_pos == "space" {
            return vec![py_lower(string)];
        }
        if self.is_base_form(token) {
            return vec![py_lower(string)];
        }
        let t = &self.tables;
        let pk = crate::strings::get_string_id(&univ_pos);
        let has_index = t.index.get(&pk).is_some_and(|x| !x.is_empty());
        let has_exc = t.exc.get(&pk).is_some_and(|x| !x.is_empty());
        let has_rules = t.rules.get(&pk).is_some_and(|x| !x.is_empty());
        if !(has_index || has_exc || has_rules) {
            return if univ_pos == "propn" { vec![string.to_string()] } else { vec![py_lower(string)] };
        }
        let empty_idx = Default::default();
        let index = t.index.get(&pk).unwrap_or(&empty_idx);
        let lower = py_lower(string);
        let mut forms: Vec<String> = Vec::new();
        let mut oov: Vec<String> = Vec::new();
        if let Some(rules) = t.rules.get(&pk) {
            for (old, new) in rules {
                if let Some(stem) = lower.strip_suffix(old.as_str()) {
                    let form = format!("{stem}{new}");
                    if form.is_empty() {
                    } else if index.contains(&form) {
                        forms.insert(0, form);
                    } else if !py_isalpha(&form) {
                        forms.push(form);
                    } else {
                        oov.push(form);
                    }
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        forms.retain(|f| seen.insert(f.clone()));
        if let Some(exc) = t.exc.get(&pk).and_then(|e| e.get(&lower)) {
            for f in exc {
                if !forms.contains(f) {
                    forms.insert(0, f.clone());
                }
            }
        }
        if forms.is_empty() {
            forms.extend(oov);
        }
        if forms.is_empty() {
            forms.push(string.to_string());
        }
        self.cache.write().expect("lock").insert(key, forms.clone());
        forms
    }

    /// Lemmatize one token.
    pub fn lemmatize(&self, token: &Token<'_>) -> Vec<String> {
        match self.mode {
            LemmatizerMode::Lookup => self.lookup_lemmatize(token),
            LemmatizerMode::Rule => self.rule_lemmatize(token),
        }
    }
}

impl Pipe for Lemmatizer {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        let mut lemmas = Vec::with_capacity(doc.len());
        for t in doc.iter() {
            if self.overwrite || t.c().lemma == 0 {
                lemmas.push(Some(self.lemmatize(&t).into_iter().next().unwrap_or_default()));
            } else {
                lemmas.push(None);
            }
        }
        let vocab = doc.vocab().clone();
        for (t, l) in doc.tokens.iter_mut().zip(lemmas) {
            if let Some(l) = l {
                t.lemma = vocab.strings.add(&l);
            }
        }
        Ok(())
    }
}
