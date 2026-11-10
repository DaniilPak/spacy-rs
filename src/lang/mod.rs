//! Language data ported from spaCy (`spacy/lang/*`, MIT License,
//! Copyright (C) 2016-2024 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal).
//!
//! The data files under `data/lang/` are generated from spaCy's Python sources
//! by `tools/export_lang.py`: compiled prefix/suffix/infix regexes, URL and token
//! match patterns, tokenizer exceptions, stop words and number words.

use crate::lexeme::LikeNum;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

/// A token in a tokenizer exception (special case).
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct SpecialToken {
    /// The verbatim text of the token.
    #[serde(rename = "ORTH")]
    pub orth: String,
    /// Optional normalized form.
    #[serde(rename = "NORM", default)]
    pub norm: Option<String>,
}

impl SpecialToken {
    /// A token with only `ORTH`.
    pub fn new(orth: impl Into<String>) -> Self {
        SpecialToken {
            orth: orth.into(),
            norm: None,
        }
    }
    /// A token with `ORTH` and `NORM`.
    pub fn with_norm(orth: impl Into<String>, norm: impl Into<String>) -> Self {
        SpecialToken {
            orth: orth.into(),
            norm: Some(norm.into()),
        }
    }
}

/// Raw language data as exported from spaCy.
#[derive(Clone, Debug, Deserialize)]
pub struct LangData {
    /// ISO code, e.g. `"en"`.
    pub lang: String,
    /// Prefix regex (Python syntax).
    pub prefix: Option<String>,
    /// Suffix regex (Python syntax).
    pub suffix: Option<String>,
    /// Infix regex (Python syntax).
    pub infix: Option<String>,
    /// Token-match regex (Python syntax).
    pub token_match: Option<String>,
    /// URL-match regex (Python syntax).
    pub url_match: Option<String>,
    /// Tokenizer exceptions.
    pub rules: BTreeMap<String, Vec<SpecialToken>>,
    /// Stop words.
    pub stop_words: Vec<String>,
    /// Number words used by `like_num`.
    #[serde(default)]
    pub num_words: Vec<String>,
    /// Ordinal words used by `like_num`.
    #[serde(default)]
    pub ordinal_words: Vec<String>,
}

macro_rules! langs {
    ($($code:literal => $name:literal),* $(,)?) => {
        /// Languages with bundled tokenizer data: `(code, English name)`.
        pub const LANGUAGES: &[(&str, &str)] = &[$(($code, $name)),*];
        fn raw(code: &str) -> Option<&'static str> {
            match code {
                $($code => Some(include_str!(concat!("../../data/lang/", $code, ".json"))),)*
                _ => None,
            }
        }
    };
}

langs! {
    "en" => "English",
    "de" => "German",
    "fr" => "French",
    "es" => "Spanish",
    "it" => "Italian",
    "pt" => "Portuguese",
    "nl" => "Dutch",
    "xx" => "Multi-language",
}

static CACHE: LazyLock<Mutex<HashMap<String, Arc<LangData>>>> = LazyLock::new(Default::default);

/// Load the bundled data for a language code.
pub fn load(code: &str) -> Option<Arc<LangData>> {
    let mut c = CACHE.lock().expect("lang cache");
    if let Some(d) = c.get(code) {
        return Some(d.clone());
    }
    let d: LangData =
        serde_json::from_str(raw(code)?).expect("bundled language data is valid JSON");
    let d = Arc::new(d);
    c.insert(code.to_string(), d.clone());
    Some(d)
}

static BASE_NORMS_JSON: &str = include_str!("../../data/lang/base_norms.json");

/// spaCy's `BASE_NORMS` table (`spacy/lang/norm_exceptions.py`).
pub fn base_norms() -> &'static HashMap<String, String> {
    static N: LazyLock<HashMap<String, String>> =
        LazyLock::new(|| serde_json::from_str(BASE_NORMS_JSON).expect("base norms"));
    &N
}

pub(crate) fn base_url_pattern() -> &'static str {
    static P: LazyLock<String> = LazyLock::new(|| {
        load("en")
            .and_then(|d| d.url_match.clone())
            .expect("en url pattern")
    });
    &P
}

/// Lexical-attribute configuration for a language.
#[derive(Clone, Debug)]
pub struct LexAttrConfig {
    /// Lowercased stop words.
    pub stop_words: HashSet<String>,
    /// `like_num` behaviour.
    pub like_num: LikeNum,
}

impl LexAttrConfig {
    /// Build the configuration for a language from its data.
    pub fn for_lang(data: &LangData) -> Self {
        let mut ln = LikeNum::base();
        ln.num_words = data.num_words.iter().cloned().collect();
        ln.ordinal_words = data.ordinal_words.iter().cloned().collect();
        match data.lang.as_str() {
            "en" => ln.ordinal_suffixes = true,
            "pt" => ln.remove = vec![',', '.', 'º', 'ª'],
            "nl" => ln.strip_sign = false,
            "de" | "it" | "xx" => {
                ln.num_words.clear();
                ln.ordinal_words.clear();
            }
            _ => {}
        }
        LexAttrConfig {
            stop_words: data.stop_words.iter().cloned().collect(),
            like_num: ln,
        }
    }
}
