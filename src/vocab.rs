//! The vocabulary: a string store plus a cache of lexemes with their
//! lexical attributes.

use crate::attrs::*;
use crate::lang::LexAttrConfig;
use crate::lexeme::{self, LexemeC};
use crate::strings::StringStore;
use crate::uchar::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// spaCy's out-of-vocabulary rank for `Lexeme.rank`.
pub const OOV_RANK: u64 = 0xffff_ffff_ffff_ffff;

/// A vocabulary shared by all docs created by a pipeline.
pub struct Vocab {
    /// Interned strings.
    pub strings: StringStore,
    lexemes: RwLock<HashMap<u64, Arc<LexemeC>>>,
    lang: String,
    lang_id: u64,
    lex_attrs: LexAttrConfig,
    norm_tables: RwLock<Vec<Arc<HashMap<u64, String>>>>,
}

impl std::fmt::Debug for Vocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vocab(lang={}, {} lexemes)", self.lang, self.len())
    }
}

impl Vocab {
    /// Create a vocabulary for a language with the given lexical attribute config.
    pub fn new(lang: &str, lex_attrs: LexAttrConfig) -> Self {
        let strings = StringStore::new();
        let lang_id = strings.add(lang);
        Vocab {
            strings,
            lexemes: RwLock::new(HashMap::new()),
            lang: lang.to_string(),
            lang_id,
            lex_attrs,
            norm_tables: RwLock::new(vec![Arc::new(
                crate::lang::base_norms()
                    .iter()
                    .map(|(k, v)| (crate::strings::get_string_id(k), v.clone()))
                    .collect(),
            )]),
        }
    }

    /// Language code.
    pub fn lang(&self) -> &str {
        &self.lang
    }

    /// Number of cached lexemes.
    pub fn len(&self) -> usize {
        self.lexemes.read().expect("lock").len()
    }

    /// Whether no lexemes are cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lexical attribute configuration.
    pub fn lex_attrs(&self) -> &LexAttrConfig {
        &self.lex_attrs
    }

    /// Prepend a `lexeme_norm` lookup table (as loaded from a trained pipeline).
    /// Lexemes already created are rebuilt lazily on next access.
    pub fn add_norm_table(&self, table: HashMap<String, String>) {
        self.add_norm_table_ids(table.into_iter().map(|(k, v)| (crate::strings::get_string_id(&k), v)).collect());
    }

    /// Like [`Vocab::add_norm_table`] with keys already hashed (spaCy `Lookups` format).
    pub fn add_norm_table_ids(&self, table: HashMap<u64, String>) {
        self.norm_tables.write().expect("lock").insert(0, Arc::new(table));
        self.lexemes.write().expect("lock").clear();
    }

    /// Get or create the lexeme for a string.
    pub fn get(&self, s: &str) -> Arc<LexemeC> {
        let orth = crate::strings::get_string_id(s);
        if let Some(l) = self.lexemes.read().expect("lock").get(&orth) {
            return l.clone();
        }
        let lex = Arc::new(self.make_lexeme(s));
        self.lexemes
            .write()
            .expect("lock")
            .entry(orth)
            .or_insert(lex)
            .clone()
    }

    /// Get the lexeme for an ID if it exists.
    pub fn get_by_orth(&self, orth: u64) -> Option<Arc<LexemeC>> {
        if let Some(l) = self.lexemes.read().expect("lock").get(&orth) {
            return Some(l.clone());
        }
        let s = self.strings.get(orth)?.to_string();
        Some(self.get(&s))
    }

    /// Overwrite a boolean flag on a lexeme (e.g. make a word a stop word).
    pub fn set_lexeme_flag(&self, s: &str, flag: u64, value: bool) {
        let mut lex = (*self.get(s)).clone();
        lex.set_flag(flag, value);
        self.lexemes.write().expect("lock").insert(lex.orth, Arc::new(lex));
    }

    /// The lexeme NORM for a string (lookup tables, then lowercase).
    pub fn norm_of(&self, s: &str) -> String {
        let id = crate::strings::get_string_id(s);
        for t in self.norm_tables.read().expect("lock").iter() {
            if let Some(n) = t.get(&id) {
                return n.clone();
            }
        }
        py_lower(s)
    }

    fn make_lexeme(&self, s: &str) -> LexemeC {
        let st = &self.strings;
        let lower = py_lower(s);
        let mut lex = LexemeC {
            orth: st.add(s),
            length: char_len(s),
            lower: st.add(&lower),
            norm: st.add(&self.norm_of(s)),
            shape: st.add(&lexeme::word_shape(s)),
            prefix: st.add(lexeme::prefix(s)),
            suffix: st.add(lexeme::suffix(s)),
            lang: self.lang_id,
            id: OOV_RANK,
            ..Default::default()
        };
        let f = [
            (IS_ALPHA, py_isalpha(s)),
            (IS_ASCII, lexeme::is_ascii(s)),
            (IS_DIGIT, py_isdigit(s)),
            (IS_LOWER, py_islower(s)),
            (IS_SPACE, py_isspace(s)),
            (IS_TITLE, py_istitle(s)),
            (IS_UPPER, py_isupper(s)),
            (IS_STOP, self.lex_attrs.stop_words.contains(&lower)),
            (LIKE_EMAIL, lexeme::like_email(s)),
            (LIKE_NUM, self.lex_attrs.like_num.check(s)),
            (IS_PUNCT, lexeme::is_punct(s)),
            (IS_BRACKET, lexeme::is_bracket(s)),
            (IS_QUOTE, lexeme::is_quote(s)),
            (IS_LEFT_PUNCT, lexeme::is_left_punct(s)),
            (IS_RIGHT_PUNCT, lexeme::is_right_punct(s)),
            (IS_CURRENCY, lexeme::is_currency(s)),
            (LIKE_URL, !s.is_empty() && lexeme::like_url(s)),
        ];
        for (flag, v) in f {
            lex.set_flag(flag, v);
        }
        lex
    }
}
