//! A faithful port of spaCy's rule-based `Tokenizer` (`spacy/tokenizer.pyx`).
//!
//! The algorithm: split on whitespace (single spaces become `whitespace_`),
//! then for each chunk repeatedly strip prefixes and suffixes (checking special
//! cases and `token_match` at every step), split the remainder on infixes
//! unless it is matched by `token_match`/`url_match`, and finally re-apply
//! special cases that span several tokens with a phrase matcher.

use crate::Error;
use crate::doc::{Doc, TokenC};
use crate::lang::{LangData, SpecialToken};
use crate::lexeme::LexemeC;
use crate::regex_compat;
use crate::strings::get_string_id;
use crate::uchar::is_space_char;
use crate::vocab::Vocab;
use fancy_regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
struct SpecialTok {
    lex: Arc<LexemeC>,
    norm: u64,
    byte_len: usize,
}

/// Compiled regexes in the shapes the tokenizer needs.
#[derive(Clone, Debug)]
struct Patterns {
    prefix: Option<Regex>,
    suffix: Option<Regex>,
    infix: Option<Regex>,
    token_match: Option<Regex>,
    url_match: Option<Regex>,
}

fn anchored(p: &str) -> Result<Regex, Error> {
    let t = regex_compat::translate(p);
    fancy_regex::RegexBuilder::new(&format!("^(?:{t})"))
        .backtrack_limit(10_000_000)
        .build()
        .map_err(|e| Error::Regex(format!("{e}")))
}

/// spaCy's tokenizer.
pub struct Tokenizer {
    vocab: Arc<Vocab>,
    rules: BTreeMap<String, Vec<SpecialToken>>,
    specials: HashMap<String, Arc<[SpecialTok]>>,
    pats: Patterns,
    pattern_src: [Option<String>; 5],
    /// Phrase patterns for special cases containing affixes: first ORTH -> [(orths, rule)].
    special_matcher: HashMap<u64, Vec<Vec<u64>>>,
    special_lens: HashSet<usize>,
    faster_heuristics: bool,
    cache: RwLock<HashMap<String, Arc<[Arc<LexemeC>]>>>,
    max_cache_size: usize,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tokenizer({} rules)", self.rules.len())
    }
}

/// Builder-style options for constructing a custom [`Tokenizer`].
#[derive(Clone, Debug, Default)]
pub struct TokenizerConfig {
    /// Prefix regex (Python `re` syntax, each alternative anchored with `^`).
    pub prefix: Option<String>,
    /// Suffix regex (Python syntax, alternatives anchored with `$`).
    pub suffix: Option<String>,
    /// Infix regex (Python syntax).
    pub infix: Option<String>,
    /// Token match regex (Python syntax, used with `re.match`).
    pub token_match: Option<String>,
    /// URL match regex (Python syntax, used with `re.match`).
    pub url_match: Option<String>,
    /// Special cases.
    pub rules: BTreeMap<String, Vec<SpecialToken>>,
    /// Restrict the special-case matcher to rules with affixes/spaces (default true).
    pub faster_heuristics: bool,
}

impl TokenizerConfig {
    /// The configuration for bundled language data.
    pub fn from_lang(d: &LangData) -> Self {
        TokenizerConfig {
            prefix: d.prefix.clone(),
            suffix: d.suffix.clone(),
            infix: d.infix.clone(),
            token_match: d.token_match.clone(),
            url_match: d.url_match.clone(),
            rules: d.rules.clone(),
            faster_heuristics: true,
        }
    }
}

/// Compile a list of prefix pieces like `spacy.util.compile_prefix_regex`.
pub fn compile_prefix_regex<S: AsRef<str>>(entries: &[S]) -> String {
    entries
        .iter()
        .map(|p| p.as_ref())
        .filter(|p| !p.trim().is_empty())
        .map(|p| format!("^{p}"))
        .collect::<Vec<_>>()
        .join("|")
}
/// Compile a list of suffix pieces like `spacy.util.compile_suffix_regex`.
pub fn compile_suffix_regex<S: AsRef<str>>(entries: &[S]) -> String {
    entries
        .iter()
        .map(|p| p.as_ref())
        .filter(|p| !p.trim().is_empty())
        .map(|p| format!("{p}$"))
        .collect::<Vec<_>>()
        .join("|")
}
/// Compile a list of infix pieces like `spacy.util.compile_infix_regex`.
pub fn compile_infix_regex<S: AsRef<str>>(entries: &[S]) -> String {
    entries
        .iter()
        .map(|p| p.as_ref())
        .filter(|p| !p.trim().is_empty())
        .collect::<Vec<_>>()
        .join("|")
}

impl Tokenizer {
    /// Build a tokenizer from a configuration.
    pub fn new(vocab: Arc<Vocab>, cfg: TokenizerConfig) -> Result<Self, Error> {
        let c = |p: &Option<String>| -> Result<Option<Regex>, Error> {
            p.as_deref().filter(|s| !s.is_empty()).map(regex_compat::compile).transpose()
        };
        let a = |p: &Option<String>| -> Result<Option<Regex>, Error> {
            p.as_deref().filter(|s| !s.is_empty()).map(anchored).transpose()
        };
        let pats = Patterns {
            prefix: c(&cfg.prefix)?,
            suffix: c(&cfg.suffix)?,
            infix: c(&cfg.infix)?,
            token_match: a(&cfg.token_match)?,
            url_match: a(&cfg.url_match)?,
        };
        let mut t = Tokenizer {
            vocab,
            rules: BTreeMap::new(),
            specials: HashMap::new(),
            pats,
            pattern_src: [cfg.prefix, cfg.suffix, cfg.infix, cfg.token_match, cfg.url_match],
            special_matcher: HashMap::new(),
            special_lens: HashSet::new(),
            faster_heuristics: cfg.faster_heuristics,
            cache: RwLock::new(HashMap::new()),
            max_cache_size: 10_000,
        };
        for (k, v) in cfg.rules {
            t.add_special_case(&k, v)?;
        }
        Ok(t)
    }

    /// Tokenizer for one of the bundled languages.
    pub fn for_lang(vocab: Arc<Vocab>, data: &LangData) -> Result<Self, Error> {
        Self::new(vocab, TokenizerConfig::from_lang(data))
    }

    /// The vocabulary.
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// The special-case rules.
    pub fn rules(&self) -> &BTreeMap<String, Vec<SpecialToken>> {
        &self.rules
    }

    /// The source patterns: prefix, suffix, infix, token_match, url_match.
    pub fn patterns(&self) -> [Option<&str>; 5] {
        [0, 1, 2, 3, 4].map(|i| self.pattern_src[i].as_deref())
    }

    /// Add a special-case tokenization rule. The `ORTH`s must concatenate to `string`.
    pub fn add_special_case(&mut self, string: &str, substrings: Vec<SpecialToken>) -> Result<(), Error> {
        let orth: String = substrings.iter().map(|s| s.orth.as_str()).collect();
        if orth != string {
            return Err(Error::Value(format!(
                "[E997] Tokenizer special cases are not allowed to modify the text. chunk: {string:?}, orth: {orth:?}"
            )));
        }
        let toks: Vec<SpecialTok> = substrings
            .iter()
            .map(|s| SpecialTok {
                lex: self.vocab.get(&s.orth),
                norm: s.norm.as_deref().map(|n| self.vocab.strings.add(n)).unwrap_or(0),
                byte_len: s.orth.len(),
            })
            .collect();
        self.specials.insert(string.to_string(), toks.into());
        self.rules.insert(string.to_string(), substrings);
        self.cache.write().expect("lock").clear();
        let needs_matcher = !self.faster_heuristics
            || self.find_prefix(string) != 0
            || !self.find_infix(string).is_empty()
            || self.find_suffix(string) != 0
            || string.contains(' ');
        if needs_matcher {
            let toks = self.tokenize_affixes(string, false);
            let orths: Vec<u64> = toks.iter().map(|t| t.lex.orth).collect();
            if let Some(&first) = orths.first() {
                self.special_lens.insert(orths.len());
                let v = self.special_matcher.entry(first).or_default();
                if !v.contains(&orths) {
                    v.push(orths);
                }
            }
        }
        Ok(())
    }

    /// Length in bytes of the prefix matched at the start of `s` (0 if none).
    pub fn find_prefix(&self, s: &str) -> usize {
        match &self.pats.prefix {
            Some(r) => r.find(s).ok().flatten().map(|m| m.end() - m.start()).unwrap_or(0),
            None => 0,
        }
    }

    /// Length in bytes of the suffix matched at the end of `s` (0 if none).
    pub fn find_suffix(&self, s: &str) -> usize {
        match &self.pats.suffix {
            Some(r) => r.find(s).ok().flatten().map(|m| m.end() - m.start()).unwrap_or(0),
            None => 0,
        }
    }

    /// Byte ranges of infixes in `s`.
    pub fn find_infix(&self, s: &str) -> Vec<(usize, usize)> {
        match &self.pats.infix {
            Some(r) => r
                .find_iter(s)
                .filter_map(|m| m.ok())
                .map(|m| (m.start(), m.end()))
                .collect(),
            None => Vec::new(),
        }
    }

    fn token_match(&self, s: &str) -> bool {
        self.pats.token_match.as_ref().is_some_and(|r| r.is_match(s).unwrap_or(false))
    }

    fn url_match(&self, s: &str) -> bool {
        self.pats.url_match.as_ref().is_some_and(|r| r.is_match(s).unwrap_or(false))
    }

    /// Tokenize a string into a [`Doc`].
    pub fn tokenize(&self, text: &str) -> Doc {
        let mut toks = self.tokenize_affixes(text, true);
        if !self.special_matcher.is_empty() {
            self.apply_special_cases(text, &mut toks);
        }
        // character offsets
        let mut chars = 0usize;
        let mut last_byte = 0usize;
        for t in toks.iter_mut() {
            chars += crate::uchar::char_len(&text[last_byte..t.byte_idx]);
            t.idx = chars;
            last_byte = t.byte_idx;
        }
        Doc::from_parts(self.vocab.clone(), text.to_string(), toks)
    }

    fn push(toks: &mut Vec<TokenC>, mut t: TokenC) {
        t.byte_idx = match toks.last() {
            Some(p) => p.byte_idx + p.byte_len + p.spacy as usize,
            None => 0,
        };
        t.spacy = false;
        toks.push(t);
    }

    fn push_lex(&self, toks: &mut Vec<TokenC>, s: &str) {
        Self::push(toks, TokenC::new(self.vocab.get(s), s.len()));
    }

    fn push_special(toks: &mut Vec<TokenC>, sp: &[SpecialTok]) {
        for s in sp {
            let mut t = TokenC::new(s.lex.clone(), s.byte_len);
            t.norm = s.norm;
            Self::push(toks, t);
        }
    }

    fn try_specials_and_cache(&self, s: &str, toks: &mut Vec<TokenC>, with_special: bool) -> (bool, bool) {
        if with_special && let Some(sp) = self.specials.get(s) {
            Self::push_special(toks, sp);
            return (true, true);
        }
        if let Some(c) = self.cache.read().expect("lock").get(s) {
            for l in c.iter() {
                let n = l.length;
                let bl = if n == 0 { 0 } else { self.vocab.strings.get_or_empty(l.orth).len() };
                Self::push(toks, TokenC::new(l.clone(), bl));
            }
            return (true, false);
        }
        (false, false)
    }

    pub(crate) fn tokenize_affixes(&self, string: &str, with_special: bool) -> Vec<TokenC> {
        let mut toks: Vec<TokenC> = Vec::with_capacity(string.len() / 4 + 1);
        if string.is_empty() {
            return toks;
        }
        let mut has_special = false;
        let mut start = 0usize;
        let mut in_ws = string.chars().next().is_some_and(is_space_char);
        for (i, uc) in string.char_indices() {
            if is_space_char(uc) != in_ws {
                if start < i {
                    self.tokenize_span(&string[start..i], &mut toks, &mut has_special, with_special);
                }
                if uc == ' ' {
                    if let Some(l) = toks.last_mut() {
                        l.spacy = true;
                    }
                    start = i + 1;
                } else {
                    start = i;
                }
                in_ws = !in_ws;
            }
        }
        if start < string.len() {
            self.tokenize_span(&string[start..], &mut toks, &mut has_special, with_special);
            if let Some(l) = toks.last_mut() {
                l.spacy = string.ends_with(' ') && !in_ws;
            }
        }
        toks
    }

    fn tokenize_span(&self, span: &str, toks: &mut Vec<TokenC>, has_special: &mut bool, with_special: bool) {
        let (hit, sp) = self.try_specials_and_cache(span, toks, with_special);
        if hit {
            *has_special |= sp;
            return;
        }
        let orig = toks.len();
        let mut prefixes: Vec<&str> = Vec::new();
        let mut suffixes: Vec<&str> = Vec::new();
        let rest = self.split_affixes(span, &mut prefixes, &mut suffixes, with_special);
        self.attach_tokens(toks, rest, &prefixes, &suffixes, has_special, with_special);
        let cache_len = self.cache.read().expect("lock").len();
        if !*has_special && toks.len() > orig && cache_len < self.max_cache_size {
            let lexes: Arc<[Arc<LexemeC>]> = toks[orig..].iter().map(|t| t.lex.clone()).collect();
            self.cache.write().expect("lock").insert(span.to_string(), lexes);
        }
    }

    fn split_affixes<'s>(
        &self,
        mut string: &'s str,
        prefixes: &mut Vec<&'s str>,
        suffixes: &mut Vec<&'s str>,
        with_special: bool,
    ) -> &'s str {
        let mut last_size = usize::MAX;
        while !string.is_empty() && string.len() != last_size {
            if self.token_match(string) {
                break;
            }
            if with_special && self.specials.contains_key(string) {
                break;
            }
            last_size = string.len();
            let pre_len = self.find_prefix(string);
            if pre_len != 0 {
                let minus_pre = &string[pre_len..];
                if !minus_pre.is_empty() && with_special && self.specials.contains_key(minus_pre) {
                    prefixes.push(&string[..pre_len]);
                    string = minus_pre;
                    break;
                }
            }
            let suf_len = self.find_suffix(&string[pre_len..]);
            if suf_len != 0 {
                let minus_suf = &string[..string.len() - suf_len];
                if !minus_suf.is_empty() && with_special && self.specials.contains_key(minus_suf) {
                    suffixes.push(&string[string.len() - suf_len..]);
                    string = minus_suf;
                    break;
                }
            }
            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= string.len() {
                prefixes.push(&string[..pre_len]);
                suffixes.push(&string[string.len() - suf_len..]);
                string = &string[pre_len..string.len() - suf_len];
            } else if pre_len != 0 {
                prefixes.push(&string[..pre_len]);
                string = &string[pre_len..];
            } else if suf_len != 0 {
                suffixes.push(&string[string.len() - suf_len..]);
                string = &string[..string.len() - suf_len];
            }
        }
        string
    }

    fn attach_tokens(
        &self,
        toks: &mut Vec<TokenC>,
        string: &str,
        prefixes: &[&str],
        suffixes: &[&str],
        has_special: &mut bool,
        with_special: bool,
    ) {
        for p in prefixes {
            self.push_lex(toks, p);
        }
        if !string.is_empty() {
            let (hit, sp) = self.try_specials_and_cache(string, toks, with_special);
            if hit {
                *has_special |= sp;
            } else if self.token_match(string) || self.url_match(string) {
                self.push_lex(toks, string);
            } else {
                let matches = self.find_infix(string);
                if matches.is_empty() {
                    self.push_lex(toks, string);
                } else {
                    let mut start = 0;
                    let start_before_infixes = start;
                    for (is, ie) in matches {
                        if is == start_before_infixes {
                            continue;
                        }
                        if is != start {
                            self.push_lex(toks, &string[start..is]);
                        }
                        if is != ie {
                            self.push_lex(toks, &string[is..ie]);
                        }
                        start = ie;
                    }
                    if start < string.len() {
                        self.push_lex(toks, &string[start..]);
                    }
                }
            }
        }
        for s in suffixes.iter().rev() {
            self.push_lex(toks, s);
        }
    }

    fn apply_special_cases(&self, text: &str, toks: &mut Vec<TokenC>) {
        // find all phrase matches
        let mut matches: Vec<(usize, usize)> = Vec::new();
        for i in 0..toks.len() {
            if let Some(pats) = self.special_matcher.get(&toks[i].lex.orth) {
                for p in pats {
                    let end = i + p.len();
                    if end <= toks.len() && toks[i..end].iter().zip(p).all(|(t, &o)| t.lex.orth == o) {
                        matches.push((i, end));
                    }
                }
            }
        }
        if matches.is_empty() {
            return;
        }
        // _filter_special_spans: longest first, then earliest start
        matches.sort_by(|a, b| {
            let (la, lb) = (a.1 - a.0, b.1 - b.0);
            la.cmp(&lb).then(b.0.cmp(&a.0))
        });
        matches.dedup();
        let mut seen: HashSet<usize> = HashSet::new();
        let mut filtered = Vec::new();
        for &(s, e) in matches.iter().rev() {
            if !seen.contains(&s) && !seen.contains(&(e - 1)) {
                filtered.push((s, e));
            }
            seen.extend(s..e);
        }
        filtered.sort_by_key(|m| m.0);
        let span_data: HashMap<usize, (usize, usize)> = filtered.iter().map(|&(s, e)| (s, (s, e))).collect();
        let mut out: Vec<TokenC> = Vec::with_capacity(toks.len());
        let mut i = 0;
        while i < toks.len() {
            let Some(&(s, e)) = span_data.get(&i) else {
                out.push(toks[i].clone());
                i += 1;
                continue;
            };
            let a = toks[s].byte_idx;
            let b = toks[e - 1].byte_idx + toks[e - 1].byte_len;
            let span_text = &text[a..b];
            match self.specials.get(span_text) {
                None => {
                    out.extend(toks[s..e].iter().cloned());
                }
                Some(sp) => {
                    let orig_final_spacy = toks[e - 1].spacy;
                    let mut off = toks[s].byte_idx;
                    let n = out.len();
                    for st in sp.iter() {
                        let mut t = TokenC::new(st.lex.clone(), st.byte_len);
                        t.norm = st.norm;
                        t.byte_idx = off;
                        off += st.byte_len;
                        out.push(t);
                    }
                    if out.len() > n {
                        out.last_mut().expect("non-empty").spacy = orig_final_spacy;
                    }
                }
            }
            i = e;
        }
        *toks = out;
    }

    /// Tokenize many texts (in parallel with the `rayon` feature).
    pub fn pipe<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Vec<Doc> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            texts.par_iter().map(|t| self.tokenize(t.as_ref())).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|t| self.tokenize(t.as_ref())).collect()
        }
    }

    /// Debugging helper like `Tokenizer.explain`: `(rule, token_text)` pairs.
    pub fn explain(&self, text: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for chunk in text.split(' ').filter(|c| !c.is_empty()) {
            let mut sub = chunk;
            let mut prefixes = Vec::new();
            let mut suffixes = Vec::new();
            let mut last = usize::MAX;
            let mut done = false;
            while !sub.is_empty() && sub.len() != last {
                if self.token_match(sub) {
                    break;
                }
                if self.specials.contains_key(sub) {
                    break;
                }
                last = sub.len();
                let p = self.find_prefix(sub);
                if p != 0 {
                    prefixes.push(("PREFIX".to_string(), sub[..p].to_string()));
                    sub = &sub[p..];
                    if self.specials.contains_key(sub) {
                        break;
                    }
                }
                let s = self.find_suffix(sub);
                if s != 0 {
                    suffixes.push(("SUFFIX".to_string(), sub[sub.len() - s..].to_string()));
                    sub = &sub[..sub.len() - s];
                    if self.specials.contains_key(sub) {
                        break;
                    }
                }
                done = true;
            }
            let _ = done;
            out.extend(prefixes);
            if let Some(sp) = self.specials.get(sub) {
                for t in sp.iter() {
                    out.push(("SPECIAL".into(), self.vocab.strings.get_or_empty(t.lex.orth).to_string()));
                }
            } else if self.token_match(sub) {
                out.push(("TOKEN_MATCH".into(), sub.to_string()));
            } else if self.url_match(sub) {
                out.push(("URL_MATCH".into(), sub.to_string()));
            } else if !sub.is_empty() {
                let mut start = 0;
                for (is, ie) in self.find_infix(sub) {
                    if is == 0 {
                        continue;
                    }
                    if is != start {
                        out.push(("TOKEN".into(), sub[start..is].to_string()));
                    }
                    if is != ie {
                        out.push(("INFIX".into(), sub[is..ie].to_string()));
                    }
                    start = ie;
                }
                if start < sub.len() {
                    out.push(("TOKEN".into(), sub[start..].to_string()));
                }
            }
            out.extend(suffixes.into_iter().rev());
        }
        out
    }
}

/// Look up the string ID used by the tokenizer for a key (exposed for tests).
pub fn key(s: &str) -> u64 {
    get_string_id(s)
}
