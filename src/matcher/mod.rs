//! Rule-based matching: [`Matcher`] (token patterns), [`PhraseMatcher`]
//! (phrase lists) — ports of `spacy/matcher/matcher.pyx` and
//! `spacy/matcher/phrasematcher.pyx`.
//!
//! Patterns use spaCy's JSON format, e.g.
//! `[{"LOWER": "hello"}, {"IS_PUNCT": true, "OP": "?"}, {"LOWER": {"IN": ["world", "there"]}}]`.

pub mod levenshtein;
mod phrase;

pub use phrase::PhraseMatcher;

use crate::Error;
use crate::attrs::*;
use crate::doc::{Doc, Span, SpanC, TokenC, token_attr};
use crate::vocab::Vocab;
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A match: `(match_id, start, end)` in token indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Match {
    /// Hash of the match key.
    pub match_id: u64,
    /// First token.
    pub start: usize,
    /// One past the last token.
    pub end: usize,
}

/// Greedy filtering of overlapping matches for a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Greedy {
    /// Prefer the earliest match.
    First,
    /// Prefer the longest match.
    Longest,
}

/// Callback invoked for each final match: `(doc, i, matches)`.
pub type OnMatch = Arc<dyn Fn(&mut Doc, usize, &[Match]) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quant {
    Zero,
    One,
    ZeroPlus,
    ZeroOne,
    Final,
}

#[derive(Clone, Debug)]
struct TokPat {
    quant: Quant,
    attrs: Vec<(u64, u64)>,
    preds: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetOp {
    In,
    NotIn,
    IsSubset,
    IsSuperset,
    Intersects,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmpOp {
    Eq,
    Ne,
    Ge,
    Le,
    Gt,
    Lt,
}

#[derive(Clone, Debug)]
enum PredKind {
    Regex(Regex),
    Set {
        op: SetOp,
        ids: HashSet<u64>,
        regexes: Vec<Regex>,
        fuzzy: Option<i64>,
        strs: Vec<String>,
    },
    Cmp {
        op: CmpOp,
        num: Option<f64>,
    },
    Fuzzy {
        value: String,
        fuzzy: i64,
    },
}

#[derive(Clone, Debug)]
struct Predicate {
    attr: u64,
    kind: PredKind,
}

#[derive(Clone, Debug)]
struct State {
    pat: usize,
    pos: usize,
    start: usize,
    length: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Action {
    Reject,
    Match,
    Advance,
    Retry,
    MatchExtend,
    RetryAdvance,
    RetryExtend,
    MatchReject,
    MatchDouble,
}

/// Token-pattern matcher, like `spacy.matcher.Matcher`.
#[derive(Clone)]
pub struct Matcher {
    vocab: Arc<Vocab>,
    patterns: Vec<(u64, Vec<TokPat>)>,
    preds: Vec<Predicate>,
    seen_preds: HashMap<String, usize>,
    filters: HashMap<u64, Option<Greedy>>,
    callbacks: HashMap<u64, OnMatch>,
    raw: HashMap<u64, Vec<Value>>,
    keys: Vec<u64>,
    seen_attrs: HashSet<u64>,
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matcher({} keys, {} patterns)", self.keys.len(), self.patterns.len())
    }
}

fn get_matcher_attr(t: &TokenC, attr: u64) -> u64 {
    if attr == SENT_START {
        return (t.sent_start == 1) as u64;
    }
    token_attr(t, attr)
}

fn ops_for(spec: &serde_json::Map<String, Value>) -> Result<Vec<Quant>, Error> {
    let op = spec.iter().find(|(k, _)| k.eq_ignore_ascii_case("OP")).map(|(_, v)| v);
    let Some(op) = op else { return Ok(vec![Quant::One]) };
    let op = op.as_str().ok_or_else(|| Error::Value("OP must be a string".into()))?;
    Ok(match op {
        "*" => vec![Quant::ZeroPlus],
        "+" => vec![Quant::One, Quant::ZeroPlus],
        "?" => vec![Quant::ZeroOne],
        "1" => vec![Quant::One],
        "!" => vec![Quant::Zero],
        o if o.starts_with('{') && o.ends_with('}') => {
            let inner = &o[1..o.len() - 1];
            let mm = if inner.contains(',') { inner.to_string() } else { format!("{inner},{inner}") };
            let (n, m) = mm.split_once(',').expect("comma");
            let isdec = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
            let bad = (!isdec(n) && !isdec(m)) || (isdec(n) && isdec(m) && n.parse::<usize>().ok() > m.parse::<usize>().ok());
            if bad {
                return Err(Error::Value(format!("[E011] Unknown operator: '{op}'")));
            }
            let n: usize = if n.is_empty() { 0 } else { n.parse().map_err(|_| Error::Value(format!("[E011] Unknown operator: '{op}'")))? };
            let mut v = vec![Quant::One; n];
            if m.is_empty() {
                v.push(Quant::ZeroPlus);
            } else {
                let m: usize = m.parse().map_err(|_| Error::Value(format!("[E011] Unknown operator: '{op}'")))?;
                v.extend(std::iter::repeat_n(Quant::ZeroOne, m.saturating_sub(n)));
            }
            v
        }
        _ => return Err(Error::Value(format!("[E011] Unknown operator: '{op}'"))),
    })
}

fn norm_attr_name(k: &str) -> Option<u64> {
    let up = k.to_ascii_uppercase();
    let up = match up.as_str() {
        "TEXT" => "ORTH",
        "IS_SENT_START" => "SENT_START",
        u => u,
    };
    attr_id(up)
}

fn value_id(vocab: &Vocab, v: &Value) -> Option<u64> {
    match v {
        Value::String(s) => Some(vocab.strings.add(s)),
        Value::Bool(b) => Some(*b as u64),
        Value::Number(n) => n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)),
        _ => None,
    }
}

impl Matcher {
    /// Create an empty matcher.
    pub fn new(vocab: &Arc<Vocab>) -> Self {
        Matcher {
            vocab: vocab.clone(),
            patterns: Vec::new(),
            preds: Vec::new(),
            seen_preds: HashMap::new(),
            filters: HashMap::new(),
            callbacks: HashMap::new(),
            raw: HashMap::new(),
            keys: Vec::new(),
            seen_attrs: HashSet::new(),
        }
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    /// Whether no patterns were added.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Whether a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains(&self.vocab.strings.add(key))
    }
    /// The patterns added for a key.
    pub fn get(&self, key: &str) -> Option<&[Value]> {
        self.raw.get(&self.vocab.strings.add(key)).map(|v| v.as_slice())
    }

    /// Add patterns (each a JSON array of token specs) under a key.
    pub fn add(&mut self, key: &str, patterns: &[Value], greedy: Option<Greedy>) -> Result<(), Error> {
        self.add_impl(key, patterns, greedy, None)
    }

    /// Add patterns with an `on_match` callback (run by [`Matcher::apply`]).
    pub fn add_with_callback(
        &mut self,
        key: &str,
        patterns: &[Value],
        greedy: Option<Greedy>,
        on_match: OnMatch,
    ) -> Result<(), Error> {
        self.add_impl(key, patterns, greedy, Some(on_match))
    }

    /// Add patterns given as a JSON string (an array of patterns).
    pub fn add_json(&mut self, key: &str, patterns_json: &str, greedy: Option<Greedy>) -> Result<(), Error> {
        let v: Value = serde_json::from_str(patterns_json).map_err(|e| Error::Value(format!("invalid JSON: {e}")))?;
        let arr = v.as_array().ok_or_else(|| Error::Value("[E948] patterns must be a list".into()))?;
        self.add(key, arr, greedy)
    }

    fn add_impl(&mut self, key: &str, patterns: &[Value], greedy: Option<Greedy>, cb: Option<OnMatch>) -> Result<(), Error> {
        let key_id = self.vocab.strings.add(key);
        let mut compiled = Vec::new();
        for p in patterns {
            let arr = p
                .as_array()
                .ok_or_else(|| Error::Value(format!("[E178] Each pattern should be a list of dicts, got {p}")))?;
            if arr.is_empty() {
                return Err(Error::Value(format!("[E012] Cannot add pattern for zero tokens to matcher. Key: {key}")));
            }
            compiled.push(self.compile_pattern(arr)?);
        }
        for c in compiled {
            self.patterns.push((key_id, c));
        }
        if !self.keys.contains(&key_id) {
            self.keys.push(key_id);
        }
        match cb {
            Some(cb) => {
                self.callbacks.insert(key_id, cb);
            }
            None => {
                self.callbacks.remove(&key_id);
            }
        }
        self.filters.insert(key_id, greedy);
        self.raw.entry(key_id).or_default().extend(patterns.iter().cloned());
        Ok(())
    }

    /// Remove a key and its patterns.
    pub fn remove(&mut self, key: &str) -> Result<(), Error> {
        let id = self.vocab.strings.add(key);
        if !self.keys.contains(&id) {
            return Err(Error::Value(format!("[E175] Can't remove rule for unknown match pattern ID: {key}")));
        }
        self.keys.retain(|k| *k != id);
        self.patterns.retain(|(k, _)| *k != id);
        self.raw.remove(&id);
        self.filters.remove(&id);
        self.callbacks.remove(&id);
        Ok(())
    }

    fn compile_pattern(&mut self, specs: &[Value]) -> Result<Vec<TokPat>, Error> {
        let mut out = Vec::new();
        for spec in specs {
            let spec = spec.as_object().ok_or_else(|| Error::Value("[E154] token spec must be a dict".into()))?;
            if spec.is_empty() {
                out.push(TokPat {
                    quant: Quant::One,
                    attrs: vec![(NULL_ATTR, 0)],
                    preds: vec![],
                });
                continue;
            }
            let ops = ops_for(spec)?;
            let mut attrs = Vec::new();
            let mut preds = Vec::new();
            for (k, v) in spec {
                let up = k.to_ascii_uppercase();
                if up == "OP" {
                    continue;
                }
                if up == "_" {
                    return Err(Error::Value("custom extension attributes (`_`) are not supported".into()));
                }
                let attr = norm_attr_name(k).ok_or_else(|| Error::Value(format!("[E152] The attribute {k} is not supported for token patterns.")))?;
                self.seen_attrs.insert(attr);
                match v {
                    Value::Object(d) => {
                        preds.extend(self.compile_preds(attr, d, false, None)?);
                    }
                    Value::String(s) if attr == ENT_IOB && ["", "I", "O", "B"].contains(&s.as_str()) => {
                        let idx = ["", "I", "O", "B"].iter().position(|x| x == s).expect("iob") as u64;
                        attrs.push((attr, idx));
                    }
                    other => {
                        let id = value_id(&self.vocab, other).ok_or_else(|| Error::Value(format!("[E153] invalid value {other} for {k}")))?;
                        attrs.push((attr, id));
                    }
                }
            }
            for q in ops {
                out.push(TokPat {
                    quant: q,
                    attrs: attrs.clone(),
                    preds: preds.clone(),
                });
            }
        }
        out.push(TokPat {
            quant: Quant::Final,
            attrs: vec![],
            preds: vec![],
        });
        Ok(out)
    }

    fn compile_preds(
        &mut self,
        attr: u64,
        d: &serde_json::Map<String, Value>,
        regex: bool,
        fuzzy: Option<i64>,
    ) -> Result<Vec<usize>, Error> {
        let mut out = Vec::new();
        for (ty, value) in d {
            let ty = ty.to_ascii_uppercase();
            let fuzzy_n = ty.strip_prefix("FUZZY").filter(|r| r.is_empty() || (r.len() == 1 && r.as_bytes()[0].is_ascii_digit()));
            if ty == "REGEX" && value.is_object() {
                out.extend(self.compile_preds(attr, value.as_object().expect("obj"), true, None)?);
                continue;
            }
            if let Some(n) = fuzzy_n
                && value.is_object()
            {
                let f = if n.is_empty() { -1 } else { n.parse::<i64>().expect("digit") };
                out.extend(self.compile_preds(attr, value.as_object().expect("obj"), false, Some(f))?);
                continue;
            }
            let cache_key = format!("{attr}|{regex}|{fuzzy:?}|{ty}|{value}");
            if let Some(&i) = self.seen_preds.get(&cache_key) {
                out.push(i);
                continue;
            }
            let kind = match ty.as_str() {
                "REGEX" => {
                    let s = value.as_str().ok_or_else(|| Error::Value("REGEX value must be a string".into()))?;
                    PredKind::Regex(crate::regex_compat::compile(s)?)
                }
                "IN" | "NOT_IN" | "IS_SUBSET" | "IS_SUPERSET" | "INTERSECTS" => {
                    let op = match ty.as_str() {
                        "IN" => SetOp::In,
                        "NOT_IN" => SetOp::NotIn,
                        "IS_SUBSET" => SetOp::IsSubset,
                        "IS_SUPERSET" => SetOp::IsSuperset,
                        _ => SetOp::Intersects,
                    };
                    let vals = value.as_array().ok_or_else(|| Error::Value(format!("{ty} value must be a list")))?;
                    let mut ids = HashSet::new();
                    let mut regexes = Vec::new();
                    let mut strs = Vec::new();
                    for v in vals {
                        if attr == MORPH {
                            let s = v.as_str().unwrap_or("");
                            ids.insert(self.vocab.strings.add(&crate::morphology::normalize(s)));
                        } else if regex {
                            regexes.push(crate::regex_compat::compile(v.as_str().unwrap_or(""))?);
                        } else if let Some(s) = v.as_str() {
                            ids.insert(if fuzzy.is_some() { self.vocab.strings.add(s) } else { crate::strings::get_string_id(s) });
                            strs.push(s.to_string());
                        } else if let Some(id) = value_id(&self.vocab, v) {
                            ids.insert(id);
                            strs.push(v.to_string());
                        }
                    }
                    PredKind::Set {
                        op,
                        ids,
                        regexes,
                        fuzzy,
                        strs,
                    }
                }
                "==" | "!=" | ">=" | "<=" | ">" | "<" => {
                    let op = match ty.as_str() {
                        "==" => CmpOp::Eq,
                        "!=" => CmpOp::Ne,
                        ">=" => CmpOp::Ge,
                        "<=" => CmpOp::Le,
                        ">" => CmpOp::Gt,
                        _ => CmpOp::Lt,
                    };
                    let num = match value {
                        Value::Number(n) => n.as_f64(),
                        Value::Bool(b) => Some(*b as u8 as f64),
                        _ => None,
                    };
                    if num.is_none() && !matches!(op, CmpOp::Eq | CmpOp::Ne) {
                        return Err(Error::Value(format!("comparison {ty} requires a number")));
                    }
                    PredKind::Cmp { op, num }
                }
                t if fuzzy_n.is_some() => {
                    let n = &t[5..];
                    let f = if n.is_empty() { -1 } else { n.parse::<i64>().expect("digit") };
                    PredKind::Fuzzy {
                        value: value.as_str().unwrap_or("").to_string(),
                        fuzzy: f,
                    }
                }
                _ => continue, // unknown predicate: spaCy warns (W035) and skips
            };
            let i = self.preds.len();
            self.preds.push(Predicate { attr, kind });
            self.seen_preds.insert(cache_key, i);
            out.push(i);
        }
        Ok(out)
    }

    fn eval_pred(&self, p: &Predicate, t: &TokenC) -> bool {
        let st = &self.vocab.strings;
        let v = get_matcher_attr(t, p.attr);
        match &p.kind {
            PredKind::Regex(r) => r.is_match(st.get_or_empty(v)).unwrap_or(false),
            PredKind::Fuzzy { value, fuzzy } => {
                let s = st.get_or_empty(v);
                s == value || levenshtein::levenshtein_compare(s, value, *fuzzy)
            }
            PredKind::Cmp { op, num } => {
                let x = v as f64;
                match (op, num) {
                    (CmpOp::Eq, Some(n)) => x == *n,
                    (CmpOp::Eq, None) => false,
                    (CmpOp::Ne, Some(n)) => x != *n,
                    (CmpOp::Ne, None) => true,
                    (CmpOp::Ge, Some(n)) => x >= *n,
                    (CmpOp::Le, Some(n)) => x <= *n,
                    (CmpOp::Gt, Some(n)) => x > *n,
                    (CmpOp::Lt, Some(n)) => x < *n,
                    _ => false,
                }
            }
            PredKind::Set {
                op,
                ids,
                regexes,
                fuzzy,
                strs,
            } => match op {
                SetOp::In | SetOp::NotIn => {
                    let hit = if !regexes.is_empty() {
                        let s = st.get_or_empty(v);
                        regexes.iter().any(|r| r.is_match(s).unwrap_or(false))
                    } else if let Some(f) = fuzzy {
                        let s = st.get_or_empty(v);
                        strs.iter().any(|x| levenshtein::levenshtein_compare(s, x, *f))
                    } else {
                        ids.contains(&v)
                    };
                    if *op == SetOp::In { hit } else { !hit }
                }
                _ => {
                    let tv: HashSet<u64> = if p.attr == MORPH {
                        crate::morphology::features(st.get_or_empty(v))
                            .iter()
                            .map(|f| crate::strings::get_string_id(f))
                            .collect()
                    } else {
                        [v].into_iter().collect()
                    };
                    match op {
                        SetOp::IsSubset => tv.is_subset(ids),
                        SetOp::IsSuperset => tv.is_superset(ids),
                        _ => !tv.is_disjoint(ids),
                    }
                }
            },
        }
    }

    fn is_match(&self, pat: &TokPat, t: &TokenC, cache: &mut [i8]) -> bool {
        for &pi in &pat.preds {
            if cache[pi] == 0 {
                cache[pi] = if self.eval_pred(&self.preds[pi], t) { 1 } else { -1 };
            }
            if cache[pi] == -1 {
                return false;
            }
        }
        pat.attrs.iter().all(|&(a, v)| get_matcher_attr(t, a) == v)
    }

    fn get_action(&self, s: &State, t: &TokenC, cache: &mut [i8]) -> Action {
        let pats = &self.patterns[s.pat].1;
        let pat = &pats[s.pos];
        let mut is_match = self.is_match(pat, t, cache);
        let mut q = pat.quant;
        let is_final = pats[s.pos + 1].quant == Quant::Final;
        if q == Quant::Zero {
            is_match = !is_match;
            q = Quant::One;
        }
        match q {
            Quant::One => match (is_match, is_final) {
                (true, true) => Action::Match,
                (true, false) => Action::Advance,
                _ => Action::Reject,
            },
            Quant::ZeroPlus => match (is_match, is_final) {
                (true, true) => Action::MatchExtend,
                (true, false) => Action::RetryExtend,
                (false, true) => Action::MatchReject,
                (false, false) => Action::Retry,
            },
            Quant::ZeroOne => match (is_match, is_final) {
                (true, true) => Action::MatchDouble,
                (true, false) => Action::RetryAdvance,
                (false, true) => Action::MatchReject,
                (false, false) => Action::Retry,
            },
            _ => Action::Reject,
        }
    }

    fn find_raw(&self, tokens: &[TokenC]) -> Vec<Match> {
        let n = tokens.len();
        let mut out = Vec::new();
        if n == 0 || self.patterns.is_empty() {
            return out;
        }
        let mut states: Vec<State> = Vec::new();
        let mut matches: Vec<(u64, usize, usize)> = Vec::new();
        let np = self.preds.len();
        let mut cache = vec![0i8; np];
        for (i, tok) in tokens.iter().enumerate() {
            for j in 0..self.patterns.len() {
                states.push(State {
                    pat: j,
                    pos: 0,
                    start: i,
                    length: 0,
                });
            }
            cache.iter_mut().for_each(|c| *c = 0);
            let mut q = 0;
            let mut new_states = Vec::new();
            for si in 0..states.len() {
                let mut action = self.get_action(&states[si], tok, &mut cache);
                if action == Action::Reject {
                    continue;
                }
                let state = states[si].clone();
                states[q] = state.clone();
                while matches!(action, Action::Retry | Action::RetryAdvance | Action::RetryExtend) {
                    if action == Action::RetryExtend {
                        new_states.push(State {
                            pat: states[q].pat,
                            pos: states[q].pos,
                            start: state.start,
                            length: state.length + 1,
                        });
                    }
                    if action == Action::RetryAdvance {
                        new_states.push(State {
                            pat: states[q].pat,
                            pos: states[q].pos + 1,
                            start: state.start,
                            length: state.length + 1,
                        });
                    }
                    states[q].pos += 1;
                    action = self.get_action(&states[q], tok, &mut cache);
                }
                let key = self.patterns[state.pat].0;
                match action {
                    Action::Reject => {}
                    Action::Advance => {
                        states[q].pos += 1;
                        states[q].length += 1;
                        q += 1;
                    }
                    Action::Match => matches.push((key, state.start, state.length + 1)),
                    Action::MatchDouble => {
                        if state.length > 0 {
                            matches.push((key, state.start, state.length));
                        }
                        matches.push((key, state.start, state.length + 1));
                    }
                    Action::MatchReject => matches.push((key, state.start, state.length)),
                    Action::MatchExtend => {
                        matches.push((key, state.start, state.length));
                        states[q].length += 1;
                        q += 1;
                    }
                    _ => {}
                }
            }
            states.truncate(q);
            states.extend(new_states);
        }
        // finish_states: states ending in zero-width patterns
        for s in &states {
            let pats = &self.patterns[s.pat].1;
            let mut pos = s.pos;
            while matches!(pats[pos].quant, Quant::ZeroPlus | Quant::ZeroOne) {
                if pats[pos + 1].quant == Quant::Final {
                    matches.push((self.patterns[s.pat].0, s.start, s.length));
                    break;
                }
                pos += 1;
            }
        }
        let mut seen = HashSet::new();
        for (k, s, l) in matches {
            let m = Match {
                match_id: k,
                start: s,
                end: s + l,
            };
            if l > 0 && seen.insert(m) {
                out.push(m);
            }
        }
        out
    }

    fn check_annotations(&self, doc: &Doc) -> Result<(), Error> {
        for (attr, pipe, name) in [
            (TAG, "tagger", "TAG"),
            (POS, "morphologizer or tagger+attribute_ruler", "POS"),
            (MORPH, "morphologizer or tagger+attribute_ruler", "MORPH"),
            (LEMMA, "lemmatizer", "LEMMA"),
            (DEP, "parser", "DEP"),
        ] {
            if self.seen_attrs.contains(&attr) && !doc.has_annotation(attr) {
                return Err(Error::Value(format!(
                    "[E155] The pipeline needs to include a {pipe} in order to use Matcher or PhraseMatcher with the attribute {name}. Try using `nlp()` instead of `nlp.make_doc()` or `list(nlp.pipe())` instead of `list(nlp.tokenizer.pipe())`."
                )));
            }
        }
        Ok(())
    }

    fn filter(&self, raw: Vec<Match>, length: usize) -> Vec<Match> {
        let mut final_matches = Vec::new();
        let mut pairs_by_id: Vec<(u64, Vec<Match>)> = Vec::new();
        for m in raw {
            match self.filters.get(&m.match_id).copied().flatten() {
                Some(_) => match pairs_by_id.iter_mut().find(|(k, _)| *k == m.match_id) {
                    Some((_, v)) => v.push(m),
                    None => pairs_by_id.push((m.match_id, vec![m])),
                },
                None => final_matches.push(m),
            }
        }
        for (key, mut pairs) in pairs_by_id {
            let mut matched = vec![false; length];
            match self.filters[&key] {
                Some(Greedy::First) => pairs.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end))),
                _ => pairs.sort_by(|a, b| {
                    let ka = (a.end - a.start, -(a.start as i64));
                    let kb = (b.end - b.start, -(b.start as i64));
                    kb.cmp(&ka)
                }),
            }
            for m in pairs {
                if matched[m.start..m.end].iter().all(|x| !x) {
                    final_matches.push(m);
                    matched[m.start..m.end].iter_mut().for_each(|x| *x = true);
                }
            }
        }
        final_matches
    }

    /// Find all matches in a doc (`matcher(doc)`); callbacks are not run.
    pub fn find(&self, doc: &Doc) -> Result<Vec<Match>, Error> {
        self.check_annotations(doc)?;
        Ok(self.filter(self.find_raw(&doc.tokens), doc.len()))
    }

    /// Like [`Matcher::find`] but skips the missing-annotation check (`allow_missing=True`).
    pub fn find_allow_missing(&self, doc: &Doc) -> Vec<Match> {
        self.filter(self.find_raw(&doc.tokens), doc.len())
    }

    /// Find matches within a span; offsets are relative to the span start, as in spaCy.
    pub fn find_in_span(&self, span: &Span<'_>) -> Result<Vec<Match>, Error> {
        self.check_annotations(span.doc())?;
        let toks = &span.doc().tokens[span.start()..span.end()];
        Ok(self.filter(self.find_raw(toks), toks.len()))
    }

    /// Matches as labelled spans (`as_spans=True`).
    pub fn find_spans<'d>(&self, doc: &'d Doc) -> Result<Vec<Span<'d>>, Error> {
        Ok(self
            .find(doc)?
            .into_iter()
            .map(|m| {
                let mut c = SpanC::new(m.start, m.end);
                c.label = m.match_id;
                doc.span_from(&c)
            })
            .collect())
    }

    /// Find matches and run `on_match` callbacks, which may modify the doc.
    pub fn apply(&self, doc: &mut Doc) -> Result<Vec<Match>, Error> {
        let matches = self.find(doc)?;
        for (i, m) in matches.iter().enumerate() {
            if let Some(cb) = self.callbacks.get(&m.match_id) {
                cb(doc, i, &matches);
            }
        }
        Ok(matches)
    }

    /// The vocab.
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }
}
