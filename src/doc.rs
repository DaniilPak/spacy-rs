//! `Doc`, `Token` and `Span`: the core annotated-text containers.

use crate::attrs::*;
use crate::lexeme::LexemeC;
use crate::vocab::Vocab;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Per-token data, mirroring spaCy's `TokenC`.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct TokenC {
    /// The lexeme (word type).
    pub lex: Arc<LexemeC>,
    /// Character offset in the doc text (Python string index).
    pub idx: usize,
    /// Byte offset in the doc text.
    pub byte_idx: usize,
    /// Length of the token text in bytes.
    pub byte_len: usize,
    /// Whether the token is followed by a single space.
    pub spacy: bool,
    /// Token-level NORM override (0 = use the lexeme's NORM).
    pub norm: u64,
    pub lemma: u64,
    pub tag: u64,
    pub pos: u64,
    pub morph: u64,
    pub dep: u64,
    /// Head offset relative to this token (0 = root or unset).
    pub head: isize,
    /// 1 = sentence start, -1 = not a sentence start, 0 = unknown.
    pub sent_start: i8,
    /// 0 = missing, 1 = I, 2 = O, 3 = B.
    pub ent_iob: u8,
    pub ent_type: u64,
    pub ent_kb_id: u64,
    pub ent_id: u64,
    /// Number of left children (set by `set_children_from_heads`).
    pub l_kids: u32,
    /// Number of right children.
    pub r_kids: u32,
    /// Leftmost token of the subtree.
    pub l_edge: usize,
    /// Rightmost token of the subtree.
    pub r_edge: usize,
}

impl TokenC {
    /// spaCy's `Token.missing_head`: no dependency annotation.
    pub fn missing_head(&self) -> bool {
        self.dep == 0 && self.head == 0
    }

    /// A fresh token for a lexeme.
    pub fn new(lex: Arc<LexemeC>, byte_len: usize) -> Self {
        TokenC {
            lex,
            idx: 0,
            byte_idx: 0,
            byte_len,
            spacy: false,
            norm: 0,
            lemma: 0,
            tag: 0,
            pos: 0,
            morph: 0,
            dep: 0,
            head: 0,
            sent_start: 0,
            ent_iob: 0,
            ent_type: 0,
            ent_kb_id: 0,
            ent_id: 0,
            l_kids: 0,
            r_kids: 0,
            l_edge: 0,
            r_edge: 0,
        }
    }
}

/// Default for tokens outside the given spans in [`Doc::set_ents_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntsDefault {
    /// Set to `O`.
    Outside,
    /// Set to missing.
    Missing,
    /// Set to blocked (`B` with no label).
    Blocked,
    /// Keep the current annotation.
    Unmodified,
}

/// A labelled span stored on a doc (`doc.spans[key]` or entity).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpanC {
    /// First token index.
    pub start: usize,
    /// One past the last token index.
    pub end: usize,
    /// Label ID.
    pub label: u64,
    /// Knowledge-base ID.
    pub kb_id: u64,
    /// Pattern / entity ID.
    pub id: u64,
}

impl SpanC {
    /// An unlabelled span.
    pub fn new(start: usize, end: usize) -> Self {
        SpanC {
            start,
            end,
            label: 0,
            kb_id: 0,
            id: 0,
        }
    }
}

/// A sequence of tokens with annotations.
#[derive(Clone)]
pub struct Doc {
    vocab: Arc<Vocab>,
    text: String,
    /// The tokens. Mutate through the pipeline components or setters.
    pub tokens: Vec<TokenC>,
    /// Named span groups (`doc.spans` in spaCy).
    pub spans: BTreeMap<String, Vec<SpanC>>,
    /// Whether spaces between tokens are unknown (docs built from words).
    pub has_unknown_spaces: bool,
    /// Optional token vectors written by `tok2vec` (row-major, `len x width`).
    pub tensor: Option<(usize, Vec<f32>)>,
}

impl fmt::Debug for Doc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Doc({:?})", self.text)
    }
}

impl fmt::Display for Doc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// How `Doc::char_span` handles character offsets that fall inside tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Alignment {
    /// Offsets must match token boundaries exactly.
    #[default]
    Strict,
    /// Snap to the tokens that overlap the range.
    Expand,
    /// Snap to the tokens fully inside the range.
    Contract,
}

impl Doc {
    /// Create a doc from tokens produced by the tokenizer.
    pub fn from_parts(vocab: Arc<Vocab>, text: String, tokens: Vec<TokenC>) -> Self {
        Doc {
            vocab,
            text,
            tokens,
            spans: BTreeMap::new(),
            has_unknown_spaces: false,
            tensor: None,
        }
    }

    /// Create a doc from words and (optional) trailing-space flags, like
    /// `Doc(vocab, words=..., spaces=...)`.
    pub fn from_words<S: AsRef<str>>(
        vocab: Arc<Vocab>,
        words: &[S],
        spaces: Option<&[bool]>,
    ) -> Self {
        let mut text = String::new();
        let mut tokens = Vec::with_capacity(words.len());
        let mut idx = 0;
        for (i, w) in words.iter().enumerate() {
            let w = w.as_ref();
            let sp = spaces
                .map(|s| s.get(i).copied().unwrap_or(false))
                .unwrap_or(true);
            let mut t = TokenC::new(vocab.get(w), w.len());
            t.l_edge = i;
            t.r_edge = i;
            t.byte_idx = text.len();
            t.idx = idx;
            t.spacy = sp;
            idx += crate::uchar::char_len(w) + sp as usize;
            text.push_str(w);
            if sp {
                text.push(' ');
            }
            tokens.push(t);
        }
        let mut d = Doc::from_parts(vocab, text, tokens);
        d.has_unknown_spaces = spaces.is_none();
        d
    }

    /// The vocabulary.
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }
    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    /// Whether the doc has no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    /// The full text, including trailing whitespace.
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Same as [`Doc::text`].
    pub fn text_with_ws(&self) -> &str {
        &self.text
    }
    /// The token at index `i` (panics if out of range).
    pub fn token(&self, i: usize) -> Token<'_> {
        assert!(i < self.len(), "token index {i} out of range");
        Token { doc: self, i }
    }
    /// The token at index `i`, or `None`.
    pub fn get(&self, i: usize) -> Option<Token<'_>> {
        (i < self.len()).then_some(Token { doc: self, i })
    }
    /// Iterate over tokens.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Token<'_>> + '_ {
        (0..self.len()).map(move |i| Token { doc: self, i })
    }
    /// A span of tokens `start..end`.
    pub fn span(&self, start: usize, end: usize) -> Span<'_> {
        assert!(
            start <= end && end <= self.len(),
            "invalid span {start}..{end}"
        );
        Span {
            doc: self,
            c: SpanC::new(start, end),
        }
    }
    /// Wrap stored span data.
    pub fn span_from(&self, c: &SpanC) -> Span<'_> {
        Span {
            doc: self,
            c: c.clone(),
        }
    }
    /// A labelled span of tokens.
    pub fn labelled_span(&self, start: usize, end: usize, label: &str) -> Span<'_> {
        let mut s = self.span(start, end);
        s.c.label = self.vocab.strings.add(label);
        s
    }

    /// A span from character offsets (Python string indices), like `Doc.char_span`.
    pub fn char_span(
        &self,
        start_char: usize,
        end_char: usize,
        label: Option<&str>,
        mode: Alignment,
    ) -> Option<Span<'_>> {
        if start_char > end_char {
            return None;
        }
        let mut start = None;
        let mut end = None;
        for (i, t) in self.tokens.iter().enumerate() {
            let ts = t.idx;
            let te = t.idx + t.lex.length;
            match mode {
                Alignment::Strict => {
                    if ts == start_char {
                        start = Some(i);
                    }
                    if te == end_char {
                        end = Some(i + 1);
                    }
                }
                Alignment::Expand => {
                    if start.is_none() && te > start_char {
                        start = Some(i);
                    }
                    if ts < end_char {
                        end = Some(i + 1);
                    }
                }
                Alignment::Contract => {
                    if start.is_none() && ts >= start_char {
                        start = Some(i);
                    }
                    if te <= end_char {
                        end = Some(i + 1);
                    }
                }
            }
        }
        let (s, e) = (start?, end?);
        if s > e || (s == e && start_char != end_char) {
            return None;
        }
        let mut sp = self.span(s, e);
        if let Some(l) = label {
            sp.c.label = self.vocab.strings.add(l);
        }
        Some(sp)
    }

    // ---------------------------------------------------------------- entities

    /// Named entities, derived from the tokens' IOB annotation (as `doc.ents`).
    pub fn ents(&self) -> Vec<Span<'_>> {
        let mut out: Vec<SpanC> = Vec::new();
        let mut start: Option<usize> = None;
        let (mut label, mut kb, mut id) = (0u64, 0u64, 0u64);
        for (i, t) in self.tokens.iter().enumerate() {
            if t.ent_iob == 1 {
                if start.is_none() {
                    // spaCy raises E093 here; be lenient and start an entity.
                    start = Some(i);
                    label = t.ent_type;
                    kb = t.ent_kb_id;
                    id = t.ent_id;
                }
            } else if t.ent_iob == 2 || t.ent_iob == 0 || (t.ent_iob == 3 && t.ent_type == 0) {
                if let Some(s) = start.take() {
                    out.push(SpanC {
                        start: s,
                        end: i,
                        label,
                        kb_id: kb,
                        id,
                    });
                }
                label = 0;
                kb = 0;
                id = 0;
            } else if t.ent_iob == 3 {
                if let Some(s) = start.take() {
                    out.push(SpanC {
                        start: s,
                        end: i,
                        label,
                        kb_id: kb,
                        id,
                    });
                }
                start = Some(i);
                label = t.ent_type;
                kb = t.ent_kb_id;
                id = t.ent_id;
            }
        }
        if let Some(s) = start {
            out.push(SpanC {
                start: s,
                end: self.len(),
                label,
                kb_id: kb,
                id,
            });
        }
        out.into_iter()
            .filter(|c| !self.vocab.strings.get_or_empty(c.label).is_empty())
            .map(|c| Span { doc: self, c })
            .collect()
    }

    /// Entity span data (owned).
    pub fn ent_spans(&self) -> Vec<SpanC> {
        self.ents().into_iter().map(|s| s.c).collect()
    }

    /// Set entities like `doc.ents = spans` (other tokens become `O`).
    pub fn set_ents(&mut self, ents: &[SpanC]) -> Result<(), crate::Error> {
        self.set_ents_with(ents, &[], &[], &[], EntsDefault::Outside)
    }

    /// `Doc.set_ents(entities, blocked=, missing=, outside=, default=)`.
    pub fn set_ents_with(
        &mut self,
        entities: &[SpanC],
        blocked: &[SpanC],
        missing: &[SpanC],
        outside: &[SpanC],
        default: EntsDefault,
    ) -> Result<(), crate::Error> {
        let entities: Vec<&SpanC> = entities.iter().filter(|e| e.label > 0).collect();
        let mut seen = vec![false; self.len()];
        for sp in entities
            .iter()
            .copied()
            .chain(blocked)
            .chain(missing)
            .chain(outside)
        {
            if sp.end > self.len() || sp.start > sp.end {
                return Err(crate::Error::Value(format!(
                    "span {}..{} out of range",
                    sp.start, sp.end
                )));
            }
            for i in sp.start..sp.end {
                if seen[i] {
                    return Err(crate::Error::Value(format!(
                        "[E1010] Unable to set entity information for token {i} which is included in more than one span in entities, blocked, missing or outside."
                    )));
                }
                seen[i] = true;
            }
        }
        for e in &entities {
            for i in e.start..e.end {
                let t = &mut self.tokens[i];
                t.ent_iob = if i == e.start { 3 } else { 1 };
                t.ent_type = e.label;
                t.ent_kb_id = e.kb_id;
                if e.id != 0 {
                    t.ent_id = e.id;
                }
            }
        }
        for (spans, iob) in [(blocked, 3u8), (missing, 0u8), (outside, 2u8)] {
            for sp in spans {
                for i in sp.start..sp.end {
                    self.tokens[i].ent_iob = iob;
                    self.tokens[i].ent_type = 0;
                }
            }
        }
        if default != EntsDefault::Unmodified {
            for (i, t) in self.tokens.iter_mut().enumerate() {
                if !seen[i] {
                    t.ent_type = 0;
                    t.ent_iob = match default {
                        EntsDefault::Outside => 2,
                        EntsDefault::Missing => 0,
                        _ => 3,
                    };
                }
            }
        }
        for i in 0..self.len().saturating_sub(1) {
            if (self.tokens[i].ent_iob == 0 || self.tokens[i].ent_iob == 2)
                && self.tokens[i + 1].ent_iob == 1
            {
                self.tokens[i + 1].ent_iob = 3;
            }
        }
        Ok(())
    }

    /// Set heads (absolute indices) and dependency labels, then recompute
    /// children, edges and sentence boundaries.
    pub fn set_heads<S: AsRef<str>>(
        &mut self,
        heads: &[usize],
        deps: &[S],
    ) -> Result<(), crate::Error> {
        if heads.len() != self.len() || deps.len() != self.len() {
            return Err(crate::Error::Value(
                "heads/deps length must equal doc length".into(),
            ));
        }
        for (i, (h, d)) in heads.iter().zip(deps).enumerate() {
            if *h >= self.len() {
                return Err(crate::Error::Value(format!("head {h} out of range")));
            }
            self.tokens[i].head = *h as isize - i as isize;
            self.tokens[i].dep = self.vocab.strings.add(d.as_ref());
        }
        crate::models::parser::set_children_from_heads(&mut self.tokens);
        Ok(())
    }

    // --------------------------------------------------------------- sentences

    /// Whether any sentence boundaries are set.
    pub fn has_sents(&self) -> bool {
        self.tokens.iter().skip(1).any(|t| t.sent_start != 0)
    }

    /// Sentences. Requires sentence boundaries (sentencizer, senter or parser);
    /// without them the whole doc is returned as one sentence.
    pub fn sents(&self) -> Vec<Span<'_>> {
        let mut out = Vec::new();
        let mut start = 0;
        for (i, t) in self.tokens.iter().enumerate().skip(1) {
            if t.sent_start == 1 {
                out.push(self.span(start, i));
                start = i;
            }
        }
        if start < self.len() {
            out.push(self.span(start, self.len()));
        }
        out
    }

    /// Whether any token has a value for the attribute (`TAG`, `DEP`, ...).
    pub fn has_annotation(&self, attr: u64) -> bool {
        match attr {
            SENT_START => self.tokens.iter().skip(1).any(|t| t.sent_start != 0),
            ENT_IOB | ENT_TYPE => self.tokens.iter().any(|t| t.ent_iob != 0),
            HEAD | DEP => self.tokens.iter().any(|t| t.dep != 0),
            _ => self.tokens.iter().any(|t| token_attr(t, attr) != 0),
        }
    }

    /// Export attribute values as rows, like `doc.to_array([...])`.
    pub fn to_array(&self, attrs: &[u64]) -> Vec<Vec<u64>> {
        self.tokens
            .iter()
            .map(|t| attrs.iter().map(|&a| token_attr(t, a)).collect())
            .collect()
    }

    /// Head index of token `i` (itself if unset).
    pub fn head_of(&self, i: usize) -> usize {
        (i as isize + self.tokens[i].head) as usize
    }

    /// Set the tensor (`tok2vec` output).
    pub fn set_tensor(&mut self, width: usize, data: Vec<f32>) {
        self.tensor = Some((width, data));
    }

    /// A JSON representation similar to `doc.to_json()`.
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        let st = &self.vocab.strings;
        let tokens: Vec<_> = self
            .iter()
            .map(|t| {
                let mut m = serde_json::Map::new();
                m.insert("id".into(), json!(t.i()));
                m.insert("start".into(), json!(t.idx()));
                m.insert("end".into(), json!(t.idx() + t.len_chars()));
                if t.c().tag != 0 {
                    m.insert("tag".into(), json!(t.tag_()));
                }
                if t.c().pos != 0 {
                    m.insert("pos".into(), json!(t.pos_()));
                }
                if t.c().morph != 0 {
                    m.insert("morph".into(), json!(t.morph_()));
                }
                if t.c().lemma != 0 {
                    m.insert("lemma".into(), json!(t.lemma_()));
                }
                if t.c().dep != 0 {
                    m.insert("dep".into(), json!(t.dep_()));
                    m.insert("head".into(), json!(t.head().i()));
                }
                serde_json::Value::Object(m)
            })
            .collect();
        let ents: Vec<_> = self
            .ents()
            .iter()
            .map(|e| json!({"start": e.start_char(), "end": e.end_char(), "label": e.label_()}))
            .collect();
        let mut out = json!({"text": self.text, "ents": ents, "tokens": tokens});
        if self.has_sents() {
            out["sents"] = self
                .sents()
                .iter()
                .map(|s| json!({"start": s.start_char(), "end": s.end_char()}))
                .collect();
        }
        let groups: serde_json::Map<String, serde_json::Value> = self
            .spans
            .iter()
            .map(|(k, v)| {
                let arr: Vec<_> = v
                    .iter()
                    .map(|c| {
                        let s = self.span_from(c);
                        json!({"start": s.start_char(), "end": s.end_char(), "label": st.get_or_empty(c.label), "kb_id": st.get_or_empty(c.kb_id)})
                    })
                    .collect();
                (k.clone(), json!(arr))
            })
            .collect();
        if !groups.is_empty() {
            out["spans"] = serde_json::Value::Object(groups);
        }
        out
    }
}

/// Numeric value of a token attribute, as spaCy's `Token.get_struct_attr`.
pub fn token_attr(t: &TokenC, attr: u64) -> u64 {
    match attr {
        LEMMA => t.lemma,
        NORM => {
            if t.norm == 0 {
                t.lex.norm
            } else {
                t.norm
            }
        }
        POS => t.pos,
        TAG => t.tag,
        DEP => t.dep,
        MORPH => t.morph,
        ENT_IOB => t.ent_iob as u64,
        ENT_TYPE => t.ent_type,
        ENT_ID => t.ent_id,
        ENT_KB_ID => t.ent_kb_id,
        HEAD => t.head as u64,
        SENT_START => t.sent_start as i64 as u64,
        SPACY => t.spacy as u64,
        IDX => t.idx as u64,
        _ => t.lex.get_attr(attr),
    }
}

/// A borrowed view of a token.
#[derive(Clone, Copy)]
pub struct Token<'a> {
    doc: &'a Doc,
    i: usize,
}

impl fmt::Debug for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({:?})", self.text())
    }
}
impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}
impl PartialEq for Token<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.doc, other.doc) && self.i == other.i
    }
}

macro_rules! flag_getters {
    ($($name:ident => $flag:ident),* $(,)?) => {
        $(
            #[doc = concat!("Lexical flag `", stringify!($flag), "`.")]
            pub fn $name(&self) -> bool { self.c().lex.check_flag($flag) }
        )*
    };
}

impl<'a> Token<'a> {
    /// The underlying token data.
    pub fn c(&self) -> &'a TokenC {
        &self.doc.tokens[self.i]
    }
    /// The doc this token belongs to.
    pub fn doc(&self) -> &'a Doc {
        self.doc
    }
    /// Index in the doc.
    pub fn i(&self) -> usize {
        self.i
    }
    fn s(&self, id: u64) -> &'a str {
        self.doc.vocab.strings.get_or_empty(id)
    }
    /// Verbatim text.
    pub fn text(&self) -> &'a str {
        let c = self.c();
        &self.doc.text[c.byte_idx..c.byte_idx + c.byte_len]
    }
    /// Text plus trailing space if present.
    pub fn text_with_ws(&self) -> &'a str {
        let c = self.c();
        &self.doc.text[c.byte_idx..c.byte_idx + c.byte_len + c.spacy as usize]
    }
    /// `" "` if followed by a space, else `""`.
    pub fn whitespace(&self) -> &'static str {
        if self.c().spacy { " " } else { "" }
    }
    /// Character offset in the doc.
    pub fn idx(&self) -> usize {
        self.c().idx
    }
    /// Byte offset in the doc.
    pub fn byte_idx(&self) -> usize {
        self.c().byte_idx
    }
    /// Length in characters.
    pub fn len_chars(&self) -> usize {
        self.c().lex.length
    }
    /// The lexeme.
    pub fn lex(&self) -> &'a LexemeC {
        &self.c().lex
    }
    /// ORTH ID.
    pub fn orth(&self) -> u64 {
        self.c().lex.orth
    }
    /// Lowercase form.
    pub fn lower_(&self) -> &'a str {
        self.s(self.c().lex.lower)
    }
    /// Normalized form (token-level override or lexeme NORM).
    pub fn norm_(&self) -> &'a str {
        self.s(self.norm())
    }
    /// NORM ID.
    pub fn norm(&self) -> u64 {
        token_attr(self.c(), NORM)
    }
    /// Word shape, e.g. `Xxxxx`.
    pub fn shape_(&self) -> &'a str {
        self.s(self.c().lex.shape)
    }
    /// First character.
    pub fn prefix_(&self) -> &'a str {
        self.s(self.c().lex.prefix)
    }
    /// Last three characters.
    pub fn suffix_(&self) -> &'a str {
        self.s(self.c().lex.suffix)
    }
    /// Language code.
    pub fn lang_(&self) -> &'a str {
        self.s(self.c().lex.lang)
    }
    /// Lemma (empty if unset).
    pub fn lemma_(&self) -> &'a str {
        self.s(self.c().lemma)
    }
    /// Fine-grained tag.
    pub fn tag_(&self) -> &'a str {
        self.s(self.c().tag)
    }
    /// Coarse universal POS.
    pub fn pos_(&self) -> &'a str {
        self.s(self.c().pos)
    }
    /// Morphological features (`Feat=Val|...`).
    pub fn morph_(&self) -> &'a str {
        match self.s(self.c().morph) {
            crate::morphology::EMPTY_MORPH => "",
            s => s,
        }
    }
    /// Dependency label.
    pub fn dep_(&self) -> &'a str {
        self.s(self.c().dep)
    }
    /// Entity type.
    pub fn ent_type_(&self) -> &'a str {
        self.s(self.c().ent_type)
    }
    /// Entity knowledge-base ID.
    pub fn ent_kb_id_(&self) -> &'a str {
        self.s(self.c().ent_kb_id)
    }
    /// Entity pattern ID.
    pub fn ent_id_(&self) -> &'a str {
        self.s(self.c().ent_id)
    }
    /// IOB code: `"B"`, `"I"`, `"O"` or `""`.
    pub fn ent_iob_(&self) -> &'static str {
        ["", "I", "O", "B"][self.c().ent_iob as usize & 3]
    }
    /// Value of any attribute by ID.
    pub fn get_attr(&self, attr: u64) -> u64 {
        token_attr(self.c(), attr)
    }
    /// Syntactic head (self for roots or unparsed docs).
    pub fn head(&self) -> Token<'a> {
        Token {
            doc: self.doc,
            i: self.doc.head_of(self.i),
        }
    }
    /// Left dependents (spaCy's `Token.lefts`, scanning from `l_edge`).
    pub fn lefts(&self) -> impl Iterator<Item = Token<'a>> + 'a {
        let doc = self.doc;
        let i = self.i;
        let le = self.c().l_edge.min(i);
        (le..i)
            .filter(move |&j| doc.head_of(j) == i)
            .map(move |j| Token { doc, i: j })
    }
    /// Right dependents (spaCy's `Token.rights`, scanning to `r_edge`).
    pub fn rights(&self) -> impl Iterator<Item = Token<'a>> + 'a {
        let doc = self.doc;
        let i = self.i;
        let re = self.c().r_edge.max(i).min(doc.len().saturating_sub(1));
        (i + 1..=re)
            .filter(move |&j| doc.head_of(j) == i)
            .map(move |j| Token { doc, i: j })
    }
    /// Direct dependents, left to right.
    pub fn children(&self) -> impl Iterator<Item = Token<'a>> + 'a {
        self.lefts().chain(self.rights())
    }
    /// Number of left children.
    pub fn n_lefts(&self) -> usize {
        self.lefts().count()
    }
    /// Number of right children.
    pub fn n_rights(&self) -> usize {
        self.rights().count()
    }
    /// Ancestors from head up to the root.
    pub fn ancestors(&self) -> Vec<Token<'a>> {
        let mut out = Vec::new();
        let mut cur = self.i;
        for _ in 0..self.doc.len() {
            let h = self.doc.head_of(cur);
            if h == cur {
                break;
            }
            out.push(Token {
                doc: self.doc,
                i: h,
            });
            cur = h;
        }
        out
    }
    /// Whether `self` is an ancestor of `other`.
    pub fn is_ancestor(&self, other: &Token<'_>) -> bool {
        other.ancestors().iter().any(|a| a.i == self.i)
    }
    /// The token and all its descendants, in order (`Token.subtree`).
    pub fn subtree(&self) -> Vec<Token<'a>> {
        let mut out = Vec::new();
        self.collect_subtree(&mut out, 0);
        out
    }
    fn collect_subtree(&self, out: &mut Vec<Token<'a>>, depth: usize) {
        if depth > self.doc.len() {
            return;
        }
        for l in self.lefts() {
            l.collect_subtree(out, depth + 1);
        }
        out.push(*self);
        for r in self.rights() {
            r.collect_subtree(out, depth + 1);
        }
    }
    /// Leftmost token of the subtree.
    pub fn left_edge(&self) -> Token<'a> {
        Token {
            doc: self.doc,
            i: self.c().l_edge.min(self.i),
        }
    }
    /// Rightmost token of the subtree.
    pub fn right_edge(&self) -> Token<'a> {
        Token {
            doc: self.doc,
            i: self.c().r_edge.max(self.i),
        }
    }
    /// Neighbouring token at offset `k`.
    pub fn nbor(&self, k: isize) -> Option<Token<'a>> {
        let j = self.i as isize + k;
        (j >= 0 && (j as usize) < self.doc.len()).then_some(Token {
            doc: self.doc,
            i: j as usize,
        })
    }
    /// Whether this token starts a sentence (`None` if unknown).
    pub fn is_sent_start(&self) -> Option<bool> {
        if self.i == 0 {
            return Some(true);
        }
        match self.c().sent_start {
            1 => Some(true),
            -1 => Some(false),
            _ => None,
        }
    }
    /// Whether this token ends a sentence (`None` if unknown).
    pub fn is_sent_end(&self) -> Option<bool> {
        if self.i + 1 == self.doc.len() {
            return Some(true);
        }
        self.doc.token(self.i + 1).is_sent_start()
    }
    /// The sentence containing this token.
    pub fn sent(&self) -> Span<'a> {
        let doc = self.doc;
        doc.sents()
            .into_iter()
            .find(|s| s.c.start <= self.i && self.i < s.c.end)
            .unwrap_or_else(|| doc.span(0, doc.len()))
    }
    /// Whether the token is out of vocabulary (no vector rank).
    pub fn is_oov(&self) -> bool {
        self.c().lex.id == crate::vocab::OOV_RANK
    }
    flag_getters! {
        is_alpha => IS_ALPHA, is_ascii => IS_ASCII, is_digit => IS_DIGIT, is_lower => IS_LOWER,
        is_upper => IS_UPPER, is_title => IS_TITLE, is_punct => IS_PUNCT, is_space => IS_SPACE,
        is_bracket => IS_BRACKET, is_quote => IS_QUOTE, is_left_punct => IS_LEFT_PUNCT,
        is_right_punct => IS_RIGHT_PUNCT, is_currency => IS_CURRENCY, like_url => LIKE_URL,
        like_num => LIKE_NUM, like_email => LIKE_EMAIL, is_stop => IS_STOP,
    }
}

/// A borrowed slice of a doc.
#[derive(Clone)]
pub struct Span<'a> {
    doc: &'a Doc,
    /// Offsets and labels.
    pub c: SpanC,
}

impl fmt::Debug for Span<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Span({}..{}, {:?}, {:?})",
            self.c.start,
            self.c.end,
            self.text(),
            self.label_()
        )
    }
}
impl fmt::Display for Span<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}

impl<'a> Span<'a> {
    /// The doc.
    pub fn doc(&self) -> &'a Doc {
        self.doc
    }
    /// First token index.
    pub fn start(&self) -> usize {
        self.c.start
    }
    /// End token index (exclusive).
    pub fn end(&self) -> usize {
        self.c.end
    }
    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.c.end - self.c.start
    }
    /// Whether the span is empty.
    pub fn is_empty(&self) -> bool {
        self.c.end == self.c.start
    }
    /// Start character offset.
    pub fn start_char(&self) -> usize {
        if self.is_empty() {
            return self
                .doc
                .tokens
                .get(self.c.start)
                .map(|t| t.idx)
                .unwrap_or_else(|| {
                    self.doc
                        .tokens
                        .last()
                        .map(|t| t.idx + t.lex.length + t.spacy as usize)
                        .unwrap_or(0)
                });
        }
        self.doc.tokens[self.c.start].idx
    }
    /// End character offset.
    pub fn end_char(&self) -> usize {
        if self.is_empty() {
            return self.start_char();
        }
        let t = &self.doc.tokens[self.c.end - 1];
        t.idx + t.lex.length
    }
    fn byte_range(&self) -> (usize, usize) {
        if self.is_empty() {
            let b = self
                .doc
                .tokens
                .get(self.c.start)
                .map(|t| t.byte_idx)
                .unwrap_or(self.doc.text.len());
            return (b, b);
        }
        let a = self.doc.tokens[self.c.start].byte_idx;
        let t = &self.doc.tokens[self.c.end - 1];
        (a, t.byte_idx + t.byte_len)
    }
    /// Text without trailing whitespace.
    pub fn text(&self) -> &'a str {
        let (a, b) = self.byte_range();
        &self.doc.text[a..b]
    }
    /// Text with the last token's trailing whitespace.
    pub fn text_with_ws(&self) -> &'a str {
        let (a, mut b) = self.byte_range();
        if !self.is_empty() && self.doc.tokens[self.c.end - 1].spacy {
            b += 1;
        }
        &self.doc.text[a..b]
    }
    /// Label string.
    pub fn label_(&self) -> &'a str {
        self.doc.vocab.strings.get_or_empty(self.c.label)
    }
    /// Knowledge-base ID string.
    pub fn kb_id_(&self) -> &'a str {
        self.doc.vocab.strings.get_or_empty(self.c.kb_id)
    }
    /// Pattern ID string.
    pub fn id_(&self) -> &'a str {
        self.doc.vocab.strings.get_or_empty(self.c.id)
    }
    /// Tokens in the span.
    pub fn iter(&self) -> impl Iterator<Item = Token<'a>> + 'a {
        let doc = self.doc;
        (self.c.start..self.c.end).map(move |i| doc.token(i))
    }
    /// Token at position `i` within the span.
    pub fn token(&self, i: usize) -> Token<'a> {
        assert!(i < self.len());
        self.doc.token(self.c.start + i)
    }
    /// The syntactic root: the token whose head lies outside the span (or itself).
    pub fn root(&self) -> Token<'a> {
        let mut best = self.c.start;
        let mut best_depth = usize::MAX;
        for i in self.c.start..self.c.end {
            let t = self.doc.token(i);
            let depth = t.ancestors().len();
            if depth < best_depth {
                best_depth = depth;
                best = i;
            }
        }
        self.doc.token(best)
    }
    /// Lemmas joined with the original whitespace.
    pub fn lemma_(&self) -> String {
        let mut s = String::new();
        for t in self.iter() {
            s.push_str(t.lemma_());
            s.push_str(t.whitespace());
        }
        s.trim_end().to_string()
    }
    /// Entities fully inside this span.
    pub fn ents(&self) -> Vec<Span<'a>> {
        self.doc
            .ents()
            .into_iter()
            .filter(|e| e.c.start >= self.c.start && e.c.end <= self.c.end)
            .collect()
    }
    /// Copy the span into a new doc.
    pub fn as_doc(&self) -> Doc {
        let mut tokens: Vec<TokenC> = self.doc.tokens[self.c.start..self.c.end].to_vec();
        let (a, _) = self.byte_range();
        let base_idx = self.start_char();
        let n = tokens.len();
        for (k, t) in tokens.iter_mut().enumerate() {
            t.byte_idx -= a;
            t.idx -= base_idx;
            let h = k as isize + t.head;
            if h < 0 || h >= n as isize {
                t.head = 0;
            }
        }
        if let Some(t) = tokens.first_mut() {
            t.sent_start = 1;
        }
        Doc::from_parts(
            self.doc.vocab.clone(),
            self.text_with_ws().to_string(),
            tokens,
        )
    }
}
