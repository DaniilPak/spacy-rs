//! Transition-based dependency parser and named entity recognizer
//! (`spacy.TransitionBasedParser.v2`, greedy decoding), ported from
//! `spacy/pipeline/_parser_internals` (`_state.pxd`, `arc_eager.pyx`,
//! `ner.pyx`, `nonproj.pyx`) and `spacy/ml/parser_model.pyx`.

use super::nn::{Linear, PrecomputableAffine};
use super::tok2vec::Tok2Vec;
use crate::Error;
use crate::attrs::IS_SPACE;
use crate::doc::{Doc, SpanC, TokenC};
use crate::language::Pipe;
use crate::strings::hash_string;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct ArcC {
    head: i32,
    child: i32,
    label: u64,
}

#[derive(Clone, Copy, Debug)]
struct EntC {
    start: i32,
    end: i32,
    label: u64,
}

/// Parser state (`StateC`).
struct State<'a> {
    sent: &'a [TokenC],
    length: i32,
    heads: Vec<i32>,
    stack: Vec<i32>,
    rebuffer: Vec<i32>,
    ents: Vec<EntC>,
    left_arcs: HashMap<i32, Vec<ArcC>>,
    right_arcs: HashMap<i32, Vec<ArcC>>,
    unshiftable: Vec<bool>,
    sent_starts: BTreeSet<i32>,
    b_i: i32,
}

impl<'a> State<'a> {
    fn new(sent: &'a [TokenC]) -> Self {
        let n = sent.len();
        State {
            sent,
            length: n as i32,
            heads: vec![-1; n],
            stack: Vec::new(),
            rebuffer: Vec::new(),
            ents: Vec::new(),
            left_arcs: HashMap::new(),
            right_arcs: HashMap::new(),
            unshiftable: vec![false; n],
            sent_starts: BTreeSet::new(),
            b_i: 0,
        }
    }
    fn s(&self, i: i32) -> i32 {
        if i < 0 || i as usize >= self.stack.len() {
            return -1;
        }
        self.stack[self.stack.len() - (i as usize + 1)]
    }
    fn b(&self, i: i32) -> i32 {
        if i < 0 {
            return -1;
        }
        let rb = self.rebuffer.len() as i32;
        if i < rb {
            return self.rebuffer[(rb - (i + 1)) as usize];
        }
        let b_i = self.b_i + (i - rb);
        if b_i >= self.length { -1 } else { b_i }
    }
    fn tok(&self, i: i32) -> Option<&TokenC> {
        if i < 0 || i >= self.length { None } else { Some(&self.sent[i as usize]) }
    }
    fn e(&self) -> i32 {
        self.ents.last().map_or(-1, |e| e.start)
    }
    fn nth_child(map: &HashMap<i32, Vec<ArcC>>, head: i32, idx: i32) -> i32 {
        if idx < 1 {
            return -1;
        }
        let Some(arcs) = map.get(&head) else { return -1 };
        let mut ci = 0;
        for a in arcs.iter().rev() {
            if a.child != -1 {
                ci += 1;
                if ci == idx {
                    return a.child;
                }
            }
        }
        -1
    }
    fn l(&self, head: i32, idx: i32) -> i32 {
        Self::nth_child(&self.left_arcs, head, idx)
    }
    fn r(&self, head: i32, idx: i32) -> i32 {
        Self::nth_child(&self.right_arcs, head, idx)
    }
    fn stack_depth(&self) -> i32 {
        self.stack.len() as i32
    }
    fn buffer_length(&self) -> i32 {
        (self.length - self.b_i) + self.rebuffer.len() as i32
    }
    fn eol(&self) -> bool {
        self.buffer_length() == 0
    }
    fn is_final(&self) -> bool {
        self.stack_depth() <= 0 && self.eol()
    }
    fn cannot_sent_start(&self, w: i32) -> bool {
        self.tok(w).is_some_and(|t| t.sent_start == -1)
    }
    fn is_sent_start(&self, w: i32) -> bool {
        match self.tok(w) {
            None => false,
            Some(t) => t.sent_start == 1 || self.sent_starts.contains(&w),
        }
    }
    fn has_head(&self, c: i32) -> bool {
        self.heads[c as usize] >= 0
    }
    fn entity_is_open(&self) -> bool {
        self.ents.last().is_some_and(|e| e.end == -1)
    }
    fn push(&mut self) {
        let b0 = if let Some(b) = self.rebuffer.pop() {
            b
        } else {
            let b = self.b_i;
            self.b_i += 1;
            b
        };
        self.stack.push(b0);
    }
    fn pop(&mut self) {
        self.stack.pop();
    }
    fn force_final(&mut self) {
        self.stack.clear();
        self.b_i = self.length;
    }
    fn unshift(&mut self) {
        let s0 = self.stack.pop().expect("non-empty stack");
        self.unshiftable[s0 as usize] = true;
        self.rebuffer.push(s0);
    }
    fn is_unshiftable(&self, item: i32) -> bool {
        item >= 0 && (item as usize) < self.unshiftable.len() && self.unshiftable[item as usize]
    }
    fn set_reshiftable(&mut self, item: i32) {
        if item >= 0 && (item as usize) < self.unshiftable.len() {
            self.unshiftable[item as usize] = false;
        }
    }
    fn del_arc(&mut self, h: i32, c: i32) {
        let map = if h > c { &mut self.left_arcs } else { &mut self.right_arcs };
        let Some(arcs) = map.get_mut(&h) else { return };
        let Some(last) = arcs.last() else { return };
        if last.head == h && last.child == c {
            arcs.pop();
        } else {
            let n = arcs.len();
            for a in arcs.iter_mut().take(n - 1) {
                if a.head == h && a.child == c {
                    a.head = -1;
                    a.child = -1;
                    a.label = 0;
                    break;
                }
            }
        }
    }
    fn add_arc(&mut self, head: i32, child: i32, label: u64) {
        if self.has_head(child) {
            self.del_arc(self.heads[child as usize], child);
        }
        let arc = ArcC { head, child, label };
        if head > child {
            self.left_arcs.entry(head).or_default().push(arc);
        } else {
            self.right_arcs.entry(head).or_default().push(arc);
        }
        self.heads[child as usize] = head;
    }
    fn open_ent(&mut self, label: u64) {
        let start = self.b(0);
        self.ents.push(EntC { start, end: -1, label });
    }
    fn close_ent(&mut self) {
        let b0 = self.b(0);
        if let Some(e) = self.ents.last_mut() {
            e.end = b0 + 1;
        }
    }
    fn arcs(&self) -> Vec<ArcC> {
        let mut out = Vec::new();
        for m in [&self.left_arcs, &self.right_arcs] {
            for arcs in m.values() {
                out.extend(arcs.iter().filter(|a| a.head != -1 && a.child != -1).copied());
            }
        }
        out
    }
    fn context_tokens(&self, n: usize, ids: &mut [i32]) {
        match n {
            1 => ids[0] = self.b(0),
            2 => {
                ids[0] = self.b(0);
                ids[1] = self.s(0);
            }
            3 => {
                ids[0] = self.b(0);
                ids[1] = if self.entity_is_open() { self.e() } else { -1 };
                ids[2] = if ids[0] == -1 || ids[1] == -1 { -1 } else { ids[0] - 1 };
            }
            8 => {
                ids[0] = self.b(0);
                ids[1] = self.b(1);
                ids[2] = self.s(0);
                ids[3] = self.s(1);
                ids[4] = self.s(2);
                ids[5] = self.l(self.b(0), 1);
                ids[6] = self.l(self.s(0), 1);
                ids[7] = self.r(self.s(0), 1);
            }
            13 => {
                ids[0] = self.b(0);
                ids[1] = self.b(1);
                ids[2] = self.s(0);
                ids[3] = self.s(1);
                ids[4] = self.s(2);
                ids[5] = self.l(self.s(0), 1);
                ids[6] = self.r(self.s(0), 1);
                ids[7] = self.l(self.b(0), 1);
                ids[8] = self.r(self.s(0), 2);
                ids[9] = self.l(self.s(1), 1);
                ids[10] = self.l(self.s(1), 2);
                ids[11] = self.r(self.s(1), 1);
                ids[12] = self.r(self.s(1), 2);
            }
            6 => {
                ids.iter_mut().for_each(|x| *x = -1);
                if self.b(0) >= 0 {
                    ids[0] = self.b(0);
                }
                if self.entity_is_open() {
                    let ent = *self.ents.last().expect("open ent");
                    let mut j = 1;
                    for i in ent.start..self.b(0) {
                        ids[j] = i;
                        j += 1;
                        if j >= 6 {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
        for x in ids.iter_mut().take(n) {
            if *x < 0 {
                *x = -1;
            }
        }
    }
}

// ------------------------------------------------------------------ moves

/// Which transition system a model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateType {
    /// Arc-eager dependency parsing.
    Parser,
    /// BILUO named entity recognition.
    Ner,
}

const SHIFT: u8 = 0;
const REDUCE: u8 = 1;
const LEFT: u8 = 2;
const RIGHT: u8 = 3;
const BREAK: u8 = 4;

const MISSING: u8 = 0;
const BEGIN: u8 = 1;
const IN: u8 = 2;
const LAST: u8 = 3;
const UNIT: u8 = 4;
const OUT: u8 = 5;

/// A transition: move type and label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    /// Move type (system-specific).
    pub mv: u8,
    /// Label ID.
    pub label: u64,
}

/// The ordered set of transitions (classes) of a model.
#[derive(Clone, Debug)]
pub struct Moves {
    /// Parser or NER.
    pub kind: StateType,
    /// Transitions in class order.
    pub c: Vec<Transition>,
    /// Label strings by ID.
    pub label_names: HashMap<u64, String>,
    subtok: u64,
}

impl Moves {
    /// Build from the `moves` JSON (`{"0": {"label": freq}, ...}`), replicating
    /// `TransitionSystem.initialize_actions` ordering.
    pub fn from_json(kind: StateType, moves_json: &str, strings: &crate::StringStore) -> Result<Self, Error> {
        let v: serde_json::Value = serde_json::from_str(moves_json).map_err(|e| Error::Format(format!("moves: {e}")))?;
        let obj = v.as_object().ok_or_else(|| Error::Format("moves must be an object".into()))?;
        let mut keys: Vec<&String> = obj.keys().collect();
        keys.sort();
        let mut m = Moves {
            kind,
            c: Vec::new(),
            label_names: HashMap::new(),
            subtok: hash_string("subtok"),
        };
        let mut added: Vec<(i64, String)> = Vec::new();
        let mut added_actions: HashMap<String, Vec<u8>> = HashMap::new();
        for k in keys {
            let action: u8 = k.parse().map_err(|_| Error::Format("bad action key".into()))?;
            let mut labels: Vec<(i64, String)> = obj[k]
                .as_object()
                .map(|o| o.iter().map(|(l, f)| (f.as_i64().unwrap_or(0), l.clone())).collect())
                .unwrap_or_default();
            labels.sort();
            labels.reverse();
            for (freq, label) in labels {
                if freq < 0 {
                    added.push((freq, label.clone()));
                    added_actions.entry(label).or_default().push(action);
                } else {
                    m.add_action(action, &label, strings);
                }
            }
        }
        added.sort();
        added.reverse();
        for (_, label) in added {
            for &a in &added_actions[&label] {
                m.add_action(a, &label, strings);
            }
        }
        Ok(m)
    }

    fn add_action(&mut self, action: u8, label: &str, strings: &crate::StringStore) {
        let id = strings.add(label);
        if self.kind == StateType::Ner && ((action == OUT && id != 0) || action == MISSING) {
            return;
        }
        if self.c.iter().any(|t| t.mv == action && t.label == id) {
            return;
        }
        self.label_names.insert(id, label.to_string());
        self.c.push(Transition { mv: action, label: id });
    }

    /// Number of classes.
    pub fn len(&self) -> usize {
        self.c.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// Human-readable class name, e.g. `"L-nsubj"` or `"B-ORG"`.
    pub fn class_name(&self, i: usize) -> String {
        let t = self.c[i];
        let l = self.label_names.get(&t.label).cloned().unwrap_or_default();
        match self.kind {
            StateType::Parser => {
                let n = ["S", "D", "L", "R", "B"][t.mv as usize];
                if l.is_empty() { n.to_string() } else { format!("{n}-{l}") }
            }
            StateType::Ner => match t.mv {
                OUT => "O".into(),
                MISSING => "M".into(),
                m => format!("{}-{l}", ["M", "B", "I", "L", "U", "O"][m as usize]),
            },
        }
    }

    fn set_valid(&self, st: &State<'_>, out: &mut [bool]) {
        match self.kind {
            StateType::Parser => {
                let v = [
                    shift_valid(st),
                    reduce_valid(st),
                    arc_valid(st, 0, self.subtok),
                    arc_valid(st, 0, self.subtok),
                    break_valid(st),
                ];
                for (i, t) in self.c.iter().enumerate() {
                    out[i] = if t.label == self.subtok && (t.mv == LEFT || t.mv == RIGHT) {
                        arc_valid(st, t.label, self.subtok)
                    } else {
                        v[t.mv as usize]
                    };
                }
            }
            StateType::Ner => {
                for (i, t) in self.c.iter().enumerate() {
                    out[i] = ner_valid(st, t.mv, t.label);
                }
            }
        }
    }

    fn apply(&self, st: &mut State<'_>, t: Transition) {
        match self.kind {
            StateType::Parser => match t.mv {
                SHIFT => st.push(),
                REDUCE => {
                    if st.has_head(st.s(0)) || st.stack_depth() == 1 {
                        st.pop();
                    } else {
                        st.unshift();
                    }
                }
                LEFT => {
                    let (b0, s0) = (st.b(0), st.s(0));
                    st.add_arc(b0, s0, t.label);
                    st.set_reshiftable(b0);
                    st.pop();
                }
                RIGHT => {
                    let (b0, s0) = (st.b(0), st.s(0));
                    st.add_arc(s0, b0, t.label);
                    st.push();
                }
                BREAK => {
                    let b1 = st.b(1);
                    st.sent_starts.insert(b1);
                }
                _ => {}
            },
            StateType::Ner => match t.mv {
                BEGIN => {
                    st.open_ent(t.label);
                    st.push();
                    st.pop();
                }
                LAST => {
                    st.close_ent();
                    st.push();
                    st.pop();
                }
                UNIT => {
                    st.open_ent(t.label);
                    st.close_ent();
                    st.push();
                    st.pop();
                }
                IN | OUT => {
                    st.push();
                    st.pop();
                }
                _ => {}
            },
        }
    }
}

fn shift_valid(st: &State<'_>) -> bool {
    if st.stack_depth() == 0 {
        true
    } else if st.buffer_length() < 2 {
        false
    } else {
        !(st.is_sent_start(st.b(0)) || st.is_unshiftable(st.b(0)))
    }
}

fn reduce_valid(st: &State<'_>) -> bool {
    if st.stack_depth() == 0 {
        false
    } else if st.buffer_length() == 0 {
        true
    } else {
        !(st.stack_depth() == 1 && st.cannot_sent_start(st.b(0)))
    }
}

fn arc_valid(st: &State<'_>, label: u64, subtok: u64) -> bool {
    if st.stack_depth() == 0 || st.buffer_length() == 0 || st.is_sent_start(st.b(0)) {
        false
    } else {
        !(label == subtok && st.s(0) != st.b(0) - 1)
    }
}

fn break_valid(st: &State<'_>) -> bool {
    if st.buffer_length() < 2 || st.b(1) != st.b(0) + 1 {
        return false;
    }
    !(st.is_sent_start(st.b(1)) || st.cannot_sent_start(st.b(1)))
}

fn ner_valid(st: &State<'_>, mv: u8, label: u64) -> bool {
    let empty = |i: i32| st.tok(st.b(i));
    let b0 = empty(0);
    let b1 = empty(1);
    let preset_iob = b0.map_or(0, |t| t.ent_iob);
    let preset_label = b0.map_or(0, |t| t.ent_type);
    let b1_iob = b1.map_or(0, |t| t.ent_iob);
    let b1_sent = b1.map_or(0, |t| t.sent_start);
    let b0_space = b0.is_some_and(|t| t.lex.check_flag(IS_SPACE));
    match mv {
        MISSING => false,
        BEGIN => {
            if st.entity_is_open() || st.buffer_length() < 2 || label == 0 || preset_iob == 1 {
                false
            } else if preset_iob == 3 {
                label == preset_label && b1_iob == 1
            } else {
                !(b1_iob == 3 || b1_sent == 1 || b0_space)
            }
        }
        IN => {
            if !st.entity_is_open() || st.buffer_length() < 2 {
                return false;
            }
            let ent = st.ents.last().expect("open");
            if label == 0 || ent.label != label || preset_iob == 3 || b1_iob == 3 {
                false
            } else if preset_iob == 1 {
                !(b1_iob == 0 || b1_iob == 2) && label == preset_label
            } else {
                !(st.b(1) != -1 && b1_sent == 1)
            }
        }
        LAST => {
            if label == 0 || !st.entity_is_open() {
                false
            } else if preset_iob == 1 && b1_iob != 1 {
                label == preset_label
            } else if st.ents.last().expect("open").label != label {
                false
            } else {
                b1_iob != 1
            }
        }
        UNIT => {
            if label == 0 {
                preset_label == 0 && preset_iob == 3
            } else if st.entity_is_open() || b1_iob == 1 {
                false
            } else if preset_iob == 3 {
                label == preset_label
            } else {
                !b0_space
            }
        }
        OUT => !(st.entity_is_open() || preset_iob == 3 || preset_iob == 1),
        _ => false,
    }
}

// ------------------------------------------------------------------ model

/// Where a transition model gets its token vectors from.
#[derive(Clone, Debug)]
pub enum Tok2VecSource {
    /// The shared `tok2vec` component (reads `doc.tensor`).
    Listener,
    /// An embedded encoder.
    Own(Arc<Tok2Vec>),
}

/// `TransitionBasedParser.v2` weights plus its transition system.
#[derive(Clone, Debug)]
pub struct TransitionModel {
    /// Token-vector source.
    pub tok2vec: Tok2VecSource,
    /// `Linear(hidden, t2v_width)` applied to token vectors.
    pub proj: Linear,
    /// Lower (feature) layer.
    pub lower: PrecomputableAffine,
    /// Upper (output) layer.
    pub upper: Linear,
    /// Transitions.
    pub moves: Moves,
}

impl TransitionModel {
    fn token_vectors(&self, doc: &Doc) -> Result<Vec<f32>, Error> {
        match &self.tok2vec {
            Tok2VecSource::Own(t) => Ok(t.forward(doc)),
            Tok2VecSource::Listener => {
                let (w, t) = doc
                    .tensor
                    .as_ref()
                    .ok_or_else(|| Error::Value("parser needs doc.tensor; add the tok2vec component first".into()))?;
                if *w != self.proj.n_in {
                    return Err(Error::Value("tensor width does not match parser".into()));
                }
                Ok(t.clone())
            }
        }
    }

    /// Greedy decoding; returns the final arcs (parser) or entities (NER).
    fn parse(&self, doc: &Doc) -> Result<(Vec<ArcC>, Vec<EntC>, BTreeSet<i32>), Error> {
        let n = doc.len();
        let tv = self.token_vectors(doc)?;
        let x = self.proj.forward(&tv, n);
        let l = &self.lower;
        let cached = l.precompute(&x, n);
        let (nf, no, np) = (l.n_f, l.n_o, l.n_p);
        let fo = no * np;
        let mut st = State::new(&doc.tokens);
        let nc = self.moves.len();
        let mut ids = vec![-1i32; nf];
        let mut unmaxed = vec![0f32; fo];
        let mut hidden = vec![0f32; no];
        let mut valid = vec![false; nc];
        while !st.is_final() {
            st.context_tokens(nf, &mut ids);
            unmaxed.iter_mut().for_each(|v| *v = 0.0);
            for f in 0..nf {
                let feat = if ids[f] < 0 {
                    &l.pad[f * fo..(f + 1) * fo]
                } else {
                    let base = ids[f] as usize * nf * fo + f * fo;
                    &cached[base..base + fo]
                };
                for (u, v) in unmaxed.iter_mut().zip(feat) {
                    *u += *v;
                }
            }
            for (u, b) in unmaxed.iter_mut().zip(&l.b) {
                *u += *b;
            }
            for j in 0..no {
                let p = &unmaxed[j * np..(j + 1) * np];
                let mut which = 0;
                for k in 1..np {
                    if p[k] > p[which] {
                        which = k;
                    }
                }
                hidden[j] = p[which];
            }
            let scores = self.upper.forward(&hidden, 1);
            self.moves.set_valid(&st, &mut valid);
            let mut best: Option<usize> = None;
            for i in 0..nc {
                if valid[i] && best.is_none_or(|b| scores[i] > scores[b]) {
                    best = Some(i);
                }
            }
            match best {
                None => st.force_final(),
                Some(g) => {
                    let t = self.moves.c[g];
                    self.moves.apply(&mut st, t);
                }
            }
        }
        Ok((st.arcs(), st.ents.clone(), st.sent_starts.clone()))
    }
}

/// The `parser` component (dependency parsing + sentence boundaries).
#[derive(Clone, Debug)]
pub struct DependencyParser {
    /// Model and moves.
    pub model: TransitionModel,
}

/// The `ner` component.
#[derive(Clone, Debug)]
pub struct EntityRecognizer {
    /// Model and moves.
    pub model: TransitionModel,
}

/// spaCy's `set_children_from_heads`: compute kids/edges and sentence starts.
pub fn set_children_from_heads(tokens: &mut [TokenC]) {
    let n = tokens.len();
    for (i, t) in tokens.iter_mut().enumerate() {
        t.l_kids = 0;
        t.r_kids = 0;
        t.l_edge = i;
        t.r_edge = i;
    }
    let mut loop_count = 0;
    loop {
        let ok = set_lr_kids_and_edges(tokens, loop_count);
        if ok || loop_count > 10 {
            break;
        }
        loop_count += 1;
    }
    for t in tokens.iter_mut() {
        t.sent_start = -1;
    }
    for i in 0..n {
        if tokens[i].head == 0 && !tokens[i].missing_head() {
            let le = tokens[i].l_edge;
            tokens[le].sent_start = 1;
        }
    }
}

fn set_lr_kids_and_edges(tokens: &mut [TokenC], loop_count: usize) -> bool {
    let n = tokens.len();
    for i in 0..n {
        let h = (i as isize + tokens[i].head) as usize;
        if loop_count == 0 && i < h {
            tokens[h].l_kids += 1;
        }
        let (cl, cr) = (tokens[i].l_edge, tokens[i].r_edge);
        if cl < tokens[h].l_edge {
            tokens[h].l_edge = cl;
        }
        if cr > tokens[h].r_edge {
            tokens[h].r_edge = cr;
        }
    }
    for i in (0..n).rev() {
        let h = (i as isize + tokens[i].head) as usize;
        if loop_count == 0 && i > h {
            tokens[h].r_kids += 1;
        }
        let (cl, cr) = (tokens[i].l_edge, tokens[i].r_edge);
        if cr > tokens[h].r_edge {
            tokens[h].r_edge = cr;
        }
        if cl < tokens[h].l_edge {
            tokens[h].l_edge = cl;
        }
    }
    let sent_starts: std::collections::HashSet<usize> =
        (0..n).filter(|&i| tokens[i].head == 0).map(|i| tokens[i].l_edge).collect();
    let mut cur_start = 0usize;
    for i in 0..n {
        if (i > 0 && sent_starts.contains(&i)) || i == n - 1 {
            let cur_end = i;
            for j in cur_start..cur_end {
                let h = tokens[j].head + j as isize;
                if h < cur_start as isize || h >= cur_end as isize + 1 {
                    return false;
                }
            }
            cur_start = i;
        }
    }
    true
}

/// `nonproj.deprojectivize`: re-attach arcs with decorated labels (`X||Y`).
pub fn deprojectivize(doc: &mut Doc) {
    let vocab = doc.vocab().clone();
    for i in 0..doc.len() {
        let label = vocab.strings.get_or_empty(doc.tokens[i].dep).to_string();
        if let Some((new_label, head_label)) = label.split_once("||") {
            let new_head = find_new_head(doc, i, head_label);
            doc.tokens[i].head = new_head as isize - i as isize;
            doc.tokens[i].dep = vocab.strings.add(new_label);
        }
    }
    set_children_from_heads(&mut doc.tokens);
}

fn find_new_head(doc: &Doc, token: usize, head_label: &str) -> usize {
    let st = &doc.vocab().strings;
    let head = doc.head_of(token);
    let mut queue = vec![head];
    let mut guard = 0;
    while !queue.is_empty() && guard < 10_000 {
        guard += 1;
        let mut next = Vec::new();
        for &q in &queue {
            for child in doc.token(q).children() {
                if child.is_space() || child.i() == token {
                    continue;
                }
                if st.get_or_empty(child.c().dep) == head_label {
                    return child.i();
                }
                next.push(child.i());
            }
        }
        queue = next;
    }
    head
}

impl Pipe for DependencyParser {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        if doc.is_empty() {
            return Ok(());
        }
        let (arcs, _, _) = self.model.parse(doc)?;
        for a in arcs {
            let c = a.child as usize;
            doc.tokens[c].head = a.head as isize - a.child as isize;
            doc.tokens[c].dep = a.label;
        }
        let root = doc.vocab().strings.add("ROOT");
        for t in doc.tokens.iter_mut() {
            if t.head == 0 {
                t.dep = root;
            }
        }
        set_children_from_heads(&mut doc.tokens);
        deprojectivize(doc);
        Ok(())
    }
}

impl Pipe for EntityRecognizer {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        if doc.is_empty() {
            return Ok(());
        }
        let (_, ents, _) = self.model.parse(doc)?;
        let spans: Vec<SpanC> = ents
            .iter()
            .filter(|e| e.start != -1 && e.end != -1)
            .map(|e| {
                let mut c = SpanC::new(e.start as usize, e.end as usize);
                c.label = e.label;
                c.kb_id = doc.tokens[e.start as usize].ent_kb_id;
                c
            })
            .collect();
        doc.set_ents_with(&spans, &[], &[], &[], crate::doc::EntsDefault::Unmodified)?;
        for t in doc.tokens.iter_mut() {
            if t.ent_iob == 0 {
                t.ent_iob = 2;
            }
        }
        Ok(())
    }
}
