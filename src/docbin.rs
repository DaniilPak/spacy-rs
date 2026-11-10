//! `DocBin`: spaCy's compact binary serialization for collections of docs
//! (zlib-compressed msgpack, `.spacy` files), readable and writable by both
//! spaCy and spacy-rs.

use crate::Error;
use crate::attrs::*;
use crate::doc::{Doc, SpanC};
use crate::models::msgpack::{self, as_str, get};
use crate::vocab::Vocab;
use rmpv::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::sync::Arc;

/// Attributes serialized by default (spaCy's `ALL_ATTRS`).
pub const DEFAULT_ATTRS: [u64; 13] = [
    ORTH, NORM, TAG, HEAD, DEP, ENT_IOB, ENT_TYPE, ENT_KB_ID, ENT_ID, LEMMA, MORPH, POS, SENT_START,
];

#[derive(Clone, Debug)]
struct DocData {
    tokens: Vec<Vec<u64>>,
    spaces: Vec<bool>,
    has_unknown_spaces: bool,
    span_groups: Vec<u8>,
    cats: Value,
    user_data: Option<Vec<u8>>,
}

/// A collection of docs in spaCy's `DocBin` format.
#[derive(Clone, Debug)]
pub struct DocBin {
    attrs: Vec<u64>,
    docs: Vec<DocData>,
    strings: BTreeSet<String>,
}

impl Default for DocBin {
    fn default() -> Self {
        Self::new(&DEFAULT_ATTRS)
    }
}

fn span_group_bytes(doc: &Doc, name: &str, spans: &[SpanC]) -> Vec<u8> {
    let mut packed = Vec::new();
    for c in spans {
        let s = doc.span_from(c);
        let mut b = Vec::with_capacity(40);
        b.extend_from_slice(&c.id.to_be_bytes());
        b.extend_from_slice(&c.kb_id.to_be_bytes());
        b.extend_from_slice(&c.label.to_be_bytes());
        b.extend_from_slice(&(c.start as i32).to_be_bytes());
        b.extend_from_slice(&(c.end as i32).to_be_bytes());
        b.extend_from_slice(&(s.start_char() as i32).to_be_bytes());
        b.extend_from_slice(&(s.end_char() as i32).to_be_bytes());
        packed.push(Value::Binary(b));
    }
    let v = Value::Map(vec![
        (Value::from("name"), Value::from(name)),
        (Value::from("attrs"), Value::Map(vec![])),
        (Value::from("spans"), Value::Array(packed)),
    ]);
    encode(&v)
}

fn encode(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, v).expect("write to Vec");
    out
}

fn empty_list_bytes() -> Vec<u8> {
    encode(&Value::Array(vec![]))
}

fn parse_span_group(bytes: &[u8]) -> Result<(String, Vec<SpanC>), Error> {
    let v = msgpack::decode(bytes)?;
    let name = get(&v, "name").and_then(as_str).unwrap_or("").to_string();
    let mut spans = Vec::new();
    for s in get(&v, "spans")
        .and_then(|s| s.as_array())
        .into_iter()
        .flatten()
    {
        let Value::Binary(b) = s else { continue };
        if b.len() != 40 {
            return Err(Error::Format("bad span record".into()));
        }
        let u = |i: usize| u64::from_be_bytes(b[i..i + 8].try_into().expect("8"));
        let l = |i: usize| i32::from_be_bytes(b[i..i + 4].try_into().expect("4"));
        spans.push(SpanC {
            id: u(0),
            kb_id: u(8),
            label: u(16),
            start: l(24).max(0) as usize,
            end: l(28).max(0) as usize,
        });
    }
    Ok((name, spans))
}

impl DocBin {
    /// A DocBin storing the given attributes (`ORTH` is always first).
    pub fn new(attrs: &[u64]) -> Self {
        let mut a: Vec<u64> = attrs
            .iter()
            .copied()
            .filter(|&x| x != ORTH && x != SPACY)
            .collect();
        a.sort_unstable();
        a.dedup();
        a.insert(0, ORTH);
        DocBin {
            attrs: a,
            docs: Vec::new(),
            strings: BTreeSet::new(),
        }
    }

    /// Number of docs.
    pub fn len(&self) -> usize {
        self.docs.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
    /// The stored attribute IDs.
    pub fn attrs(&self) -> &[u64] {
        &self.attrs
    }

    /// Add a doc.
    pub fn add(&mut self, doc: &Doc) {
        let tokens = doc.to_array(&self.attrs);
        let spaces = doc.tokens.iter().map(|t| t.spacy).collect();
        let st = &doc.vocab().strings;
        for t in doc.iter() {
            for s in [
                t.text(),
                t.tag_(),
                t.lemma_(),
                t.dep_(),
                t.ent_type_(),
                t.ent_kb_id_(),
                t.ent_id_(),
                t.norm_(),
            ] {
                self.strings.insert(s.to_string());
            }
            self.strings.insert(t.morph_().to_string());
        }
        let span_groups = if doc.spans.is_empty() {
            empty_list_bytes()
        } else {
            let mut entries: Vec<(Value, Value)> = Vec::new();
            for (k, v) in &doc.spans {
                for c in v {
                    for id in [c.label, c.kb_id, c.id] {
                        if let Some(s) = st.get(id) {
                            self.strings.insert(s.to_string());
                        }
                    }
                }
                entries.push((
                    Value::Binary(span_group_bytes(doc, k, v)),
                    Value::Array(vec![Value::from(k.as_str())]),
                ));
            }
            encode(&Value::Map(entries))
        };
        self.docs.push(DocData {
            tokens,
            spaces,
            has_unknown_spaces: doc.has_unknown_spaces,
            span_groups,
            cats: Value::Map(vec![]),
            user_data: None,
        });
    }

    /// Reconstruct the docs with a vocab.
    pub fn get_docs(&self, vocab: &Arc<Vocab>) -> Result<Vec<Doc>, Error> {
        for s in &self.strings {
            vocab.strings.add(s);
        }
        let mut out = Vec::with_capacity(self.docs.len());
        for d in &self.docs {
            let words: Vec<String> = d
                .tokens
                .iter()
                .map(|row| {
                    vocab
                        .strings
                        .get(row[0])
                        .map(str::to_string)
                        .ok_or_else(|| Error::Format(format!("unknown string id {}", row[0])))
                })
                .collect::<Result<_, _>>()?;
            let spaces = (!d.has_unknown_spaces).then_some(d.spaces.as_slice());
            let mut doc = Doc::from_words(vocab.clone(), &words, spaces);
            doc.from_array(&self.attrs, &d.tokens)?;
            if !d.span_groups.is_empty() && d.span_groups != empty_list_bytes() {
                match msgpack::decode(&d.span_groups)? {
                    Value::Array(groups) => {
                        for g in groups {
                            if let Value::Binary(b) = g {
                                let (name, spans) = parse_span_group(&b)?;
                                doc.spans.insert(name, spans);
                            }
                        }
                    }
                    Value::Map(m) => {
                        for (bytes, keys) in m {
                            let Value::Binary(b) = bytes else { continue };
                            let (_, spans) = parse_span_group(&b)?;
                            for k in keys.as_array().into_iter().flatten() {
                                doc.spans
                                    .insert(as_str(k).unwrap_or("").to_string(), spans.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
            out.push(doc);
        }
        Ok(out)
    }

    /// Serialize (zlib-compressed msgpack), like `DocBin.to_bytes()`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut tokens = Vec::new();
        let mut spaces = Vec::new();
        let mut lengths = Vec::new();
        for d in &self.docs {
            for row in &d.tokens {
                for v in row {
                    tokens.extend_from_slice(&v.to_le_bytes());
                }
            }
            spaces.extend(d.spaces.iter().map(|&b| b as u8));
            lengths.extend_from_slice(&(d.tokens.len() as i32).to_le_bytes());
        }
        let mut msg = vec![
            (Value::from("version"), Value::from("0.1")),
            (
                Value::from("attrs"),
                Value::Array(self.attrs.iter().map(|&a| Value::from(a)).collect()),
            ),
            (Value::from("tokens"), Value::Binary(tokens)),
            (Value::from("spaces"), Value::Binary(spaces)),
            (Value::from("lengths"), Value::Binary(lengths)),
            (
                Value::from("strings"),
                Value::Array(
                    self.strings
                        .iter()
                        .map(|s| Value::from(s.as_str()))
                        .collect(),
                ),
            ),
            (
                Value::from("cats"),
                Value::Array(self.docs.iter().map(|d| d.cats.clone()).collect()),
            ),
            (
                Value::from("flags"),
                Value::Array(
                    self.docs
                        .iter()
                        .map(|d| {
                            Value::Map(vec![(
                                Value::from("has_unknown_spaces"),
                                Value::from(d.has_unknown_spaces),
                            )])
                        })
                        .collect(),
                ),
            ),
            (
                Value::from("span_groups"),
                Value::Array(
                    self.docs
                        .iter()
                        .map(|d| Value::Binary(d.span_groups.clone()))
                        .collect(),
                ),
            ),
        ];
        if self.docs.iter().any(|d| d.user_data.is_some()) {
            msg.push((
                Value::from("user_data"),
                Value::Array(
                    self.docs
                        .iter()
                        .map(|d| Value::Binary(d.user_data.clone().unwrap_or_default()))
                        .collect(),
                ),
            ));
        }
        let raw = encode(&Value::Map(msg));
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&raw).expect("write to Vec");
        enc.finish().expect("zlib")
    }

    /// Deserialize, like `DocBin().from_bytes(data)`.
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        let mut raw = Vec::new();
        flate2::read::ZlibDecoder::new(data)
            .read_to_end(&mut raw)
            .map_err(|e| Error::Format(format!("zlib: {e}")))?;
        let v = msgpack::decode(&raw)?;
        let attrs: Vec<u64> = get(&v, "attrs")
            .and_then(|a| a.as_array())
            .ok_or_else(|| Error::Format("DocBin without attrs".into()))?
            .iter()
            .map(|a| a.as_u64().unwrap_or(0))
            .collect();
        let bin = |k: &str| -> Vec<u8> {
            match get(&v, k) {
                Some(Value::Binary(b)) => b.clone(),
                _ => Vec::new(),
            }
        };
        let tokens = bin("tokens");
        let spaces = bin("spaces");
        let lengths: Vec<usize> = bin("lengths")
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]).max(0) as usize)
            .collect();
        let na = attrs.len().max(1);
        let flat: Vec<u64> = tokens
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
            .collect();
        let strings: BTreeSet<String> = get(&v, "strings")
            .and_then(|s| s.as_array())
            .into_iter()
            .flatten()
            .filter_map(|s| as_str(s).map(str::to_string))
            .collect();
        let arr = |k: &str| {
            get(&v, k)
                .and_then(|x| x.as_array())
                .cloned()
                .unwrap_or_default()
        };
        let cats = arr("cats");
        let flags = arr("flags");
        let groups = arr("span_groups");
        let user = arr("user_data");
        let mut docs = Vec::new();
        let mut off = 0;
        for (i, &n) in lengths.iter().enumerate() {
            if (off + n) * na > flat.len() || off + n > spaces.len() {
                return Err(Error::Format("DocBin data truncated".into()));
            }
            let toks = (off..off + n)
                .map(|r| flat[r * na..(r + 1) * na].to_vec())
                .collect();
            let sp = spaces[off..off + n].iter().map(|&b| b != 0).collect();
            off += n;
            docs.push(DocData {
                tokens: toks,
                spaces: sp,
                has_unknown_spaces: flags
                    .get(i)
                    .and_then(|f| get(f, "has_unknown_spaces"))
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false),
                span_groups: match groups.get(i) {
                    Some(Value::Binary(b)) => b.clone(),
                    _ => Vec::new(),
                },
                cats: cats.get(i).cloned().unwrap_or(Value::Map(vec![])),
                user_data: match user.get(i) {
                    Some(Value::Binary(b)) => Some(b.clone()),
                    _ => None,
                },
            });
        }
        Ok(DocBin {
            attrs,
            docs,
            strings,
        })
    }

    /// Write a `.spacy` file.
    pub fn to_disk(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }

    /// Read a `.spacy` file.
    pub fn from_disk(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Text categories per doc (as JSON), if any were stored by spaCy.
    pub fn cats(&self) -> Vec<serde_json::Value> {
        self.docs
            .iter()
            .map(|d| msgpack::to_json(&d.cats))
            .collect()
    }
}

impl Doc {
    /// Set token attributes from rows of values, like `Doc.from_array`.
    pub fn from_array(&mut self, attrs: &[u64], rows: &[Vec<u64>]) -> Result<(), Error> {
        if rows.len() != self.len() {
            return Err(Error::Value(format!(
                "[E971] Found incompatible lengths: array has {} rows, doc has {} tokens",
                rows.len(),
                self.len()
            )));
        }
        let n = self.len() as i64;
        if let Some(col) = attrs.iter().position(|&a| a == HEAD) {
            for (i, r) in rows.iter().enumerate() {
                let abs = r[col] as u32 as i32 as i64 + i as i64;
                if abs < 0 || abs >= n {
                    return Err(Error::Value(format!(
                        "[E190] Token head out of range at index {i}"
                    )));
                }
            }
        }
        let vocab = self.vocab().clone();
        for (t, r) in self.tokens.iter_mut().zip(rows) {
            for (&a, &v) in attrs.iter().zip(r) {
                match a {
                    LEMMA => t.lemma = v,
                    NORM => t.norm = v,
                    POS => t.pos = v,
                    TAG => t.tag = v,
                    DEP => t.dep = v,
                    HEAD => t.head = v as u32 as i32 as isize,
                    SENT_START => t.sent_start = v as i8,
                    SPACY => t.spacy = v != 0,
                    ENT_IOB => t.ent_iob = v as u8,
                    ENT_TYPE => t.ent_type = v,
                    ENT_ID => t.ent_id = v,
                    ENT_KB_ID => t.ent_kb_id = v,
                    MORPH => {
                        t.morph = if v == 0 {
                            0
                        } else {
                            let s = vocab.strings.get_or_empty(v).to_string();
                            vocab.strings.add(&crate::morphology::normalize(&s))
                        }
                    }
                    _ => {}
                }
            }
        }
        let has_head = attrs.contains(&HEAD);
        if let Some(col) = attrs.iter().position(|&a| a == DEP)
            && has_head
            && rows.iter().any(|r| r[col] != 0)
        {
            crate::models::parser::set_children_from_heads(&mut self.tokens);
        }
        Ok(())
    }
}

/// Merge string tables (helper used by tests).
pub fn strings_of(bin: &DocBin) -> BTreeMap<u64, String> {
    bin.strings
        .iter()
        .map(|s| (crate::strings::get_string_id(s), s.clone()))
        .collect()
}
