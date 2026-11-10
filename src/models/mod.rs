//! Loading and running trained spaCy pipelines (CNN/`sm`-style models such as
//! `en_core_web_sm`) from their on-disk format, natively in Rust.
//!
//! Supported components: `tok2vec` (`Tok2Vec.v2` with `MultiHashEmbed.v2` and
//! `MaxoutWindowEncoder.v2`), `tagger` (`Tagger.v2`), `parser` and `ner`
//! (`TransitionBasedParser.v2`, greedy decoding), `attribute_ruler`,
//! `lemmatizer` (rule/lookup), `entity_ruler`, `sentencizer`.
//! Weights are never bundled: point [`load`] at an unpacked pipeline directory.

pub mod msgpack;
pub mod nn;
pub mod parser;
pub mod tagger;
pub mod tok2vec;

use crate::Error;
use crate::doc::Doc;
use crate::lang::{LexAttrConfig, SpecialToken};
use crate::language::{Language, Pipe};
use crate::pipeline::lemmatizer::{LemmaTables, Lemmatizer, LemmatizerMode};
use crate::pipeline::{
    AttributeRuler, AttributeRulerPattern, EntityRuler, RulerPattern, Sentencizer,
};
use crate::tokenizer::{Tokenizer, TokenizerConfig};
use crate::vocab::Vocab;
use msgpack::{as_str, get, int_key, to_json};
use parser::{
    DependencyParser, EntityRecognizer, Moves, StateType, Tok2VecSource, TransitionModel,
};
use rmpv::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tagger::Tagger;
use tok2vec::Tok2Vec;

/// The `tok2vec` component: writes token vectors to `doc.tensor`.
#[derive(Clone, Debug)]
pub struct Tok2VecPipe {
    /// The encoder.
    pub model: Arc<Tok2Vec>,
}

impl Pipe for Tok2VecPipe {
    fn process(&self, doc: &mut Doc) -> Result<(), Error> {
        let t = self.model.forward(doc);
        doc.set_tensor(self.model.width(), t);
        Ok(())
    }
}

/// A minimal reader for spaCy's `config.cfg` (INI with JSON-ish values).
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// `section -> key -> raw value`.
    pub sections: BTreeMap<String, BTreeMap<String, String>>,
}

impl Config {
    /// Parse config text.
    pub fn parse(text: &str) -> Self {
        let mut c = Config::default();
        let mut cur = String::new();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if l.starts_with('[') && l.ends_with(']') {
                cur = l[1..l.len() - 1].to_string();
                c.sections.entry(cur.clone()).or_default();
            } else if let Some((k, v)) = l.split_once('=') {
                c.sections
                    .entry(cur.clone())
                    .or_default()
                    .insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        c
    }
    /// A raw value.
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.sections.get(section)?.get(key).map(|s| s.as_str())
    }
    /// A value parsed as JSON (strings unquoted).
    pub fn get_json(&self, section: &str, key: &str) -> Option<serde_json::Value> {
        let raw = self.get(section, key)?;
        serde_json::from_str(raw)
            .ok()
            .or_else(|| Some(serde_json::Value::String(raw.to_string())))
    }
    /// A list of strings.
    pub fn get_list(&self, section: &str, key: &str) -> Vec<String> {
        match self.get_json(section, key) {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })
}

/// Resolve a pipeline path: either the data directory (with `config.cfg`) or
/// an installed package directory containing one versioned data directory.
fn resolve_dir(path: &Path) -> Result<PathBuf, Error> {
    if path.join("config.cfg").exists() {
        return Ok(path.to_path_buf());
    }
    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            if e.path().join("config.cfg").exists() {
                return Ok(e.path());
            }
        }
    }
    Err(Error::Format(format!(
        "{} does not contain a spaCy pipeline (config.cfg)",
        path.display()
    )))
}

fn load_tokenizer(
    vocab: &Arc<Vocab>,
    dir: &Path,
    lang_rules: Option<&crate::lang::LangData>,
) -> Result<Tokenizer, Error> {
    let p = dir.join("tokenizer");
    if !p.exists() {
        let d = lang_rules.ok_or_else(|| Error::Format("no tokenizer data".into()))?;
        return Tokenizer::for_lang(vocab.clone(), d);
    }
    let v = msgpack::decode(&read(&p)?)?;
    let s = |k: &str| get(&v, k).and_then(as_str).map(str::to_string);
    let mut rules = BTreeMap::new();
    if let Some(Value::Map(m)) = get(&v, "exceptions") {
        for (k, toks) in m {
            let key = as_str(k).unwrap_or("").to_string();
            let mut out = Vec::new();
            for t in toks.as_array().into_iter().flatten() {
                let mut orth = String::new();
                let mut norm = None;
                if let Value::Map(attrs) = t {
                    for (ak, av) in attrs {
                        let id = match ak {
                            Value::Integer(i) => i.as_u64().unwrap_or(0),
                            other => {
                                crate::attrs::attr_id(as_str(other).unwrap_or("")).unwrap_or(0)
                            }
                        };
                        match id {
                            crate::attrs::ORTH => orth = as_str(av).unwrap_or("").to_string(),
                            crate::attrs::NORM => norm = as_str(av).map(str::to_string),
                            _ => {}
                        }
                    }
                }
                out.push(SpecialToken { orth, norm });
            }
            rules.insert(key, out);
        }
    }
    let cfg = TokenizerConfig {
        prefix: s("prefix_search"),
        suffix: s("suffix_search"),
        infix: s("infix_finditer"),
        token_match: s("token_match"),
        url_match: s("url_match"),
        rules,
        faster_heuristics: get(&v, "faster_heuristics")
            .and_then(|b| b.as_bool())
            .unwrap_or(true),
    };
    Tokenizer::new(vocab.clone(), cfg)
}

fn load_transition(dir: &Path, kind: StateType, vocab: &Vocab) -> Result<TransitionModel, Error> {
    let nodes = msgpack::load_nodes(&read(&dir.join("model"))?)?;
    let root = nodes
        .first()
        .ok_or_else(|| Error::Format("empty parser model".into()))?;
    let rf = |name: &str| -> Result<usize, Error> {
        root.refs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| *i)
            .ok_or_else(|| Error::Format(format!("parser model has no {name} ref")))
    };
    let (li, ui) = (rf("lower")?, rf("upper")?);
    let lower = nn::PrecomputableAffine::from_node(&nodes[li])?;
    let upper = nn::Linear::from_node(&nodes[ui])?;
    let proj_node = nodes
        .iter()
        .enumerate()
        .filter(|(i, n)| n.name == "linear" && *i != ui)
        .map(|(_, n)| n)
        .next_back()
        .ok_or_else(|| Error::Format("parser model has no projection layer".into()))?;
    let proj = nn::Linear::from_node(proj_node)?;
    let tok2vec = if nodes.iter().any(|n| n.name == "hashembed") {
        Tok2VecSource::Own(Arc::new(Tok2Vec::from_nodes(&nodes)?))
    } else {
        Tok2VecSource::Listener
    };
    let mv = msgpack::decode(&read(&dir.join("moves"))?)?;
    let moves_json = get(&mv, "moves")
        .and_then(as_str)
        .ok_or_else(|| Error::Format("moves file without moves".into()))?;
    let moves = Moves::from_json(kind, moves_json, &vocab.strings)?;
    if moves.len() != upper.n_out {
        return Err(Error::Format(format!(
            "{} moves but output layer has {} classes",
            moves.len(),
            upper.n_out
        )));
    }
    Ok(TransitionModel {
        tok2vec,
        proj,
        lower,
        upper,
        moves,
    })
}

fn load_lemma_tables(path: &Path) -> Result<LemmaTables, Error> {
    let v = msgpack::decode(&read(path)?)?;
    let mut t = LemmaTables::default();
    let map = |name: &str| -> Vec<(u64, &Value)> {
        match get(&v, name) {
            Some(Value::Map(m)) => m
                .iter()
                .filter_map(|(k, v)| Some((int_key(k)?, v)))
                .collect(),
            _ => Vec::new(),
        }
    };
    for (k, val) in map("lemma_lookup") {
        if let Some(s) = as_str(val) {
            t.lookup.insert(k, s.to_string());
        }
    }
    for (k, val) in map("lemma_rules") {
        let rs = val
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| {
                let p = p.as_array()?;
                Some((
                    as_str(p.first()?)?.to_string(),
                    as_str(p.get(1)?)?.to_string(),
                ))
            })
            .collect();
        t.rules.insert(k, rs);
    }
    for (k, val) in map("lemma_exc") {
        let mut h = HashMap::new();
        if let Value::Map(m) = val {
            for (f, ls) in m {
                let ls = ls
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|x| as_str(x).map(str::to_string))
                    .collect();
                h.insert(as_str(f).unwrap_or("").to_string(), ls);
            }
        }
        t.exc.insert(k, h);
    }
    for (k, val) in map("lemma_index") {
        let set = val
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|x| as_str(x).map(str::to_string))
            .collect();
        t.index.insert(k, set);
    }
    Ok(t)
}

/// Load a trained pipeline from a directory, like `spacy.load(path)`.
///
/// `path` may be the pipeline data directory (containing `config.cfg`, e.g.
/// `.../en_core_web_sm/en_core_web_sm-3.8.0`) or its parent package directory.
pub fn load(path: impl AsRef<Path>) -> Result<Language, Error> {
    load_with(path, &[])
}

/// Like [`load`], but skip the named components (`exclude=[...]`).
pub fn load_with(path: impl AsRef<Path>, exclude: &[&str]) -> Result<Language, Error> {
    let dir = resolve_dir(path.as_ref())?;
    let cfg = Config::parse(&String::from_utf8_lossy(&read(&dir.join("config.cfg"))?));
    let lang = cfg
        .get_json("nlp", "lang")
        .and_then(|v| v.as_str().map(str::to_string))
        .ok_or_else(|| Error::Format("config has no nlp.lang".into()))?;
    let data = crate::lang::load(&lang).ok_or_else(|| Error::UnknownLanguage(lang.clone()))?;
    let vocab = Arc::new(Vocab::new(&lang, LexAttrConfig::for_lang(&data)));
    let lk = dir.join("vocab").join("lookups.bin");
    if lk.exists() {
        let v = msgpack::decode(&read(&lk)?)?;
        if let Some(Value::Map(m)) = get(&v, "lexeme_norm") {
            let table: HashMap<u64, String> = m
                .iter()
                .filter_map(|(k, v)| Some((int_key(k)?, as_str(v)?.to_string())))
                .collect();
            vocab.add_norm_table_ids(table);
        }
    }
    let tokenizer = load_tokenizer(&vocab, &dir, Some(&data))?;
    let mut nlp = Language::from_parts(vocab.clone(), tokenizer);
    let disabled = cfg.get_list("nlp", "disabled");
    for name in cfg.get_list("nlp", "pipeline") {
        if disabled.contains(&name) || exclude.contains(&name.as_str()) {
            continue;
        }
        let sec = format!("components.{name}");
        let factory = cfg
            .get_json(&sec, "factory")
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| name.clone());
        let cdir = dir.join(&name);
        match factory.as_str() {
            "tok2vec" => {
                let nodes = msgpack::load_nodes(&read(&cdir.join("model"))?)?;
                nlp.add_pipe(
                    name,
                    Tok2VecPipe {
                        model: Arc::new(Tok2Vec::from_nodes(&nodes)?),
                    },
                );
            }
            "tagger" => {
                let nodes = msgpack::load_nodes(&read(&cdir.join("model"))?)?;
                let sm = nodes
                    .iter()
                    .find(|n| n.name == "softmax")
                    .ok_or_else(|| Error::Format("tagger without softmax".into()))?;
                let c: serde_json::Value = serde_json::from_slice(&read(&cdir.join("cfg"))?)
                    .map_err(|e| Error::Format(format!("tagger cfg: {e}")))?;
                let labels = c["labels"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                if nodes.iter().any(|n| n.name == "hashembed") {
                    return Err(Error::Format(
                        "taggers with their own tok2vec are not supported yet".into(),
                    ));
                }
                nlp.add_pipe(
                    name,
                    Tagger {
                        output: nn::Linear::from_node(sm)?,
                        labels,
                        overwrite: c["overwrite"].as_bool().unwrap_or(false),
                    },
                );
            }
            "parser" => {
                let model = load_transition(&cdir, StateType::Parser, &vocab)?;
                nlp.add_pipe(name, DependencyParser { model });
            }
            "ner" => {
                let model = load_transition(&cdir, StateType::Ner, &vocab)?;
                nlp.add_pipe(name, EntityRecognizer { model });
            }
            "attribute_ruler" => {
                let mut ar = AttributeRuler::new(&vocab);
                let p = cdir.join("patterns");
                if p.exists() {
                    let v = to_json(&msgpack::decode(&read(&p)?)?);
                    let pats: Vec<AttributeRulerPattern> = serde_json::from_value(v)
                        .map_err(|e| Error::Format(format!("attribute_ruler patterns: {e}")))?;
                    ar.add_patterns(&pats)?;
                }
                nlp.add_pipe(name, ar);
            }
            "lemmatizer" => {
                let mode = match cfg
                    .get_json(&sec, "mode")
                    .and_then(|v| v.as_str().map(str::to_string))
                    .as_deref()
                {
                    Some("lookup") => LemmatizerMode::Lookup,
                    Some("rule") | None => LemmatizerMode::Rule,
                    Some(other) => {
                        return Err(Error::Format(format!(
                            "unsupported lemmatizer mode {other}"
                        )));
                    }
                };
                let tables = load_lemma_tables(&cdir.join("lookups").join("lookups.bin"))?;
                let mut l = Lemmatizer::with_tables(mode, Arc::new(tables), lang == "en");
                l.overwrite = cfg.get(&sec, "overwrite") == Some("true");
                nlp.add_pipe(name, l);
            }
            "sentencizer" => nlp.add_pipe(name, Sentencizer::new()),
            "entity_ruler" => {
                let mut r = EntityRuler::new(&nlp);
                let p = cdir.join("patterns.jsonl");
                if p.exists() {
                    let pats: Vec<RulerPattern> =
                        crate::pipeline::read_jsonl_patterns(&String::from_utf8_lossy(&read(&p)?))?;
                    r.add_patterns(&nlp, &pats)?;
                }
                nlp.add_pipe(name, r);
            }
            other => {
                return Err(Error::Format(format!(
                    "component {name:?} (factory {other:?}) is not supported by spacy-rs yet; pass it in `exclude`"
                )));
            }
        }
    }
    Ok(nlp)
}
