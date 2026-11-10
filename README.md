# spacy-rs

[![CI](https://github.com/DaniilPak/spacy-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/DaniilPak/spacy-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/spacy-rs.svg)](https://crates.io/crates/spacy-rs)
[![docs.rs](https://docs.rs/spacy-rs/badge.svg)](https://docs.rs/spacy-rs)

**Industrial-strength NLP in pure Rust**: a port of [spaCy](https://github.com/explosion/spaCy)'s
tokenizer, `Doc`/`Token`/`Span` model, rule-based matchers and components, and **native
inference for trained spaCy pipelines** such as `en_core_web_sm` — no Python, no bindings,
no ONNX.

> spacy-rs is an independent project. It is **not affiliated with, endorsed by, or supported
> by Explosion** (the makers of spaCy). "spaCy" is used here only to describe compatibility.

Everything is tested for **exact parity** against real spaCy 3.8 on golden fixtures:

* the tokenizer reproduces spaCy's output token-for-token (and all lexical attributes) on
  ~40,000 tokens of hand-written, harvested and randomly fuzzed text in 8 languages;
* `en_core_web_sm` 3.8.0 run natively in Rust gives **identical** tags, POS, morphology,
  lemmas, dependency heads and labels, sentence boundaries, noun chunks and named entities
  on a 1,020-document / 10,280-token test set (100.000% on every attribute).

## Install

```toml
[dependencies]
spacy-rs = "0.1"
```

The library is imported as `spacy`. Model weights are never bundled; trained pipelines are
loaded from a directory you provide (see [Trained pipelines](#trained-pipelines)).

## Quick start

### Tokenization and lexical attributes

```rust
let nlp = spacy::blank("en").unwrap();
let doc = nlp.make_doc("I can't visit https://example.com today, e.g. at 5pm!");

let words: Vec<&str> = doc.iter().map(|t| t.text()).collect();
assert_eq!(
    words,
    ["I", "ca", "n't", "visit", "https://example.com", "today", ",", "e.g.", "at", "5", "pm", "!"]
);
let t = doc.token(2);
assert_eq!((t.norm_(), t.is_stop(), t.shape_()), ("not", true, "x'x"));
assert!(doc.token(4).like_url());
assert_eq!(doc.token(4).idx(), 14); // character offsets, like spaCy
```

### Matcher (token patterns)

Patterns use spaCy's JSON format, including operators (`?`, `*`, `+`, `!`, `{n,m}`) and
predicates (`IN`, `NOT_IN`, `REGEX`, `FUZZY`, `>=`, `IS_SUBSET`, ...):

```rust
use serde_json::json;
use spacy::matcher::Matcher;

let nlp = spacy::blank("en").unwrap();
let mut matcher = Matcher::new(nlp.vocab());
matcher
    .add(
        "HELLO_WORLD",
        &[json!([{"LOWER": "hello"}, {"IS_PUNCT": true, "OP": "?"}, {"LOWER": {"IN": ["world", "there"]}}])],
        None,
    )
    .unwrap();
let doc = nlp.make_doc("Hello, world! Hello there.");
let spans: Vec<String> = matcher.find_spans(&doc).unwrap().iter().map(|s| s.text().to_string()).collect();
assert_eq!(spans, ["Hello, world", "Hello there"]);
```

### PhraseMatcher, EntityRuler, Sentencizer, Lemmatizer

```rust
use serde_json::json;
use spacy::pipeline::{EntityRuler, Lemmatizer, LemmatizerMode, RulerPattern, Sentencizer};

let mut nlp = spacy::blank("en").unwrap();
let mut ruler = EntityRuler::new(&nlp);
ruler
    .add_patterns(
        &nlp,
        &[
            RulerPattern::phrase("ORG", "Apple Inc."),
            RulerPattern::tokens("GPE", json!([{"LOWER": "san"}, {"LOWER": "francisco"}])).with_id("sf"),
        ],
    )
    .unwrap();
nlp.add_pipe("entity_ruler", ruler);
nlp.add_pipe("sentencizer", Sentencizer::new());
nlp.add_pipe("lemmatizer", Lemmatizer::english(LemmatizerMode::Lookup));

let doc = nlp.call("Apple Inc. opened offices in San Francisco. They were hiring.").unwrap();
let ents: Vec<(&str, &str)> = doc.ents().iter().map(|e| (e.text(), e.label_())).collect();
assert_eq!(ents, [("Apple Inc.", "ORG"), ("San Francisco", "GPE")]);
assert_eq!(doc.sents().len(), 2);
assert_eq!(doc.token(1).lemma_(), "Inc."); // lookup lemmas from spacy-lookups-data
assert_eq!(doc.token(9).lemma_(), "be");
```

### Trained pipelines

Download an official pipeline (MIT-licensed weights) and point `spacy::load` at it:

```sh
tools/fetch_model.sh en_core_web_sm 3.8.0 models   # or: python -m spacy download en_core_web_sm
```

```rust,no_run
let nlp = spacy::load("models/en_core_web_sm").unwrap();
let doc = nlp.call("Apple is looking at buying U.K. startup for $1 billion").unwrap();
for t in doc.iter() {
    println!("{:<8} {:<6} {:<5} {:<8} {:<6} {}", t.text(), t.tag_(), t.pos_(), t.dep_(), t.head().text(), t.lemma_());
}
for e in doc.ents() {
    println!("{} {}", e.text(), e.label_()); // Apple ORG, U.K. GPE, $1 billion MONEY
}
for np in doc.noun_chunks().unwrap() {
    println!("{np}");
}
// Batch processing; batches run in parallel with the default `rayon` feature.
let docs = nlp.pipe(&["First text.", "Second text."], 64).unwrap();
```

`spacy::load` accepts the pipeline data directory (with `config.cfg`) or the installed package
directory (e.g. `site-packages/en_core_web_sm`).

### DocBin (`.spacy` files)

```rust
let nlp = spacy::blank("en").unwrap();
let mut bin = spacy::DocBin::default();
bin.add(&nlp.make_doc("Serialized with spaCy's DocBin format."));
let bytes = bin.to_bytes(); // readable by `spacy.tokens.DocBin().from_bytes(...)`
let docs = spacy::DocBin::from_bytes(&bytes).unwrap().get_docs(nlp.vocab()).unwrap();
assert_eq!(docs[0].text(), "Serialized with spaCy's DocBin format.");
```

`spacy::displacy::render_ents(&doc)` renders entities as displaCy-style HTML.

## Parity with spaCy

| Feature | spaCy | spacy-rs | Parity tests |
|---|---|---|---|
| `StringStore` 64-bit hashes (MurmurHash64A, symbols) | ✓ | ✓ | exact IDs |
| `Vocab` / `Lexeme` attributes (`lower`, `norm`, `shape`, `prefix`, `suffix`, `is_*`, `like_*`, `is_stop`) | ✓ | ✓ | 25 attributes × all fixture tokens |
| `Doc`, `Token`, `Span` (offsets, whitespace, heads, children, subtree, ents, sents, `char_span`, `to_array`, `to_json`) | ✓ | ✓ | via component tests |
| Tokenizer: prefixes/suffixes/infixes, exceptions, special cases with affixes, `url_match`, `token_match` | ✓ | ✓ | exact on ~40k tokens |
| Languages with bundled tokenizer data | 70+ | en, de, fr, es, it, pt, nl, xx | each tested |
| `Matcher` (`?`, `*`, `+`, `!`, `{n,m}`; `IN`, `NOT_IN`, `IS_SUBSET`, `IS_SUPERSET`, `INTERSECTS`, `REGEX`, `FUZZY*`, comparisons, `LENGTH`; greedy `FIRST`/`LONGEST`) | ✓ | ✓ | exact matches **and match order** |
| `Matcher` extension attributes (`_`) and `with_alignments` | ✓ | ✗ | – |
| `PhraseMatcher` (any attribute) | ✓ | ✓ | exact |
| `DependencyMatcher` | ✓ | ✗ | – |
| `EntityRuler`, `SpanRuler` | ✓ | ✓ | exact |
| `AttributeRuler` | ✓ | ✓ | exact (via `en_core_web_sm`) |
| `Sentencizer` | ✓ | ✓ | exact |
| `Lemmatizer` lookup mode (English tables bundled) | ✓ | ✓ | exact |
| `Lemmatizer` rule mode (English) | ✓ | ✓ | exact (via `en_core_web_sm`) |
| `DocBin` read/write | ✓ | ✓ | spaCy ⇄ spacy-rs round trip |
| `doc.noun_chunks` (English) | ✓ | ✓ | exact |
| displaCy | ✓ | entities only | – |
| **Trained pipelines** (`spacy.load`) | ✓ | CNN pipelines (`*_sm`, `*_md`, `*_lg` architecture) | see below |
| ↳ `tok2vec` (`Tok2Vec.v2`, `MultiHashEmbed.v2`, `MaxoutWindowEncoder.v2`) | ✓ | ✓ | 100% (downstream) |
| ↳ `tagger` (`Tagger.v2`) | ✓ | ✓ | 100.000% tags |
| ↳ `parser` (`TransitionBasedParser.v2`, arc-eager, greedy, pseudo-projective) | ✓ | ✓ | 100.000% heads/labels/sentences |
| ↳ `ner` (`TransitionBasedParser.v2`, BILUO) | ✓ | ✓ | 100.000% entities |
| ↳ `attribute_ruler`, `lemmatizer` | ✓ | ✓ | 100.000% POS/morph/lemmas |
| ↳ `senter`, `morphologizer`, `textcat`, `entity_linker`, static vectors (`*_md`/`*_lg`), transformers (`*_trf`), beam search | ✓ | ✗ (load with `spacy::load_with(path, &["senter"])` to skip) | – |
| Training | ✓ | ✗ | – |

Numbers are from `tests/model_parity.rs` (`en_core_web_sm` 3.8.0, 1,020 docs, 10,280 tokens):

| Attribute | tag | pos | morph | lemma | dep | head | sentence starts | ent IOB/type | noun chunks |
|---|---|---|---|---|---|---|---|---|---|
| Agreement with spaCy | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100% of docs |

Floating-point sums use a different order than spaCy's BLAS, so bit-for-bit identical scores
are not guaranteed; on the test set this never changes a decision. The test tolerates up to
0.5% flips to stay robust across platforms.

## Performance

`cargo run --release --example bench -- models/en_core_web_sm` on an Apple-silicon laptop,
news paragraphs (spaCy 3.8.18 on Python 3.12 for comparison):

| | spaCy | spacy-rs |
|---|---|---|
| Tokenizer, 1 thread | 2.5 M tokens/s | 9.6 M tokens/s |
| Tokenizer, all cores | – | 15 M tokens/s |
| `en_core_web_sm`, `nlp(text)` one doc at a time | 13 k tokens/s | 29 k tokens/s |
| `en_core_web_sm`, `nlp.pipe` | 26 k tokens/s | 154 k tokens/s (rayon) |

## How it stays exact

* Language data (prefix/suffix/infix regexes, exceptions, URL pattern, stop words, number
  words) is **exported from spaCy itself** by `tools/export_lang.py` and embedded as JSON.
* Python `re` patterns are translated to `fancy-regex` (character-class quirks, `\w`/`\s`
  with Python's exact Unicode definitions, `\Z`, `(?P=name)`, `{,n}` ...). For speed, the
  lookaround-free alternatives run on a DFA and only the few lookaround alternatives use the
  backtracking VM — this keeps `re.search` leftmost-first semantics exactly.
* Python's `str.isalpha/isdigit/isupper/islower/istitle/isspace/lower` are reproduced with
  tables generated from Python 3.12 (`tools/gen_unicode.py`).
* The tokenizer, matcher state machine, phrase matcher, rulers, lemmatizer, transition
  systems, `set_children_from_heads` and `deprojectivize` are line-by-line ports of the Cython
  sources; thinc layers (`HashEmbed` with MurmurHash3, `Maxout`, `LayerNorm` with numpy's
  pairwise summation, `expand_window` with padding) are reimplemented in plain Rust.
* Golden fixtures are produced by `tools/make_fixtures.py`, `make_rule_fixtures.py`,
  `make_model_fixtures.py` and `make_docbin_fixture.py` with real spaCy.

## Status and roadmap

v0.1 covers the CNN English pipeline end to end. Next: more bundled languages, `senter` and
`morphologizer`, static vectors (`*_md`/`*_lg`), `DependencyMatcher`, `Retokenizer`
(merge/split), Matcher extension attributes, and an optional downloader.

## Credits and license

spacy-rs is a port of [spaCy](https://github.com/explosion/spaCy) by
[Explosion](https://explosion.ai) (Matthew Honnibal, Ines Montani and contributors), and of
the relevant parts of [thinc](https://github.com/explosion/thinc). The bundled language data
is derived from spaCy (MIT); the English lemmatizer tables come from
[spacy-lookups-data](https://github.com/explosion/spacy-lookups-data) (MIT), which derives
them from [WordNet 3.0](https://wordnet.princeton.edu/) (WordNet license, see
`data/lookups/LICENSE-WordNet.txt`). Trained pipelines are not included; the official ones
are published by Explosion under the MIT license.

spacy-rs is released under the MIT License (see `LICENSE`, which also reproduces spaCy's
and spacy-lookups-data's MIT notices).
