#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod attrs;
pub mod doc;
pub mod lang;
pub mod language;
pub mod lexeme;
pub mod matcher;
pub mod models;
pub mod morphology;
pub mod pipeline;
pub mod regex_compat;
pub mod strings;
pub mod syntax;
pub mod tokenizer;
pub mod uchar;
mod unicode_tables;
pub mod vocab;

pub use doc::{Alignment, Doc, Span, SpanC, Token, TokenC};
pub use language::{Language, Pipe, blank};
pub use models::{load, load_with};
pub use strings::{StringStore, hash_string};
pub use tokenizer::Tokenizer;
pub use vocab::Vocab;

/// Errors returned by spacy-rs.
#[derive(Debug)]
pub enum Error {
    /// A regex failed to compile.
    Regex(String),
    /// Invalid argument or data.
    Value(String),
    /// Language code without bundled data.
    UnknownLanguage(String),
    /// I/O failure.
    Io(std::io::Error),
    /// Malformed model or serialized data.
    Format(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Regex(e) => write!(f, "regex error: {e}"),
            Error::Value(e) => write!(f, "{e}"),
            Error::UnknownLanguage(l) => write!(f, "no bundled language data for {l:?}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(e) => write!(f, "format error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
