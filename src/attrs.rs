//! Attribute IDs, identical to spaCy's `spacy.attrs` / `spacy.symbols`.

#![allow(missing_docs)]

pub const NULL_ATTR: u64 = 0;
pub const IS_ALPHA: u64 = 1;
pub const IS_ASCII: u64 = 2;
pub const IS_DIGIT: u64 = 3;
pub const IS_LOWER: u64 = 4;
pub const IS_PUNCT: u64 = 5;
pub const IS_SPACE: u64 = 6;
pub const IS_TITLE: u64 = 7;
pub const IS_UPPER: u64 = 8;
pub const LIKE_URL: u64 = 9;
pub const LIKE_NUM: u64 = 10;
pub const LIKE_EMAIL: u64 = 11;
pub const IS_STOP: u64 = 12;
pub const IS_OOV_DEPRECATED: u64 = 13;
pub const IS_BRACKET: u64 = 14;
pub const IS_QUOTE: u64 = 15;
pub const IS_LEFT_PUNCT: u64 = 16;
pub const IS_RIGHT_PUNCT: u64 = 17;
pub const IS_CURRENCY: u64 = 18;
pub const ID: u64 = 64;
pub const ORTH: u64 = 65;
pub const LOWER: u64 = 66;
pub const NORM: u64 = 67;
pub const SHAPE: u64 = 68;
pub const PREFIX: u64 = 69;
pub const SUFFIX: u64 = 70;
pub const LENGTH: u64 = 71;
pub const LEMMA: u64 = 73;
pub const POS: u64 = 74;
pub const TAG: u64 = 75;
pub const DEP: u64 = 76;
pub const ENT_IOB: u64 = 77;
pub const ENT_TYPE: u64 = 78;
pub const HEAD: u64 = 79;
pub const SENT_START: u64 = 80;
pub const SPACY: u64 = 81;
pub const LANG: u64 = 83;
pub const ENT_KB_ID: u64 = 452;
pub const MORPH: u64 = 453;
pub const ENT_ID: u64 = 454;
pub const IDX: u64 = 455;

/// Resolve an attribute name such as `"LOWER"`, `"is_alpha"` or `"TEXT"`.
pub fn attr_id(name: &str) -> Option<u64> {
    let up = name.to_ascii_uppercase();
    Some(match up.as_str() {
        "TEXT" | "ORTH" => ORTH,
        "IS_ALPHA" => IS_ALPHA,
        "IS_ASCII" => IS_ASCII,
        "IS_DIGIT" => IS_DIGIT,
        "IS_LOWER" => IS_LOWER,
        "IS_PUNCT" => IS_PUNCT,
        "IS_SPACE" => IS_SPACE,
        "IS_TITLE" => IS_TITLE,
        "IS_UPPER" => IS_UPPER,
        "LIKE_URL" => LIKE_URL,
        "LIKE_NUM" => LIKE_NUM,
        "LIKE_EMAIL" => LIKE_EMAIL,
        "IS_STOP" => IS_STOP,
        "IS_BRACKET" => IS_BRACKET,
        "IS_QUOTE" => IS_QUOTE,
        "IS_LEFT_PUNCT" => IS_LEFT_PUNCT,
        "IS_RIGHT_PUNCT" => IS_RIGHT_PUNCT,
        "IS_CURRENCY" => IS_CURRENCY,
        "ID" => ID,
        "LOWER" => LOWER,
        "NORM" => NORM,
        "SHAPE" => SHAPE,
        "PREFIX" => PREFIX,
        "SUFFIX" => SUFFIX,
        "LENGTH" => LENGTH,
        "LEMMA" => LEMMA,
        "POS" => POS,
        "TAG" => TAG,
        "DEP" => DEP,
        "ENT_IOB" => ENT_IOB,
        "ENT_TYPE" => ENT_TYPE,
        "ENT_ID" => ENT_ID,
        "ENT_KB_ID" => ENT_KB_ID,
        "HEAD" => HEAD,
        "SENT_START" | "IS_SENT_START" => SENT_START,
        "SPACY" => SPACY,
        "LANG" => LANG,
        "MORPH" => MORPH,
        "IDX" => IDX,
        _ => return None,
    })
}

/// Universal part-of-speech tags with their spaCy symbol IDs.
pub mod pos {
    pub const NO_TAG: u64 = 0;
    pub const ADJ: u64 = 84;
    pub const ADP: u64 = 85;
    pub const ADV: u64 = 86;
    pub const AUX: u64 = 87;
    pub const CONJ: u64 = 88;
    pub const CCONJ: u64 = 89;
    pub const DET: u64 = 90;
    pub const INTJ: u64 = 91;
    pub const NOUN: u64 = 92;
    pub const NUM: u64 = 93;
    pub const PART: u64 = 94;
    pub const PRON: u64 = 95;
    pub const PROPN: u64 = 96;
    pub const PUNCT: u64 = 97;
    pub const SCONJ: u64 = 98;
    pub const SYM: u64 = 99;
    pub const VERB: u64 = 100;
    pub const X: u64 = 101;
    pub const EOL: u64 = 102;
    pub const SPACE: u64 = 103;

    /// Map a universal POS name to its ID.
    pub fn from_str(s: &str) -> Option<u64> {
        let id = crate::strings::symbol_id(s)?;
        (ADJ..=SPACE).contains(&id).then_some(id)
    }
}
