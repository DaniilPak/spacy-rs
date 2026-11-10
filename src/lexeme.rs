//! Lexemes (context-independent word types) and spaCy's lexical attribute
//! functions (`like_num`, `like_url`, `word_shape`, ...).

use crate::attrs::*;
use crate::uchar::*;
use std::collections::HashSet;
use std::sync::LazyLock;

/// A word type in the vocabulary, mirroring spaCy's `LexemeC`.
#[derive(Clone, Debug, Default)]
#[allow(missing_docs)]
pub struct LexemeC {
    pub orth: u64,
    /// Length in Unicode characters.
    pub length: usize,
    pub lower: u64,
    pub norm: u64,
    pub shape: u64,
    pub prefix: u64,
    pub suffix: u64,
    pub lang: u64,
    /// Boolean flags; bit `n` is the flag with attribute ID `n` (e.g. `IS_ALPHA`).
    pub flags: u64,
    pub id: u64,
    pub prob: f32,
    pub cluster: u64,
    pub sentiment: f32,
}

impl LexemeC {
    /// Check a boolean flag by attribute ID.
    #[inline]
    pub fn check_flag(&self, flag: u64) -> bool {
        flag < 64 && self.flags & (1u64 << flag) != 0
    }
    /// Set a boolean flag by attribute ID.
    #[inline]
    pub fn set_flag(&mut self, flag: u64, value: bool) {
        if flag >= 64 {
            return;
        }
        if value {
            self.flags |= 1u64 << flag;
        } else {
            self.flags &= !(1u64 << flag);
        }
    }
    /// Numeric value of an attribute, as spaCy's `Lexeme.get_struct_attr`.
    pub fn get_attr(&self, attr: u64) -> u64 {
        match attr {
            a if a < 64 => self.check_flag(a) as u64,
            ID => self.id,
            ORTH => self.orth,
            LOWER => self.lower,
            NORM => self.norm,
            SHAPE => self.shape,
            PREFIX => self.prefix,
            SUFFIX => self.suffix,
            LENGTH => self.length as u64,
            LANG => self.lang,
            _ => 0,
        }
    }
}

/// spaCy's `word_shape`.
pub fn word_shape(text: &str) -> String {
    if char_len(text) >= 100 {
        return "LONG".to_string();
    }
    let mut shape = String::new();
    let mut last: Option<char> = None;
    let mut seq = 0;
    for c in text.chars() {
        let sc = if is_alpha_char(c) {
            if is_upper_char(c) { 'X' } else { 'x' }
        } else if is_digit_char(c) {
            'd'
        } else {
            c
        };
        if Some(sc) == last {
            seq += 1;
        } else {
            seq = 0;
            last = Some(sc);
        }
        if seq < 4 {
            shape.push(sc);
        }
    }
    shape
}

/// `text[:1]` (first character).
pub fn prefix(text: &str) -> &str {
    match text.char_indices().nth(1) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

/// `text[-3:]` (last three characters).
pub fn suffix(text: &str) -> &str {
    let n = char_len(text);
    if n <= 3 {
        return text;
    }
    &text[byte_offset(text, n - 3)..]
}

/// All characters are Unicode punctuation (`P*`).
pub fn is_punct(text: &str) -> bool {
    text.chars().all(is_punct_char)
}
/// All characters are ASCII.
pub fn is_ascii(text: &str) -> bool {
    text.is_ascii()
}
/// One of `()[]{}<>`.
pub fn is_bracket(text: &str) -> bool {
    matches!(text, "(" | ")" | "[" | "]" | "{" | "}" | "<" | ">")
}
/// A quotation mark.
pub fn is_quote(text: &str) -> bool {
    matches!(
        text,
        "\"" | "'"
            | "`"
            | "«"
            | "»"
            | "‘"
            | "’"
            | "‚"
            | "‛"
            | "“"
            | "”"
            | "„"
            | "‟"
            | "‹"
            | "›"
            | "❮"
            | "❯"
            | "''"
            | "``"
    )
}
/// Left punctuation.
pub fn is_left_punct(text: &str) -> bool {
    matches!(
        text,
        "(" | "["
            | "{"
            | "<"
            | "\""
            | "'"
            | "«"
            | "‘"
            | "‚"
            | "‛"
            | "“"
            | "„"
            | "‟"
            | "‹"
            | "❮"
            | "``"
    )
}
/// Right punctuation.
pub fn is_right_punct(text: &str) -> bool {
    matches!(
        text,
        ")" | "]" | "}" | ">" | "\"" | "'" | "»" | "’" | "”" | "›" | "❯" | "''"
    )
}
/// All characters are currency symbols (`Sc`).
pub fn is_currency(text: &str) -> bool {
    text.chars().all(is_currency_char)
}

static EMAIL_RE: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    crate::regex_compat::compile(r"^([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
        .expect("email regex")
});

/// spaCy's `like_email`.
pub fn like_email(text: &str) -> bool {
    EMAIL_RE.is_match(text).unwrap_or(false)
}

const TLDS: &str = "com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|\
name|pro|tel|travel|xyz|icu|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|\
ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|\
cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|\
ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|\
gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|\
je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|\
lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|\
na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|\
pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|\
ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|\
ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw";

static TLD_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| TLDS.split('|').collect());

/// spaCy's base `URL_MATCH` regex (`spacy.lang.tokenizer_exceptions`).
pub(crate) static URL_MATCH: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    crate::regex_compat::compile(crate::lang::base_url_pattern()).expect("url regex")
});

/// spaCy's `like_url`.
pub fn like_url(text: &str) -> bool {
    if text.starts_with("http://") || text.starts_with("https://") {
        return true;
    }
    if text.starts_with("www.") && char_len(text) >= 5 {
        return true;
    }
    if text.is_empty() {
        // Python would raise IndexError; spaCy never calls this on "".
        return false;
    }
    if text.starts_with('.') || text.ends_with('.') {
        return false;
    }
    if text.contains('@') {
        return false;
    }
    if !text.contains('.') {
        return false;
    }
    let tld_full = text.rsplit_once('.').map(|(_, t)| t).unwrap_or("");
    let tld = tld_full.split_once(':').map(|(a, _)| a).unwrap_or(tld_full);
    if tld.ends_with('/') {
        return true;
    }
    if py_isalpha(tld) && TLD_SET.contains(tld) {
        return true;
    }
    URL_MATCH.is_match(text).unwrap_or(false)
}

/// Configuration of the language-specific `like_num` function.
#[derive(Clone, Debug, Default)]
pub struct LikeNum {
    /// Strip a leading `+`, `-`, `±` or `~`.
    pub strip_sign: bool,
    /// Characters removed before the digit test (usually `,` and `.`).
    pub remove: Vec<char>,
    /// Cardinal number words (compared lowercased).
    pub num_words: HashSet<String>,
    /// Ordinal number words (compared lowercased).
    pub ordinal_words: HashSet<String>,
    /// Accept `<digits>st|nd|rd|th` (English).
    pub ordinal_suffixes: bool,
}

impl LikeNum {
    /// The language-independent default.
    pub fn base() -> Self {
        LikeNum {
            strip_sign: true,
            remove: vec![',', '.'],
            ..Default::default()
        }
    }

    /// Evaluate `like_num` on a string.
    pub fn check(&self, text: &str) -> bool {
        let mut t = text;
        if self.strip_sign {
            for p in ["+", "-", "±", "~"] {
                if let Some(rest) = t.strip_prefix(p) {
                    t = rest;
                    break;
                }
            }
        }
        let t: String = t.chars().filter(|c| !self.remove.contains(c)).collect();
        if py_isdigit(&t) {
            return true;
        }
        if t.matches('/').count() == 1 {
            let (num, denom) = t.split_once('/').expect("one slash");
            if py_isdigit(num) && py_isdigit(denom) {
                return true;
            }
        }
        if self.num_words.is_empty() && self.ordinal_words.is_empty() && !self.ordinal_suffixes {
            return false;
        }
        let low = py_lower(&t);
        if self.num_words.contains(&low) || self.ordinal_words.contains(&low) {
            return true;
        }
        if self.ordinal_suffixes {
            for suf in ["st", "nd", "rd", "th"] {
                if let Some(stem) = low.strip_suffix(suf)
                    && py_isdigit(stem)
                {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shapes() {
        assert_eq!(word_shape("Apple"), "Xxxxx");
        assert_eq!(word_shape("C3PO"), "XdXX");
        assert_eq!(word_shape("aaaaaaa"), "xxxx");
        assert_eq!(suffix("hello"), "llo");
        assert_eq!(prefix("über"), "ü");
    }
    #[test]
    fn urls() {
        assert!(like_url("www.google.com"));
        assert!(like_url("google.com"));
        assert!(!like_url("hello"));
        assert!(like_email("a.b@c.org"));
    }
}
