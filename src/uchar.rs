//! Character predicates with exactly the semantics of Python 3.12's `str`
//! methods (`isalpha`, `isdigit`, `isupper`, ...), which spaCy's lexical
//! attributes are defined in terms of.

use crate::unicode_tables as t;

fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    table
        .binary_search_by(|&(a, b)| {
            if c < a {
                std::cmp::Ordering::Greater
            } else if c > b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Python `ch.isalpha()` (general category L*).
pub fn is_alpha_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_table(t::ALPHA, c)
}
/// Python `ch.isdigit()`.
pub fn is_digit_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_table(t::DIGIT, c)
}
/// Python `ch.isspace()`.
pub fn is_space_char(c: char) -> bool {
    if c.is_ascii() {
        return matches!(
            c,
            ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f'
        );
    }
    in_table(t::SPACE, c)
}
/// Unicode `Uppercase` property as used by Python.
pub fn is_upper_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_uppercase();
    }
    in_table(t::UPPER, c)
}
/// Unicode `Lowercase` property as used by Python.
pub fn is_lower_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_lowercase();
    }
    in_table(t::LOWER, c)
}
/// Titlecase letter (category Lt).
pub fn is_title_char(c: char) -> bool {
    !c.is_ascii() && in_table(t::TITLE, c)
}
/// Unicode punctuation (category P*).
pub fn is_punct_char(c: char) -> bool {
    in_table(t::PUNCT, c)
}
/// Currency symbol (category Sc).
pub fn is_currency_char(c: char) -> bool {
    in_table(t::CURRENCY, c)
}

/// Python `s.isalpha()`.
pub fn py_isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_alpha_char)
}
/// Python `s.isdigit()`.
pub fn py_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_digit_char)
}
/// Python `s.isspace()`.
pub fn py_isspace(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_space_char)
}
/// Python `s.islower()`.
pub fn py_islower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if is_upper_char(c) || is_title_char(c) {
            return false;
        }
        if !cased && is_lower_char(c) {
            cased = true;
        }
    }
    cased
}
/// Python `s.isupper()`.
pub fn py_isupper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if is_lower_char(c) || is_title_char(c) {
            return false;
        }
        if !cased && is_upper_char(c) {
            cased = true;
        }
    }
    cased
}
/// Python `s.istitle()`.
pub fn py_istitle(s: &str) -> bool {
    let mut cased = false;
    let mut prev_cased = false;
    for c in s.chars() {
        if is_upper_char(c) || is_title_char(c) {
            if prev_cased {
                return false;
            }
            prev_cased = true;
            cased = true;
        } else if is_lower_char(c) {
            if !prev_cased {
                return false;
            }
            prev_cased = true;
            cased = true;
        } else {
            prev_cased = false;
        }
    }
    cased
}

fn lower_char_into(c: char, out: &mut String) {
    if c.is_ascii() {
        out.push(c.to_ascii_lowercase());
        return;
    }
    let cp = c as u32;
    match t::LOWER_MAP.binary_search_by_key(&cp, |&(k, _)| k) {
        Ok(i) => out.push_str(t::LOWER_MAP[i].1),
        Err(_) => out.push(c),
    }
}

fn is_cased(c: char) -> bool {
    is_upper_char(c) || is_lower_char(c) || is_title_char(c)
}

fn is_case_ignorable(c: char) -> bool {
    in_table(t::CASE_IGNORABLE, c)
}

/// Python `s.lower()`, including the final-sigma rule.
pub fn py_lower(s: &str) -> String {
    if s.is_ascii() {
        return s.to_ascii_lowercase();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == '\u{3A3}' {
            // Final sigma: preceded by a cased letter and not followed by one.
            let before = chars[..i]
                .iter()
                .rev()
                .find(|&&x| !is_case_ignorable(x))
                .is_some_and(|&x| is_cased(x));
            let after = chars[i + 1..]
                .iter()
                .find(|&&x| !is_case_ignorable(x))
                .is_some_and(|&x| is_cased(x));
            out.push(if before && !after {
                '\u{3C2}'
            } else {
                '\u{3C3}'
            });
        } else {
            lower_char_into(c, &mut out);
        }
    }
    out
}

/// Number of Unicode scalar values (Python `len`).
pub fn char_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// Byte offset of the `n`-th character (or `s.len()` if out of range).
pub fn byte_offset(s: &str, n: usize) -> usize {
    if s.is_ascii() {
        return n.min(s.len());
    }
    s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_semantics() {
        assert!(py_isalpha("Café"));
        assert!(!py_isalpha("a1"));
        assert!(py_isdigit("²3"));
        assert!(py_istitle("Hello World"));
        assert!(!py_istitle("HeLLo"));
        assert!(py_isupper("ABC1"));
        assert!(py_islower("abc!"));
        assert_eq!(py_lower("ΟΔΟΣ"), "οδο\u{3C2}");
        assert_eq!(py_lower("ΟΔΟΣ ΣΑ"), "οδο\u{3C2} \u{3C3}α");
        assert_eq!(py_lower("İ"), "i\u{307}");
        assert!(py_isspace("\u{1c}"));
    }
}
