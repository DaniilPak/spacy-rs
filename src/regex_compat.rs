//! Translation of Python `re` patterns (as used throughout spaCy's language
//! data) into the syntax understood by [`fancy_regex`].
//!
//! The differences handled here:
//! * inside character classes, `[`, `&`, `~` and `-` are literal in Python but
//!   have special meaning (nesting / set operations) in Rust;
//! * `\w` and `\s` follow Python's definitions exactly;
//! * `\Z`, `(?P=name)`, `{,n}`, octal/`\0` escapes, lone braces, and `(?u)`.

use crate::unicode_tables::WORD;

fn word_ranges() -> String {
    let mut s = String::new();
    for &(a, b) in WORD {
        if a == b {
            s.push_str(&format!("\\x{{{a:X}}}"));
        } else {
            s.push_str(&format!("\\x{{{a:X}}}-\\x{{{b:X}}}"));
        }
    }
    s
}

const PY_SPACE: &str = "\\s\\x1C-\\x1F";

#[derive(Clone, Debug)]
enum ClassItem {
    Lit(char),
    Raw(String), // already-translated class fragment (\d, \w ranges, ...)
}

fn emit_lit(c: char, out: &mut String) {
    if c.is_ascii_punctuation() {
        if c == '<' || c == '>' {
            out.push(c);
        } else {
            out.push('\\');
            out.push(c);
        }
    } else if c.is_ascii_control() || c == ' ' {
        out.push_str(&format!("\\x{{{:X}}}", c as u32));
    } else {
        out.push(c);
    }
}

fn emit_class_lit(c: char, out: &mut String) {
    if c.is_ascii_punctuation() {
        out.push('\\');
        out.push(c);
    } else if c.is_ascii_control() || c == ' ' {
        out.push_str(&format!("\\x{{{:X}}}", c as u32));
    } else {
        out.push(c);
    }
}

struct P<'a> {
    c: Vec<char>,
    i: usize,
    word: &'a str,
}

impl P<'_> {
    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }
    fn peek_at(&self, k: usize) -> Option<char> {
        self.c.get(self.i + k).copied()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        self.i += 1;
        c
    }
    fn hex(&mut self, n: usize) -> Option<char> {
        let s: String = self.c.get(self.i..self.i + n)?.iter().collect();
        let v = u32::from_str_radix(&s, 16).ok()?;
        self.i += n;
        char::from_u32(v)
    }
    fn octal(&mut self, first: char) -> char {
        let mut v = first.to_digit(8).unwrap_or(0);
        for _ in 0..2 {
            match self.peek().and_then(|c| c.to_digit(8)) {
                Some(d) => {
                    v = v * 8 + d;
                    self.i += 1;
                }
                None => break,
            }
        }
        char::from_u32(v).unwrap_or('\0')
    }

    /// Parse an escape (the backslash already consumed) inside a class.
    fn class_escape(&mut self) -> ClassItem {
        let c = self.next().unwrap_or('\\');
        match c {
            'd' => ClassItem::Raw("\\d".into()),
            'D' => ClassItem::Raw("\\D".into()),
            's' => ClassItem::Raw(PY_SPACE.into()),
            'S' => ClassItem::Raw(format!("[^{PY_SPACE}]")),
            'w' => ClassItem::Raw(self.word.to_string()),
            'W' => ClassItem::Raw(format!("[^{}]", self.word)),
            'n' => ClassItem::Lit('\n'),
            't' => ClassItem::Lit('\t'),
            'r' => ClassItem::Lit('\r'),
            'f' => ClassItem::Lit('\x0c'),
            'v' => ClassItem::Lit('\x0b'),
            'a' => ClassItem::Lit('\x07'),
            'b' => ClassItem::Lit('\x08'),
            'x' => ClassItem::Lit(self.hex(2).unwrap_or('x')),
            'u' => ClassItem::Lit(self.hex(4).unwrap_or('u')),
            'U' => ClassItem::Lit(self.hex(8).unwrap_or('U')),
            '0'..='7' => ClassItem::Lit(self.octal(c)),
            other => ClassItem::Lit(other),
        }
    }

    fn class(&mut self, out: &mut String) {
        // '[' already consumed
        out.push('[');
        if self.peek() == Some('^') {
            self.i += 1;
            out.push('^');
        }
        let mut first = true;
        loop {
            let Some(this) = self.next() else { break };
            if this == ']' && !first {
                break;
            }
            first = false;
            let item = if this == '\\' {
                self.class_escape()
            } else {
                ClassItem::Lit(this)
            };
            if self.peek() == Some('-') {
                // potential range
                self.i += 1;
                match self.peek() {
                    None | Some(']') => {
                        self.emit_item(&item, out);
                        out.push_str("\\-");
                    }
                    Some(_) => {
                        let that = self.next().unwrap_or('-');
                        let hi = if that == '\\' {
                            self.class_escape()
                        } else {
                            ClassItem::Lit(that)
                        };
                        match (&item, &hi) {
                            (ClassItem::Lit(a), ClassItem::Lit(b)) => {
                                emit_class_lit(*a, out);
                                out.push('-');
                                emit_class_lit(*b, out);
                            }
                            _ => {
                                self.emit_item(&item, out);
                                out.push_str("\\-");
                                self.emit_item(&hi, out);
                            }
                        }
                    }
                }
            } else {
                self.emit_item(&item, out);
            }
        }
        out.push(']');
    }

    fn emit_item(&self, item: &ClassItem, out: &mut String) {
        match item {
            ClassItem::Lit(c) => emit_class_lit(*c, out),
            ClassItem::Raw(s) => out.push_str(s),
        }
    }

    /// Try to parse a `{m,n}` quantifier at the current position ('{' consumed).
    fn quantifier(&mut self) -> Option<String> {
        let start = self.i;
        let mut lo = String::new();
        while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
            lo.push(c);
            self.i += 1;
        }
        let mut hi = None;
        if self.peek() == Some(',') {
            self.i += 1;
            let mut h = String::new();
            while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
                h.push(c);
                self.i += 1;
            }
            hi = Some(h);
        }
        if self.peek() == Some('}') && !(lo.is_empty() && hi.as_deref().is_none_or(str::is_empty)) {
            self.i += 1;
            let lo = if lo.is_empty() { "0".to_string() } else { lo };
            return Some(match hi {
                None => format!("{{{lo}}}"),
                Some(h) if h.is_empty() => format!("{{{lo},}}"),
                Some(h) => format!("{{{lo},{h}}}"),
            });
        }
        self.i = start;
        None
    }

    fn run(&mut self) -> String {
        let mut out = String::with_capacity(self.c.len() + 16);
        while let Some(c) = self.next() {
            match c {
                '\\' => {
                    let Some(e) = self.next() else {
                        out.push_str("\\\\");
                        break;
                    };
                    match e {
                        'd' | 'D' | 'b' | 'B' | 'A' => {
                            out.push('\\');
                            out.push(e);
                        }
                        'Z' => out.push_str("\\z"),
                        's' => out.push_str(&format!("[{PY_SPACE}]")),
                        'S' => out.push_str(&format!("[^{PY_SPACE}]")),
                        'w' => out.push_str(&format!("[{}]", self.word)),
                        'W' => out.push_str(&format!("[^{}]", self.word)),
                        'n' => out.push_str("\\n"),
                        't' => out.push_str("\\t"),
                        'r' => out.push_str("\\r"),
                        'f' => out.push_str("\\x0C"),
                        'v' => out.push_str("\\x0B"),
                        'a' => out.push_str("\\x07"),
                        'x' => {
                            let ch = self.hex(2).unwrap_or('x');
                            emit_lit(ch, &mut out)
                        }
                        'u' => {
                            let ch = self.hex(4).unwrap_or('u');
                            emit_lit(ch, &mut out)
                        }
                        'U' => {
                            let ch = self.hex(8).unwrap_or('U');
                            emit_lit(ch, &mut out)
                        }
                        '0' => {
                            let ch = self.octal('0');
                            emit_lit(ch, &mut out)
                        }
                        '1'..='9' => {
                            out.push('\\');
                            out.push(e);
                            while let Some(d) = self.peek().filter(|d| d.is_ascii_digit()) {
                                out.push(d);
                                self.i += 1;
                            }
                        }
                        other => emit_lit(other, &mut out),
                    }
                }
                '[' => self.class(&mut out),
                '{' => match self.quantifier() {
                    Some(q) => out.push_str(&q),
                    None => out.push_str("\\{"),
                },
                '}' => out.push_str("\\}"),
                '(' if self.peek() == Some('?') => {
                    let rest: String = self.c[self.i..].iter().take(4).collect();
                    if rest.starts_with("?P=") {
                        self.i += 3;
                        let mut name = String::new();
                        while let Some(ch) = self.next() {
                            if ch == ')' {
                                break;
                            }
                            name.push(ch);
                        }
                        out.push_str(&format!("\\k<{name}>"));
                    } else if rest.starts_with("?#") {
                        while let Some(ch) = self.next() {
                            if ch == ')' {
                                break;
                            }
                        }
                    } else if self.peek_at(1).is_some_and(|ch| ch.is_ascii_alphabetic())
                        && self.peek_at(1) != Some('P')
                    {
                        // inline flags: (?aiLmsux) or (?flags:...)
                        self.i += 1;
                        let mut flags = String::new();
                        while let Some(ch) = self.peek().filter(|ch| ch.is_ascii_alphabetic() || *ch == '-') {
                            flags.push(ch);
                            self.i += 1;
                        }
                        let kept: String = flags.chars().filter(|f| !matches!(f, 'u' | 'a' | 'L')).collect();
                        let term = self.next();
                        if term == Some(')') {
                            if !kept.is_empty() && kept != "-" {
                                out.push_str(&format!("(?{kept})"));
                            }
                        } else if kept.is_empty() || kept == "-" {
                            out.push_str("(?:");
                        } else {
                            out.push_str(&format!("(?{kept}:"));
                        }
                    } else {
                        out.push('(');
                    }
                }
                other => out.push(other),
            }
        }
        out
    }
}

/// Translate a Python `re` pattern to `fancy_regex` syntax.
pub fn translate(pattern: &str) -> String {
    let word = word_ranges();
    let mut p = P {
        c: pattern.chars().collect(),
        i: 0,
        word: &word,
    };
    p.run()
}

/// Compile a Python `re` pattern.
pub fn compile(pattern: &str) -> Result<fancy_regex::Regex, crate::Error> {
    let t = translate(pattern);
    fancy_regex::RegexBuilder::new(&t)
        .backtrack_limit(10_000_000)
        .build()
        .map_err(|e| crate::Error::Regex(format!("{e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn translates_classes() {
        let r = compile(r"[a-zA-Z0-9-.]+").unwrap();
        assert!(r.is_match("a-b.c").unwrap());
        let r = compile(r"[[\]&~]").unwrap();
        assert!(r.is_match("[").unwrap() && r.is_match("~").unwrap() && r.is_match("&").unwrap());
        let r = compile(r"^x{,2}$").unwrap();
        assert!(r.is_match("xx").unwrap());
        let r = compile(r"(?u)^\w+$").unwrap();
        assert!(r.is_match("héllo_1").unwrap());
        assert!(!r.is_match("a\u{301}").unwrap());
        let r = compile(r"(?P<q>['\x22])x(?P=q)").unwrap();
        assert!(r.is_match("'x'").unwrap());
    }
}
