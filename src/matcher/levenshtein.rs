//! Levenshtein distance used by `FUZZY` matcher predicates.

/// Edit distance between two strings over Unicode scalar values, capped at `k + 1`.
pub fn levenshtein(a: &str, b: &str, k: Option<usize>) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if let Some(k) = k
        && a.len().abs_diff(b.len()) > k
    {
        return k + 1;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = (a[i - 1] != b[j - 1]) as usize;
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[b.len()];
    match k {
        Some(k) if d > k => k + 1,
        _ => d,
    }
}

/// spaCy's default `levenshtein_compare`: with `fuzzy < 0`, allow
/// `max(2, round(0.3 * len(pattern)))` edits.
pub fn levenshtein_compare(input: &str, pattern: &str, fuzzy: i64) -> bool {
    let max_edits = if fuzzy >= 0 {
        fuzzy as usize
    } else {
        let n = pattern.chars().count() as f64;
        ((0.3 * n).round_ties_even() as usize).max(2)
    };
    levenshtein(input, pattern, Some(max_edits)) <= max_edits
}
