//! Morphological feature strings (UD FEATS), normalized like spaCy's `Morphology`.

/// spaCy's `Morphology.EMPTY_MORPH`.
pub const EMPTY_MORPH: &str = "_";

/// Parse `"Case=Nom|Number=Sing"` into sorted `(field, values)` pairs.
pub fn feats_to_pairs(feats: &str) -> Vec<(String, String)> {
    if feats.is_empty() || feats == EMPTY_MORPH {
        return Vec::new();
    }
    let mut out: Vec<(String, String)> = Vec::new();
    for feat in feats.split('|') {
        let (field, values) = feat.split_once('=').unwrap_or((feat, ""));
        let mut vs: Vec<&str> = values.split(',').collect();
        vs.sort_unstable();
        let v = vs.join(",");
        if let Some(e) = out.iter_mut().find(|(f, _)| f == field) {
            e.1 = v;
        } else {
            out.push((field.to_string(), v));
        }
    }
    out
}

/// Normalize a FEATS string (sorted fields and values; empty → `"_"`).
pub fn normalize(feats: &str) -> String {
    let mut items: Vec<String> = feats_to_pairs(feats)
        .into_iter()
        .map(|(f, v)| format!("{f}={v}"))
        .collect();
    items.sort();
    if items.is_empty() { EMPTY_MORPH.to_string() } else { items.join("|") }
}

/// Individual `Field=Value` features (multi-valued fields are split).
pub fn features(feats: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (f, v) in feats_to_pairs(feats) {
        for val in v.split(',') {
            out.push(format!("{f}={val}"));
        }
    }
    out
}

/// Look up the value(s) of a field, e.g. `get(feats, "Number")`.
pub fn get(feats: &str, field: &str) -> Vec<String> {
    feats_to_pairs(feats)
        .into_iter()
        .filter(|(f, _)| f == field)
        .flat_map(|(_, v)| v.split(',').map(str::to_string).collect::<Vec<_>>())
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalizes() {
        assert_eq!(super::normalize("Number=Sing|Case=Nom,Acc"), "Case=Acc,Nom|Number=Sing");
        assert_eq!(super::normalize(""), "_");
        assert_eq!(super::features("PronType=Prs|Case=Nom,Acc"), vec!["Case=Acc", "Case=Nom", "PronType=Prs"]);
    }
}
