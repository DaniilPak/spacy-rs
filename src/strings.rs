//! String interning compatible with spaCy's `StringStore`.
//!
//! spaCy identifies every string by a 64-bit hash: `MurmurHash64A(utf8, seed=1)`,
//! except for a fixed set of "symbols" (attribute names, part-of-speech tags,
//! dependency labels, entity types, ...) which map to small integers. This module
//! reproduces both so that IDs are interchangeable with spaCy's.

use crate::fxhash::FxHashMap as HashMap;
use std::sync::{LazyLock, RwLock};

/// MurmurHash64A, as used by spaCy via the `murmurhash` package (`hash64`).
pub fn murmurhash64a(key: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;
    let mut h: u64 = seed ^ (key.len() as u64).wrapping_mul(M);
    let chunks = key.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        let mut k = u64::from_le_bytes(c.try_into().expect("8 bytes"));
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    if !rem.is_empty() {
        for (i, &b) in rem.iter().enumerate() {
            h ^= (b as u64) << (8 * i);
        }
        h = h.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// spaCy's `strings.hash_string`: MurmurHash64A of the UTF-8 bytes with seed 1.
///
/// Note that this ignores symbols; use [`get_string_id`] for the ID a
/// [`StringStore`] would assign.
pub fn hash_string(s: &str) -> u64 {
    murmurhash64a(s.as_bytes(), 1)
}

static SYMBOLS_JSON: &str = include_str!("../data/symbols.json");

struct Symbols {
    by_str: HashMap<&'static str, u64>,
    by_id: Vec<&'static str>,
}

static SYMBOLS: LazyLock<Symbols> = LazyLock::new(|| {
    let items: Vec<(String, u64)> = serde_json::from_str(SYMBOLS_JSON).expect("symbols.json");
    let mut by_str = HashMap::default();
    let mut by_id = vec![""; items.len()];
    for (s, id) in items {
        let s: &'static str = Box::leak(s.into_boxed_str());
        by_str.insert(s, id);
        by_id[id as usize] = s;
    }
    Symbols { by_str, by_id }
});

/// The symbol ID of `s` if it is one of spaCy's built-in symbols.
pub fn symbol_id(s: &str) -> Option<u64> {
    SYMBOLS.by_str.get(s).copied()
}

/// The string of a built-in symbol ID.
pub fn symbol_str(id: u64) -> Option<&'static str> {
    SYMBOLS.by_id.get(id as usize).copied()
}

/// The ID spaCy's `StringStore` assigns to `s` (symbol ID or hash).
pub fn get_string_id(s: &str) -> u64 {
    if s.is_empty() {
        return 0;
    }
    symbol_id(s).unwrap_or_else(|| hash_string(s))
}

/// A thread-safe, append-only string store.
///
/// Strings are never removed, so references returned by [`StringStore::get`]
/// stay valid for the lifetime of the store.
#[derive(Default)]
pub struct StringStore {
    map: [RwLock<HashMap<u64, Box<str>>>; crate::fxhash::SHARDS],
}

impl StringStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a string and return its ID (spaCy-compatible).
    pub fn add(&self, s: &str) -> u64 {
        if s.is_empty() {
            return 0;
        }
        if let Some(id) = symbol_id(s) {
            return id;
        }
        let h = hash_string(s);
        let shard = &self.map[crate::fxhash::shard_of(h)];
        {
            let r = shard.read().expect("string store lock");
            if r.contains_key(&h) {
                return h;
            }
        }
        shard
            .write()
            .expect("string store lock")
            .entry(h)
            .or_insert_with(|| s.into());
        h
    }

    /// Look up the string for an ID. Symbols always resolve.
    pub fn get(&self, id: u64) -> Option<&str> {
        if id == 0 {
            return Some("");
        }
        if let Some(s) = symbol_str(id).filter(|_| id < SYMBOLS.by_id.len() as u64) {
            return Some(s);
        }
        let r = self.map[crate::fxhash::shard_of(id)]
            .read()
            .expect("string store lock");
        let p = r.get(&id).map(|b| &**b as *const str)?;
        // SAFETY: entries are boxed, never removed or mutated, so the heap
        // allocation outlives `&self` even if the map rehashes.
        Some(unsafe { &*p })
    }

    /// Like [`StringStore::get`] but returns `""` for unknown IDs.
    pub fn get_or_empty(&self, id: u64) -> &str {
        self.get(id).unwrap_or("")
    }

    /// Whether the string is in the store (symbols are always present).
    pub fn contains(&self, s: &str) -> bool {
        s.is_empty() || symbol_id(s).is_some() || {
            let h = hash_string(s);
            self.map[crate::fxhash::shard_of(h)]
                .read()
                .expect("lock")
                .contains_key(&h)
        }
    }

    /// Number of non-symbol strings.
    pub fn len(&self) -> usize {
        self.map.iter().map(|m| m.read().expect("lock").len()).sum()
    }

    /// Whether the store has no non-symbol strings.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All non-symbol strings, in arbitrary order.
    pub fn strings(&self) -> Vec<String> {
        self.map
            .iter()
            .flat_map(|m| {
                m.read()
                    .expect("lock")
                    .values()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl std::fmt::Debug for StringStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StringStore({} strings)", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hashes_match_spacy() {
        assert_eq!(hash_string("apple"), 8566208034543834098);
        assert_eq!(hash_string(""), 14313749767032693980);
        assert_eq!(hash_string("café"), 32833993555699147);
        assert_eq!(hash_string(&"a".repeat(37)), 13950130860597449643);
        let s = StringStore::new();
        assert_eq!(s.add("NOUN"), 92);
        assert_eq!(s.add("nsubj"), 429);
        assert_eq!(s.add("PERSON"), 380);
        let h = s.add("apple");
        assert_eq!(s.get(h), Some("apple"));
    }
}
