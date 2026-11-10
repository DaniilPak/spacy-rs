//! Small, fast hashers for internal maps (keys are mostly already hashes).

use std::hash::{BuildHasherDefault, Hasher};

/// The FxHash algorithm (as used by rustc), good for short strings.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().expect("8")));
        }
        let r = chunks.remainder();
        if !r.is_empty() {
            let mut buf = [0u8; 8];
            buf[..r.len()].copy_from_slice(r);
            self.add(u64::from_le_bytes(buf));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// `HashMap` with [`FxHasher`].
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// `HashSet` with [`FxHasher`].
pub type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

/// Number of lock shards used by concurrent maps.
pub const SHARDS: usize = 16;

/// Pick a shard for an already-hashed key.
#[inline]
pub fn shard_of(h: u64) -> usize {
    (h.wrapping_mul(SEED) >> 60) as usize & (SHARDS - 1)
}

/// FxHash of a string.
#[inline]
pub fn hash_str(s: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(s.as_bytes());
    h.finish()
}
