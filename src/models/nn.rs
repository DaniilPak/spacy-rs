//! Minimal `f32` implementations of the thinc layers used by spaCy's CNN
//! pipelines (`HashEmbed`, `Maxout`, `LayerNorm`, `Linear`, `expand_window`,
//! `PrecomputableAffine`).

use super::msgpack::{Array, Node};
use crate::Error;

/// thinc's `MurmurHash3_x86_128_uint64` (used by `HashEmbed`).
pub fn murmur3_x86_128_u64(val: u64, seed: u32) -> [u32; 4] {
    let mut h1: u64 = val;
    h1 = h1.wrapping_mul(0x87c3_7b91_1142_53d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5_ad43_2745_937f);
    h1 ^= seed as u64;
    h1 ^= 8;
    let mut h2: u64 = seed as u64;
    h2 ^= 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let fmix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^= k >> 33;
        k
    };
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

/// numpy's pairwise summation for float32 (used by `mean`/`var`).
pub fn pairwise_sum(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f32;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let (x, y) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// `Y = X @ W.T + b` for a row-major `X (n, nI)` and `W (nO, nI)`.
pub fn affine(
    x: &[f32],
    n: usize,
    n_in: usize,
    w: &[f32],
    b: Option<&[f32]>,
    n_out: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; n * n_out];
    if n == 1 {
        for o in 0..n_out {
            y[o] = dot(x, &w[o * n_in..(o + 1) * n_in]) + b.map_or(0.0, |b| b[o]);
        }
        return y;
    }
    if let Some(b) = b {
        for row in y.chunks_exact_mut(n_out) {
            row.copy_from_slice(b);
        }
    }
    // Y (n x nO) = X (n x nI) * W^T (nI x nO); W is row-major (nO x nI).
    // SAFETY: all slices are sized n*n_in, n_out*n_in and n*n_out.
    unsafe {
        matrixmultiply::sgemm(
            n,
            n_in,
            n_out,
            1.0,
            x.as_ptr(),
            n_in as isize,
            1,
            w.as_ptr(),
            1,
            n_in as isize,
            if b.is_some() { 1.0 } else { 0.0 },
            y.as_mut_ptr(),
            n_out as isize,
            1,
        );
    }
    y
}

/// `Linear` / `Softmax` weights.
#[derive(Clone, Debug)]
pub struct Linear {
    /// `(nO, nI)`.
    pub w: Vec<f32>,
    /// `(nO,)`.
    pub b: Vec<f32>,
    /// Output width.
    pub n_out: usize,
    /// Input width.
    pub n_in: usize,
}

impl Linear {
    /// From a `linear` or `softmax` node.
    pub fn from_node(n: &Node) -> Result<Self, Error> {
        let w = n.param("W")?;
        let b = n.param("b")?;
        Ok(Linear {
            n_out: w.shape[0],
            n_in: w.shape[1],
            w: w.data.clone(),
            b: b.data.clone(),
        })
    }
    /// Forward `X (n, nI)`.
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        affine(x, n, self.n_in, &self.w, Some(&self.b), self.n_out)
    }
}

/// `Maxout` (optionally followed by `LayerNorm`).
#[derive(Clone, Debug)]
pub struct Maxout {
    w: Vec<f32>,
    b: Vec<f32>,
    n_out: usize,
    n_piece: usize,
    n_in: usize,
    norm: Option<(Vec<f32>, Vec<f32>)>,
}

impl Maxout {
    /// From a `maxout` node and an optional `layernorm` node.
    pub fn from_nodes(m: &Node, ln: Option<&Node>) -> Result<Self, Error> {
        let w: &Array = m.param("W")?;
        let b = m.param("b")?;
        let norm = match ln {
            Some(l) => Some((l.param("G")?.data.clone(), l.param("b")?.data.clone())),
            None => None,
        };
        Ok(Maxout {
            n_out: w.shape[0],
            n_piece: w.shape[1],
            n_in: w.shape[2],
            w: w.data.clone(),
            b: b.data.clone(),
            norm,
        })
    }
    /// Input width.
    pub fn n_in(&self) -> usize {
        self.n_in
    }
    /// Forward `X (n, nI)` → `(n, nO)`.
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let op = self.n_out * self.n_piece;
        let z = affine(x, n, self.n_in, &self.w, Some(&self.b), op);
        let mut y = vec![0f32; n * self.n_out];
        for i in 0..n {
            for o in 0..self.n_out {
                let base = i * op + o * self.n_piece;
                let mut best = z[base];
                for p in 1..self.n_piece {
                    if z[base + p] > best {
                        best = z[base + p];
                    }
                }
                y[i * self.n_out + o] = best;
            }
        }
        if let Some((g, bb)) = &self.norm {
            layer_norm(&mut y, n, self.n_out, g, bb);
        }
        y
    }
}

/// thinc `LayerNorm` in place.
pub fn layer_norm(y: &mut [f32], n: usize, d: usize, g: &[f32], b: &[f32]) {
    let mut tmp = vec![0f32; d];
    for i in 0..n {
        let row = &mut y[i * d..(i + 1) * d];
        let mu = pairwise_sum(row) / d as f32;
        for k in 0..d {
            let x = row[k] - mu;
            tmp[k] = x * x;
        }
        let var = pairwise_sum(&tmp) / d as f32 + 1e-8;
        let inv = var.powf(-0.5);
        for k in 0..d {
            row[k] = (row[k] - mu) * inv * g[k] + b[k];
        }
    }
}

/// `expand_window(window_size)`: concatenate neighbours, zero at edges.
pub fn seq2col(x: &[f32], n: usize, d: usize, nw: usize) -> Vec<f32> {
    let w = d * (2 * nw + 1);
    let mut out = vec![0f32; n * w];
    for i in 0..n {
        for (slot, off) in (-(nw as isize)..=nw as isize).enumerate() {
            let j = i as isize + off;
            if j >= 0 && (j as usize) < n {
                let j = j as usize;
                out[i * w + slot * d..i * w + (slot + 1) * d]
                    .copy_from_slice(&x[j * d..(j + 1) * d]);
            }
        }
    }
    out
}

/// `HashEmbed`.
#[derive(Clone, Debug)]
pub struct HashEmbed {
    e: Vec<f32>,
    n_v: usize,
    n_o: usize,
    seed: u32,
}

impl HashEmbed {
    /// From a `hashembed` node.
    pub fn from_node(n: &Node) -> Result<Self, Error> {
        let e = n.param("E")?;
        let seed = n
            .attr("seed")
            .and_then(|s| s.as_u64())
            .ok_or_else(|| Error::Format("hashembed without seed".into()))?;
        Ok(HashEmbed {
            n_v: e.shape[0],
            n_o: e.shape[1],
            e: e.data.clone(),
            seed: seed as u32,
        })
    }
    /// Output width.
    pub fn width(&self) -> usize {
        self.n_o
    }
    /// Embed one ID into `out` (sum of four hashed rows).
    pub fn embed_into(&self, id: u64, out: &mut [f32]) {
        let keys = murmur3_x86_128_u64(id, self.seed);
        out.iter_mut().for_each(|x| *x = 0.0);
        for k in keys {
            let row = (k as usize) % self.n_v;
            let r = &self.e[row * self.n_o..(row + 1) * self.n_o];
            for (o, v) in out.iter_mut().zip(r) {
                *o += *v;
            }
        }
    }
}

/// `PrecomputableAffine` (the parser's lower layer).
#[derive(Clone, Debug)]
pub struct PrecomputableAffine {
    /// `(nF, nO, nP, nI)`.
    pub w: Vec<f32>,
    /// `(nO, nP)`.
    pub b: Vec<f32>,
    /// `(nF, nO, nP)`.
    pub pad: Vec<f32>,
    /// Feature slots.
    pub n_f: usize,
    /// Hidden width.
    pub n_o: usize,
    /// Maxout pieces.
    pub n_p: usize,
    /// Input width.
    pub n_i: usize,
}

impl PrecomputableAffine {
    /// From a `precomputable_affine` node.
    pub fn from_node(n: &Node) -> Result<Self, Error> {
        let w = n.param("W")?;
        Ok(PrecomputableAffine {
            n_f: w.shape[0],
            n_o: w.shape[1],
            n_p: w.shape[2],
            n_i: w.shape[3],
            w: w.data.clone(),
            b: n.param("b")?.data.clone(),
            pad: n.param("pad")?.data.clone(),
        })
    }
    /// Precompute `X @ W.T` for every token: `(n, nF*nO*nP)`.
    pub fn precompute(&self, x: &[f32], n: usize) -> Vec<f32> {
        affine(
            x,
            n,
            self.n_i,
            &self.w,
            None,
            self.n_f * self.n_o * self.n_p,
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn murmur_matches_thinc() {
        // values from thinc.api.NumpyOps().hash(numpy.array([...], dtype="uint64"), seed)
        assert_eq!(super::murmur3_x86_128_u64(0, 8), [0, 0, 0, 0]);
        assert_eq!(
            super::murmur3_x86_128_u64(8566208034543834098, 8),
            [2653853934, 3922826080, 3980780901, 3736755472]
        );
        assert_eq!(
            super::murmur3_x86_128_u64(92, 8),
            [777148692, 2333324333, 3313206686, 3499986499]
        );
        assert_eq!(
            super::murmur3_x86_128_u64(0, 12),
            [3628889788, 3483432925, 485828337, 1486234390]
        );
        assert_eq!(
            super::murmur3_x86_128_u64(1, 12),
            [2938686206, 2291508213, 2246688320, 3554967022]
        );
    }
}
