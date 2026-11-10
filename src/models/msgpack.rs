//! Reading thinc / srsly msgpack files (model weights, moves, lookups).

use crate::Error;
use rmpv::Value;

/// Decode a msgpack blob.
pub fn decode(bytes: &[u8]) -> Result<Value, Error> {
    rmpv::decode::read_value(&mut &bytes[..]).map_err(|e| Error::Format(format!("msgpack: {e}")))
}

/// Look up a key in a msgpack map (string or binary keys).
pub fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_map()?.iter().find_map(|(k, val)| {
        let matches = match k {
            Value::String(s) => s.as_str() == Some(key),
            Value::Binary(b) => b.as_slice() == key.as_bytes(),
            _ => false,
        };
        matches.then_some(val)
    })
}

/// Interpret a value as a UTF-8 string (string or binary).
pub fn as_str(v: &Value) -> Option<&str> {
    match v {
        Value::String(s) => s.as_str(),
        Value::Binary(b) => std::str::from_utf8(b).ok(),
        _ => None,
    }
}

/// An `f32` array decoded from msgpack-numpy encoding.
#[derive(Clone, Debug)]
pub struct Array {
    /// Shape.
    pub shape: Vec<usize>,
    /// Row-major data.
    pub data: Vec<f32>,
}

/// Decode a msgpack-numpy array (`{nd: true, type: "<f4", shape, data}`).
pub fn as_array(v: &Value) -> Result<Array, Error> {
    let ty = get(v, "type").and_then(as_str).unwrap_or("");
    if ty != "<f4" {
        return Err(Error::Format(format!("unsupported array dtype {ty:?}")));
    }
    let shape: Vec<usize> = get(v, "shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| Error::Format("array without shape".into()))?
        .iter()
        .map(|x| x.as_u64().unwrap_or(0) as usize)
        .collect();
    let raw = match get(v, "data") {
        Some(Value::Binary(b)) => b.as_slice(),
        _ => return Err(Error::Format("array without data".into())),
    };
    let n: usize = shape.iter().product();
    if raw.len() != n * 4 {
        return Err(Error::Format("array size mismatch".into()));
    }
    let data = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Array { shape, data })
}

/// A node of a serialized thinc model.
#[derive(Clone, Debug)]
pub struct Node {
    /// Layer name, e.g. `"maxout"`.
    pub name: String,
    /// Parameters by name.
    pub params: Vec<(String, Array)>,
    /// Attributes (msgpack-encoded values, decoded).
    pub attrs: Vec<(String, Value)>,
    /// References to other nodes by name (`model.get_ref`).
    pub refs: Vec<(String, usize)>,
}

impl Node {
    /// A parameter by name.
    pub fn param(&self, name: &str) -> Result<&Array, Error> {
        self.params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a)
            .ok_or_else(|| Error::Format(format!("layer {} has no param {name}", self.name)))
    }
    /// An attribute by name.
    pub fn attr(&self, name: &str) -> Option<&Value> {
        self.attrs.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

/// Load the flat node list of a thinc `Model.to_bytes()` blob (walk order).
pub fn load_nodes(bytes: &[u8]) -> Result<Vec<Node>, Error> {
    let v = decode(bytes)?;
    let nodes = get(&v, "nodes")
        .and_then(|n| n.as_array())
        .ok_or_else(|| Error::Format("no nodes".into()))?;
    let attrs = get(&v, "attrs")
        .and_then(|n| n.as_array())
        .ok_or_else(|| Error::Format("no attrs".into()))?;
    let params = get(&v, "params")
        .and_then(|n| n.as_array())
        .ok_or_else(|| Error::Format("no params".into()))?;
    let mut out = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        let name = get(n, "name").and_then(as_str).unwrap_or("").to_string();
        let mut ps = Vec::new();
        if let Some(Value::Map(m)) = params.get(i) {
            for (k, val) in m {
                if val.is_nil() {
                    continue;
                }
                ps.push((as_str(k).unwrap_or("").to_string(), as_array(val)?));
            }
        }
        let mut ats = Vec::new();
        if let Some(Value::Map(m)) = attrs.get(i) {
            for (k, val) in m {
                let decoded = match val {
                    Value::Binary(b) => decode(b).unwrap_or(Value::Nil),
                    other => other.clone(),
                };
                ats.push((as_str(k).unwrap_or("").to_string(), decoded));
            }
        }
        let refs = match get(n, "refs") {
            Some(Value::Map(m)) => m
                .iter()
                .filter_map(|(k, v)| Some((as_str(k)?.to_string(), v.as_u64()? as usize)))
                .collect(),
            _ => Vec::new(),
        };
        out.push(Node {
            name,
            params: ps,
            attrs: ats,
            refs,
        });
    }
    Ok(out)
}

/// Convert a msgpack value to JSON (map keys become strings).
pub fn to_json(v: &Value) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        Value::Nil => J::Null,
        Value::Boolean(b) => J::Bool(*b),
        Value::Integer(i) => {
            if let Some(u) = i.as_u64() {
                J::from(u)
            } else {
                J::from(i.as_i64().unwrap_or(0))
            }
        }
        Value::F32(f) => J::from(*f as f64),
        Value::F64(f) => J::from(*f),
        Value::String(s) => J::String(s.as_str().unwrap_or("").to_string()),
        Value::Binary(b) => J::String(String::from_utf8_lossy(b).into_owned()),
        Value::Array(a) => J::Array(a.iter().map(to_json).collect()),
        Value::Map(m) => {
            let mut o = serde_json::Map::new();
            for (k, val) in m {
                let key = match k {
                    Value::String(s) => s.as_str().unwrap_or("").to_string(),
                    Value::Integer(i) => i.to_string(),
                    other => format!("{other}"),
                };
                o.insert(key, to_json(val));
            }
            J::Object(o)
        }
        Value::Ext(_, _) => J::Null,
    }
}

/// Integer key of a msgpack map entry (spaCy `Lookups` tables use hashed keys).
pub fn int_key(k: &Value) -> Option<u64> {
    match k {
        Value::Integer(i) => i.as_u64(),
        _ => None,
    }
}
