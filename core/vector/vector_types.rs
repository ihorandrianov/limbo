use crate::types::{OwnedValue, OwnedValueType};
use crate::vdbe::{Register, StringPool};
use crate::{LimboError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum VectorType {
    Float32,
    Float64,
}

impl VectorType {
    pub fn size_to_dims(&self, size: usize) -> usize {
        match self {
            VectorType::Float32 => size / 4,
            VectorType::Float64 => size / 8,
        }
    }
}

#[derive(Debug)]
pub struct Vector {
    pub vector_type: VectorType,
    pub dims: usize,
    pub data: Vec<u8>,
}

impl Vector {
    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.dims) }
    }

    pub fn as_f64_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f64, self.dims) }
    }
}

/// Parse a vector in text representation into a Vector.
///
/// The format of a vector in text representation looks as follows:
///
/// ```console
/// [1.0, 2.0, 3.0]
/// ```
pub fn parse_string_vector(
    vector_type: VectorType,
    value: &OwnedValue,
    pool: &mut StringPool,
) -> Result<Vector> {
    let Some(text) = value.to_text(pool) else {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    };
    let text = text.trim();
    let mut chars = text.chars();
    if chars.next() != Some('[') || chars.last() != Some(']') {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    }
    let mut data: Vec<u8> = Vec::new();
    let text = &text[1..text.len() - 1];
    if text.trim().is_empty() {
        return Ok(Vector {
            vector_type,
            dims: 0,
            data,
        });
    }
    let xs = text.split(',');
    for x in xs {
        let x = x.trim();
        if x.is_empty() {
            return Err(LimboError::ConversionError(
                "Invalid vector value".to_string(),
            ));
        }
        match vector_type {
            VectorType::Float32 => {
                let x = x
                    .parse::<f32>()
                    .map_err(|_| LimboError::ConversionError("Invalid vector value".to_string()))?;
                if !x.is_finite() {
                    return Err(LimboError::ConversionError(
                        "Invalid vector value".to_string(),
                    ));
                }
                data.extend_from_slice(&x.to_le_bytes());
            }
            VectorType::Float64 => {
                let x = x
                    .parse::<f64>()
                    .map_err(|_| LimboError::ConversionError("Invalid vector value".to_string()))?;
                if !x.is_finite() {
                    return Err(LimboError::ConversionError(
                        "Invalid vector value".to_string(),
                    ));
                }
                data.extend_from_slice(&x.to_le_bytes());
            }
        };
    }
    let dims = vector_type.size_to_dims(data.len());
    Ok(Vector {
        vector_type,
        dims,
        data,
    })
}

pub fn parse_vector(
    value: &Register,
    vec_ty: Option<VectorType>,
    pool: &mut StringPool,
) -> Result<Vector> {
    match value.get_owned_value().value_type() {
        OwnedValueType::Text => parse_string_vector(
            vec_ty.unwrap_or(VectorType::Float32),
            value.get_owned_value(),
            pool,
        ),
        OwnedValueType::Blob => {
            let Some(blob) = value.get_owned_value().to_blob() else {
                return Err(LimboError::ConversionError(
                    "Invalid vector value".to_string(),
                ));
            };
            let vector_type = vector_type(&blob)?;
            if let Some(vec_ty) = vec_ty {
                if vec_ty != vector_type {
                    return Err(LimboError::ConversionError(
                        "Invalid vector type".to_string(),
                    ));
                }
            }
            vector_deserialize(vector_type, &blob)
        }
        _ => Err(LimboError::ConversionError(
            "Invalid vector type".to_string(),
        )),
    }
}

pub fn vector_to_text(vector: &Vector) -> String {
    let mut text = String::new();
    text.push('[');
    match vector.vector_type {
        VectorType::Float32 => {
            let data = vector.as_f32_slice();
            for i in 0..vector.dims {
                text.push_str(&data[i].to_string());
                if i < vector.dims - 1 {
                    text.push(',');
                }
            }
        }
        VectorType::Float64 => {
            let data = vector.as_f64_slice();
            for i in 0..vector.dims {
                text.push_str(&data[i].to_string());
                if i < vector.dims - 1 {
                    text.push(',');
                }
            }
        }
    }
    text.push(']');
    text
}

pub fn vector_deserialize(vector_type: VectorType, blob: &[u8]) -> Result<Vector> {
    match vector_type {
        VectorType::Float32 => vector_deserialize_f32(blob),
        VectorType::Float64 => vector_deserialize_f64(blob),
    }
}

pub fn vector_serialize_f64(x: Vector) -> OwnedValue {
    let mut blob = Vec::with_capacity(x.dims * 8 + 1);
    blob.extend_from_slice(&x.data);
    blob.push(2);
    OwnedValue::from_blob(blob)
}

pub fn vector_deserialize_f64(blob: &[u8]) -> Result<Vector> {
    Ok(Vector {
        vector_type: VectorType::Float64,
        dims: (blob.len() - 1) / 8,
        data: blob[..blob.len() - 1].to_vec(),
    })
}

pub fn vector_serialize_f32(x: Vector) -> OwnedValue {
    OwnedValue::from_blob(x.data)
}

pub fn vector_deserialize_f32(blob: &[u8]) -> Result<Vector> {
    Ok(Vector {
        vector_type: VectorType::Float32,
        dims: blob.len() / 4,
        data: blob.to_vec(),
    })
}

pub fn do_vector_distance_cos(v1: &Vector, v2: &Vector) -> Result<f64> {
    match v1.vector_type {
        VectorType::Float32 => vector_f32_distance_cos(v1, v2),
        VectorType::Float64 => vector_f64_distance_cos(v1, v2),
    }
}

pub fn vector_f32_distance_cos(v1: &Vector, v2: &Vector) -> Result<f64> {
    if v1.dims != v2.dims {
        return Err(LimboError::ConversionError(
            "Invalid vector dimensions".to_string(),
        ));
    }
    if v1.vector_type != v2.vector_type {
        return Err(LimboError::ConversionError(
            "Invalid vector type".to_string(),
        ));
    }
    let (mut dot, mut norm1, mut norm2) = (0.0, 0.0, 0.0);
    let v1_data = v1.as_f32_slice();
    let v2_data = v2.as_f32_slice();

    // Check for non-finite values
    if v1_data.iter().any(|x| !x.is_finite()) || v2_data.iter().any(|x| !x.is_finite()) {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    }

    for i in 0..v1.dims {
        let e1 = v1_data[i];
        let e2 = v2_data[i];
        dot += e1 * e2;
        norm1 += e1 * e1;
        norm2 += e2 * e2;
    }

    // Check for zero norms to avoid division by zero
    if norm1 == 0.0 || norm2 == 0.0 {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    }

    Ok(1.0 - (dot / (norm1 * norm2).sqrt()) as f64)
}

pub fn vector_f64_distance_cos(v1: &Vector, v2: &Vector) -> Result<f64> {
    if v1.dims != v2.dims {
        return Err(LimboError::ConversionError(
            "Invalid vector dimensions".to_string(),
        ));
    }
    if v1.vector_type != v2.vector_type {
        return Err(LimboError::ConversionError(
            "Invalid vector type".to_string(),
        ));
    }
    let (mut dot, mut norm1, mut norm2) = (0.0, 0.0, 0.0);
    let v1_data = v1.as_f64_slice();
    let v2_data = v2.as_f64_slice();

    // Check for non-finite values
    if v1_data.iter().any(|x| !x.is_finite()) || v2_data.iter().any(|x| !x.is_finite()) {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    }

    for i in 0..v1.dims {
        let e1 = v1_data[i];
        let e2 = v2_data[i];
        dot += e1 * e2;
        norm1 += e1 * e1;
        norm2 += e2 * e2;
    }

    // Check for zero norms
    if norm1 == 0.0 || norm2 == 0.0 {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    }

    Ok(1.0 - (dot / (norm1 * norm2).sqrt()))
}

pub fn vector_type(blob: &[u8]) -> Result<VectorType> {
    if blob.is_empty() {
        return Err(LimboError::ConversionError(
            "Invalid vector value".to_string(),
        ));
    }
    // Even-sized blobs are always float32.
    if blob.len() % 2 == 0 {
        return Ok(VectorType::Float32);
    }
    // Odd-sized blobs have type byte at the end
    let (data_blob, type_byte) = blob.split_at(blob.len() - 1);
    let vector_type = type_byte[0];
    match vector_type {
        1 => {
            if data_blob.len() % 4 != 0 {
                return Err(LimboError::ConversionError(
                    "Invalid vector value".to_string(),
                ));
            }
            Ok(VectorType::Float32)
        }
        2 => {
            if data_blob.len() % 8 != 0 {
                return Err(LimboError::ConversionError(
                    "Invalid vector value".to_string(),
                ));
            }
            Ok(VectorType::Float64)
        }
        _ => Err(LimboError::ConversionError(
            "Invalid vector type".to_string(),
        )),
    }
}
