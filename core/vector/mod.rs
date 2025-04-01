use crate::types::OwnedValue;
use crate::vdbe::Register;
use crate::vdbe::StringPool;
use crate::LimboError;
use crate::Result;

pub mod vector_types;
use vector_types::*;

pub fn vector32(args: &[Register], pool: &mut StringPool) -> Result<OwnedValue> {
    if args.len() != 1 {
        return Err(LimboError::ConversionError(
            "vector32 requires exactly one argument".to_string(),
        ));
    }
    let x = parse_vector(&args[0], Some(VectorType::Float32), pool)?;
    // Extract the Vec<u8> from OwnedValue
    if let OwnedValue::Blob(data) = vector_serialize_f32(x) {
        Ok(OwnedValue::Blob(data))
    } else {
        Err(LimboError::ConversionError(
            "Failed to serialize vector".to_string(),
        ))
    }
}

pub fn vector64(args: &[Register], pool: &mut StringPool) -> Result<OwnedValue> {
    if args.len() != 1 {
        return Err(LimboError::ConversionError(
            "vector64 requires exactly one argument".to_string(),
        ));
    }
    let x = parse_vector(&args[0], Some(VectorType::Float64), pool)?;
    // Extract the Vec<u8> from OwnedValue
    if let OwnedValue::Blob(data) = vector_serialize_f64(x) {
        Ok(OwnedValue::Blob(data))
    } else {
        Err(LimboError::ConversionError(
            "Failed to serialize vector".to_string(),
        ))
    }
}

pub fn vector_extract(args: &[Register], pool: &mut StringPool) -> Result<OwnedValue> {
    if args.len() != 1 {
        return Err(LimboError::ConversionError(
            "vector_extract requires exactly one argument".to_string(),
        ));
    }

    let blob = match &args[0].get_owned_value() {
        OwnedValue::Blob(b) => b,
        _ => {
            return Err(LimboError::ConversionError(
                "Expected blob value".to_string(),
            ))
        }
    };

    if blob.is_empty() {
        return Ok(OwnedValue::build_text("[]", pool));
    }

    let vector_type = vector_type(blob)?;
    let vector = vector_deserialize(vector_type, blob)?;
    Ok(OwnedValue::build_text(&vector_to_text(&vector), pool))
}

pub fn vector_distance_cos(args: &[Register], pool: &mut StringPool) -> Result<OwnedValue> {
    if args.len() != 2 {
        return Err(LimboError::ConversionError(
            "vector_distance_cos requires exactly two arguments".to_string(),
        ));
    }

    let x = parse_vector(&args[0], None, pool)?;
    let y = parse_vector(&args[1], None, pool)?;
    let dist = do_vector_distance_cos(&x, &y)?;
    Ok(OwnedValue::Float(dist))
}
