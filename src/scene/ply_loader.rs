use crate::scene::raw_vertex::RawPlyVertex;
use crate::scene::soa::GaussianSceneSoa;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;

/// Byte layout declared by the PLY header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

impl PlyFormat {
    /// True for either binary flavour.
    #[inline]
    pub fn is_binary(self) -> bool {
        matches!(self, Self::BinaryLittleEndian | Self::BinaryBigEndian)
    }

    /// True when scalar values are stored least-significant byte first.
    #[inline]
    pub fn is_little_endian(self) -> bool {
        matches!(self, Self::BinaryLittleEndian)
    }
}

/// Scalar types permitted by the PLY specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyScalarType {
    /// Parses a PLY type keyword.
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "char" | "int8" => Some(Self::I8),
            "uchar" | "uint8" => Some(Self::U8),
            "short" | "int16" => Some(Self::I16),
            "ushort" | "uint16" => Some(Self::U16),
            "int" | "int32" => Some(Self::I32),
            "uint" | "uint32" => Some(Self::U32),
            "float" | "float32" => Some(Self::F32),
            "double" | "float64" => Some(Self::F64),
            _ => None,
        }
    }

    /// Size in bytes.
    #[inline]
    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// True for `float`/`double` properties.
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Reads a scalar and widens it to `f32`.
    #[inline]
    pub fn read_f32(self, bytes: &[u8], little_endian: bool) -> f32 {
        match (self, little_endian) {
            (Self::F32, true) => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            (Self::F32, false) => f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            (Self::F64, true) => f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as f32,
            (Self::F64, false) => f64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as f32,
            (Self::U8, _) => f32::from(bytes[0]),
            (Self::I8, _) => f32::from(bytes[0] as i8),
            (Self::U16, true) => f32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            (Self::U16, false) => f32::from(u16::from_be_bytes([bytes[0], bytes[1]])),
            (Self::I16, true) => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])),
            (Self::I16, false) => f32::from(i16::from_be_bytes([bytes[0], bytes[1]])),
            (Self::U32, true) => {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
            }
            (Self::U32, false) => {
                u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
            }
            (Self::I32, true) => {
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
            }
            (Self::I32, false) => {
                i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
            }
        }
    }
}

/// One scalar property inside the vertex element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlyProperty {
    /// Property name exactly as written in the header.
    pub name: String,
    /// Declared scalar type.
    pub ty: PlyScalarType,
    /// Byte offset from the start of the vertex record.
    pub offset: usize,
}

/// Parsed PLY header describing the vertex record layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlyHeader {
    /// Declared payload encoding.
    pub format: PlyFormat,
    /// Number of vertices.
    pub vertex_count: usize,
    /// Bytes per vertex record.
    pub vertex_stride: usize,
    /// Vertex properties in declaration order.
    pub properties: Vec<PlyProperty>,
    /// Byte offset of the binary payload (immediately after `end_header`).
    pub payload_offset: usize,
    /// Spherical-harmonics degree inferred from the number of `f_rest_*` properties.
    pub sh_degree: u32,
}

impl PlyHeader {
    /// Looks up a property by name.
    pub fn property(&self, name: &str) -> Option<&PlyProperty> {
        self.properties
            .iter()
            .find(|property| property.name == name)
    }

    /// Required byte length of the vertex payload.
    #[inline]
    pub fn payload_len(&self) -> usize {
        self.vertex_count * self.vertex_stride
    }
}

/// Errors that can occur during PLY loading and parsing.
#[derive(Debug, Error)]
pub enum PlyError {
    #[error("I/O error reading PLY file: {0}")]
    Io(#[from] std::io::Error),

    #[error("PLY header marker 'end_header' not found")]
    MissingEndHeader,

    #[error("Invalid end_header terminator: expected '\\n' or '\\r\\n'")]
    InvalidHeaderTerminator,

    #[error("Not a valid PLY file (missing leading 'ply' magic header)")]
    InvalidMagic,

    #[error("PLY header is not valid UTF-8")]
    InvalidHeaderUtf8,

    #[error(
        "unsupported PLY format '{0}': only binary_little_endian and binary_big_endian are supported"
    )]
    UnsupportedFormat(String),

    #[error("PLY header declares no 'format' line")]
    MissingFormat,

    #[error("PLY header declares no 'element vertex' count")]
    MissingVertexElement,

    #[error("unsupported vertex layout: {0}")]
    UnsupportedLayout(String),

    #[error("unsupported list property '{0}': PLY lists are not valid in a 3DGS vertex record")]
    UnsupportedListProperty(String),

    #[error("unsupported PLY element '{name}' with {count} records")]
    UnsupportedElement { name: String, count: usize },

    #[error("vertex property '{0}' is required but missing")]
    MissingProperty(String),

    #[error("vertex property '{name}' must be a float, found {ty:?}")]
    InvalidPropertyType { name: String, ty: PlyScalarType },

    #[error(
        "spherical harmonics layout with {count} f_rest properties is unsupported (expected 0, 3, 8, or 45)"
    )]
    UnsupportedShLayout { count: usize },

    #[error("vertex payload is truncated: need {needed} bytes, file provides {available}")]
    TruncatedPayload { needed: usize, available: usize },

    #[error(
        "Payload byte size {size} is not a multiple of RawPlyVertex size ({vertex_size} bytes)"
    )]
    InvalidPayloadSize { size: usize, vertex_size: usize },

    #[error("Declared vertex count {declared} does not match binary payload count {actual}")]
    VertexCountMismatch { declared: usize, actual: usize },

    #[error(
        "Payload memory at address {address:#x} is misaligned for RawPlyVertex (requires {align} bytes alignment)"
    )]
    MisalignedPayload { address: usize, align: usize },

    #[error("Bytemuck cast error: {0:?}")]
    CastError(bytemuck::PodCastError),
}

impl From<bytemuck::PodCastError> for PlyError {
    fn from(err: bytemuck::PodCastError) -> Self {
        Self::CastError(err)
    }
}

/// Locates the byte offset where the binary PLY payload begins.
///
/// Searches for the ASCII marker `end_header\n` or `end_header\r\n`.
/// The binary payload begins immediately at the next byte following the newline.
pub fn find_end_header(data: &[u8]) -> Result<usize, PlyError> {
    const MARKER: &[u8] = b"end_header";
    let mut search_idx = 0;

    while let Some(rel_idx) = data[search_idx..]
        .windows(MARKER.len())
        .position(|w| w == MARKER)
    {
        let pos = search_idx + rel_idx;
        // Verify marker starts at the beginning of a line
        let is_line_start = pos == 0 || data[pos - 1] == b'\n' || data[pos - 1] == b'\r';
        if is_line_start {
            let after = pos + MARKER.len();
            if after < data.len() {
                if data[after..].starts_with(b"\r\n") {
                    return Ok(after + 2);
                } else if data[after..].starts_with(b"\n") {
                    return Ok(after + 1);
                }
            }
        }
        search_idx = pos + 1;
    }

    // Fallback search without line start constraint
    if let Some(pos) = data.windows(MARKER.len()).position(|w| w == MARKER) {
        let after = pos + MARKER.len();
        if after < data.len() {
            if data[after..].starts_with(b"\r\n") {
                return Ok(after + 2);
            } else if data[after..].starts_with(b"\n") {
                return Ok(after + 1);
            }
        }
    }

    Err(PlyError::MissingEndHeader)
}

/// Extracts declared `element vertex <N>` count from the PLY ASCII header.
pub fn parse_element_vertex_count(header: &str) -> Option<usize> {
    for line in header.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("element")
            && parts.next() == Some("vertex")
            && let Some(count_str) = parts.next()
        {
            return count_str.parse::<usize>().ok();
        }
    }
    None
}

const CANONICAL_PREFIX_NAMES: [&str; 6] = ["x", "y", "z", "nx", "ny", "nz"];

/// Parses the ASCII header and validates the vertex schema.
pub fn parse_header(bytes: &[u8]) -> Result<PlyHeader, PlyError> {
    if !bytes.starts_with(b"ply\n") && !bytes.starts_with(b"ply\r\n") {
        return Err(PlyError::InvalidMagic);
    }

    let payload_offset = find_end_header(bytes)?;
    let header_bytes = &bytes[..payload_offset];
    let header_text = std::str::from_utf8(header_bytes).map_err(|_| PlyError::InvalidHeaderUtf8)?;

    let mut format: Option<PlyFormat> = None;
    let mut elements: Vec<(String, usize)> = Vec::new();
    let mut vertex_properties: Vec<PlyProperty> = Vec::new();
    let mut vertex_stride = 0usize;
    let mut current_element: Option<String> = None;

    for line in header_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("comment") || line.starts_with("obj_info") {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("ply") => {}
            Some("format") => {
                let token = parts.next().unwrap_or("");
                format = Some(match token {
                    "ascii" => {
                        return Err(PlyError::UnsupportedFormat(
                            "ascii (re-export the scene as binary_little_endian or binary_big_endian)"
                                .to_string(),
                        ));
                    }
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "binary_big_endian" => PlyFormat::BinaryBigEndian,
                    other => {
                        return Err(PlyError::UnsupportedFormat(other.to_string()));
                    }
                });
            }
            Some("element") => {
                let name = parts.next().unwrap_or("").to_string();
                let count = parts
                    .next()
                    .and_then(|token| token.parse::<usize>().ok())
                    .ok_or_else(|| {
                        PlyError::UnsupportedLayout(format!("element '{name}' has no count"))
                    })?;
                if name == "vertex" {
                    current_element = Some(name.clone());
                } else if current_element.as_deref() == Some("vertex") {
                    // Leave the vertex element context; further properties belong
                    // to the new element.
                    current_element = None;
                }
                elements.push((name, count));
            }
            Some("property") => {
                if current_element.as_deref() != Some("vertex") {
                    continue;
                }

                let maybe_list = parts.next().unwrap_or("");
                if maybe_list == "list" {
                    let name = parts.nth(2).unwrap_or("property").to_string();
                    return Err(PlyError::UnsupportedListProperty(name));
                }

                let ty = PlyScalarType::parse(maybe_list).ok_or_else(|| {
                    PlyError::UnsupportedLayout(format!("unknown property type '{maybe_list}'"))
                })?;
                let name = parts.next().unwrap_or("").to_string();
                if name.is_empty() {
                    return Err(PlyError::UnsupportedLayout(
                        "property declaration without a name".to_string(),
                    ));
                }

                vertex_properties.push(PlyProperty {
                    name,
                    ty,
                    offset: vertex_stride,
                });
                vertex_stride += ty.size();
            }
            Some("end_header") | None => {}
            Some(_) => {}
        }
    }

    let format = format.ok_or(PlyError::MissingFormat)?;

    let vertex_count = elements
        .iter()
        .find(|(name, _)| name == "vertex")
        .map(|(_, count)| *count)
        .ok_or(PlyError::MissingVertexElement)?;

    for (name, count) in &elements {
        if name != "vertex" && *count > 0 {
            return Err(PlyError::UnsupportedElement {
                name: name.clone(),
                count: *count,
            });
        }
    }

    if vertex_count == 0 {
        return Ok(PlyHeader {
            format,
            vertex_count,
            vertex_stride,
            properties: vertex_properties,
            payload_offset,
            sh_degree: 0,
        });
    }

    if vertex_properties.is_empty() {
        return Err(PlyError::UnsupportedLayout(
            "vertex element has no properties".to_string(),
        ));
    }

    // Required scalar properties.
    let required = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ];
    for name in required {
        let Some(property) = vertex_properties.iter().find(|p| p.name == name) else {
            return Err(PlyError::MissingProperty(name.to_string()));
        };
        if !property.ty.is_float() {
            return Err(PlyError::InvalidPropertyType {
                name: name.to_string(),
                ty: property.ty,
            });
        }
    }

    let mut rest_count = 0usize;
    while let Some(property) = vertex_properties
        .iter()
        .find(|property| property.name == format!("f_rest_{rest_count}"))
    {
        if !property.ty.is_float() {
            return Err(PlyError::InvalidPropertyType {
                name: property.name.clone(),
                ty: property.ty,
            });
        }
        rest_count += 1;
    }

    let sh_degree = match rest_count {
        0 => 0,
        3 => 1,
        8 => 2,
        45 => 3,
        count => return Err(PlyError::UnsupportedShLayout { count }),
    };

    Ok(PlyHeader {
        format,
        vertex_count,
        vertex_stride,
        properties: vertex_properties,
        payload_offset,
        sh_degree,
    })
}

/// Casts a binary payload slice directly to `&[RawPlyVertex]` zero-copy.
///
/// # Errors
/// Returns `PlyError::MisalignedPayload` if the slice is not aligned to 4 bytes,
/// or `PlyError::InvalidPayloadSize` if the byte length is not a multiple of 248.
pub fn cast_raw_vertices(payload: &[u8]) -> Result<&[RawPlyVertex], PlyError> {
    let align = std::mem::align_of::<RawPlyVertex>();
    let addr = payload.as_ptr() as usize;
    if (addr % align) != 0 {
        return Err(PlyError::MisalignedPayload {
            address: addr,
            align,
        });
    }

    let vertex_size = std::mem::size_of::<RawPlyVertex>();
    if (payload.len() % vertex_size) != 0 {
        return Err(PlyError::InvalidPayloadSize {
            size: payload.len(),
            vertex_size,
        });
    }

    Ok(bytemuck::cast_slice::<u8, RawPlyVertex>(payload))
}

fn is_canonical_layout(header: &PlyHeader) -> bool {
    if header.format != PlyFormat::BinaryLittleEndian
        || header.vertex_stride != std::mem::size_of::<RawPlyVertex>()
        || header.properties.len() != 62
    {
        return false;
    }

    // Canonical names in canonical order: positions, normals, DC, f_rest_0..44,
    // opacity, scale_0..2, rot_0..3.
    let expected_names: Vec<String> = CANONICAL_PREFIX_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .chain((0..3).map(|i| format!("f_dc_{i}")))
        .chain((0..45).map(|i| format!("f_rest_{i}")))
        .chain(std::iter::once("opacity".to_string()))
        .chain((0..3).map(|i| format!("scale_{i}")))
        .chain((0..4).map(|i| format!("rot_{i}")))
        .collect();

    header
        .properties
        .iter()
        .zip(expected_names.iter())
        .all(|(property, expected)| property.name == *expected && property.ty == PlyScalarType::F32)
}

fn read_property(
    payload: &[u8],
    header: &PlyHeader,
    name: &str,
    row_base: usize,
) -> Result<f32, PlyError> {
    let property = header
        .property(name)
        .ok_or_else(|| PlyError::MissingProperty(name.to_string()))?;
    let start = row_base + property.offset;
    let end = start + property.ty.size();
    if end > payload.len() {
        return Err(PlyError::TruncatedPayload {
            needed: end,
            available: payload.len(),
        });
    }
    Ok(property
        .ty
        .read_f32(&payload[start..end], header.format.is_little_endian()))
}

fn parse_vertex_payload(header: &PlyHeader, payload: &[u8]) -> Result<Vec<RawPlyVertex>, PlyError> {
    let needed = header.payload_len();
    if payload.len() < needed {
        return Err(PlyError::TruncatedPayload {
            needed,
            available: payload.len(),
        });
    }

    let rest_count = match header.sh_degree {
        0 => 0,
        1 => 3,
        2 => 8,
        _ => 45,
    };

    let mut vertices = Vec::with_capacity(header.vertex_count);
    for index in 0..header.vertex_count {
        let base = index * header.vertex_stride;
        let mut vertex = RawPlyVertex::ZERO;

        for (axis, name) in ["x", "y", "z"].iter().enumerate() {
            vertex.position[axis] = read_property(payload, header, name, base)?;
        }
        for (axis, name) in ["nx", "ny", "nz"].iter().enumerate() {
            if header.property(name).is_some() {
                vertex.normal[axis] = read_property(payload, header, name, base)?;
            }
        }
        for (channel, name) in ["f_dc_0", "f_dc_1", "f_dc_2"].iter().enumerate() {
            vertex.f_dc[channel] = read_property(payload, header, name, base)?;
        }
        for rest_index in 0..rest_count {
            let name = format!("f_rest_{rest_index}");
            vertex.f_rest[rest_index] = read_property(payload, header, &name, base)?;
        }

        vertex.opacity = read_property(payload, header, "opacity", base)?;
        for (axis, name) in ["scale_0", "scale_1", "scale_2"].iter().enumerate() {
            vertex.scale[axis] = read_property(payload, header, name, base)?;
        }
        for (axis, name) in ["rot_0", "rot_1", "rot_2", "rot_3"].iter().enumerate() {
            vertex.rot[axis] = read_property(payload, header, name, base)?;
        }

        vertices.push(vertex);
    }

    Ok(vertices)
}

/// Memory-mapped PLY file loader and parser.
pub struct PlyLoader;

impl PlyLoader {
    /// Loads a 3DGS binary PLY file from disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<GaussianSceneSoa, PlyError> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only for the lifetime of the call, and the
        // file is not mutated by this process.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::load_from_bytes(&mmap)
    }

    /// Loads a PLY file from disk without memory mapping (portable target).
    #[cfg(target_arch = "wasm32")]
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<GaussianSceneSoa, PlyError> {
        let bytes = std::fs::read(path)?;
        Self::load_from_bytes(&bytes)
    }

    /// Parses a byte buffer containing a 3DGS binary PLY file into `GaussianSceneSoa`.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<GaussianSceneSoa, PlyError> {
        let header = parse_header(bytes)?;

        if header.vertex_count == 0 {
            return Ok(GaussianSceneSoa::new());
        }

        let payload = &bytes[header.payload_offset..];

        if payload.len() < header.payload_len() {
            return Err(PlyError::TruncatedPayload {
                needed: header.payload_len(),
                available: payload.len(),
            });
        }
        let payload = &payload[..header.payload_len()];

        if is_canonical_layout(&header) {
            let align = std::mem::align_of::<RawPlyVertex>();
            if ((payload.as_ptr() as usize) % align) == 0 {
                return Ok(GaussianSceneSoa::from_raw_vertices_par(
                    bytemuck::cast_slice::<u8, RawPlyVertex>(payload),
                ));
            }
        }

        let vertices = parse_vertex_payload(&header, payload)?;
        Ok(GaussianSceneSoa::from_raw_vertices_par(&vertices))
    }

    /// Zero-copy parser returning a direct `&[RawPlyVertex]` slice from memory-mapped bytes.
    ///
    /// Only valid for the canonical 62-float little-endian layout; other layouts
    /// must go through [`Self::load_from_bytes`].
    pub fn parse_raw_vertices(bytes: &[u8]) -> Result<&[RawPlyVertex], PlyError> {
        let header = parse_header(bytes)?;
        if !is_canonical_layout(&header) {
            return Err(PlyError::UnsupportedLayout(
                "zero-copy view requires the canonical 62-float little-endian layout".to_string(),
            ));
        }
        let payload = &bytes[header.payload_offset..];
        if payload.len() < header.payload_len() {
            return Err(PlyError::TruncatedPayload {
                needed: header.payload_len(),
                available: payload.len(),
            });
        }
        cast_raw_vertices(&payload[..header.payload_len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_text(format: &str, count: usize, properties: &[(&str, &str)]) -> String {
        let mut text = format!("ply\nformat {format}\nelement vertex {count}\n");
        for (ty, name) in properties {
            text.push_str(&format!("property {ty} {name}\n"));
        }
        text.push_str("end_header\n");
        text
    }

    fn standard_properties() -> Vec<(&'static str, &'static str)> {
        vec![
            ("float", "x"),
            ("float", "y"),
            ("float", "z"),
            ("float", "f_dc_0"),
            ("float", "f_dc_1"),
            ("float", "f_dc_2"),
            ("float", "opacity"),
            ("float", "scale_0"),
            ("float", "scale_1"),
            ("float", "scale_2"),
            ("float", "rot_0"),
            ("float", "rot_1"),
            ("float", "rot_2"),
            ("float", "rot_3"),
        ]
    }

    #[test]
    fn test_end_of_header_offset_detection_lf_and_crlf() {
        let header_lf = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nend_header\n";
        let offset_lf = find_end_header(header_lf).expect("Failed to find LF end_header");
        assert_eq!(offset_lf, header_lf.len());

        let header_crlf =
            b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 2\r\nend_header\r\n";
        let offset_crlf = find_end_header(header_crlf).expect("Failed to find CRLF end_header");
        assert_eq!(offset_crlf, header_crlf.len());

        let header_with_comments = b"ply\ncomment author: test\ncomment end_header in comment\nelement vertex 10\nend_header\n";
        let offset_comments = find_end_header(header_with_comments).expect("Failed with comments");
        assert_eq!(offset_comments, header_with_comments.len());

        let invalid_header = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n";
        assert!(matches!(
            find_end_header(invalid_header),
            Err(PlyError::MissingEndHeader)
        ));
    }

    #[test]
    fn test_parse_element_vertex_count() {
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 14205\nproperty float x\nend_header\n";
        assert_eq!(parse_element_vertex_count(header), Some(14205));

        let header_empty = "ply\nformat binary_little_endian 1.0\nend_header\n";
        assert_eq!(parse_element_vertex_count(header_empty), None);
    }

    fn canonical_header(count: usize) -> String {
        let mut text = format!("ply\nformat binary_little_endian 1.0\nelement vertex {count}\n");
        for name in ["x", "y", "z", "nx", "ny", "nz"] {
            text.push_str(&format!("property float {name}\n"));
        }
        for i in 0..3 {
            text.push_str(&format!("property float f_dc_{i}\n"));
        }
        for i in 0..45 {
            text.push_str(&format!("property float f_rest_{i}\n"));
        }
        text.push_str("property float opacity\n");
        for i in 0..3 {
            text.push_str(&format!("property float scale_{i}\n"));
        }
        for i in 0..4 {
            text.push_str(&format!("property float rot_{i}\n"));
        }

        // Align the payload to 4 bytes so the zero-copy path is exercised.
        let partial_len = text.len() + "end_header\n".len() + "comment \n".len();
        let pad = (4 - partial_len % 4) % 4;
        text.push_str(&format!("comment {}\n", "x".repeat(pad)));
        text.push_str("end_header\n");
        text
    }

    #[test]
    fn test_mmap_load_roundtrip() {
        use std::io::Write;

        let mut vertex = RawPlyVertex::ZERO;
        vertex.position = [10.0, -5.0, 3.5];
        vertex.opacity = 2.0;
        vertex.scale = [-0.5, 0.0, 1.2];
        vertex.rot = [1.0, 0.0, 0.0, 0.0];
        vertex.f_dc = [0.5, 0.6, 0.7];

        let header_str = canonical_header(1);
        assert_eq!(
            header_str.len() % 4,
            0,
            "Header must be 4-byte aligned for zero-copy"
        );

        let mut file_content = Vec::new();
        file_content.extend_from_slice(header_str.as_bytes());
        file_content.extend_from_slice(vertex.as_bytes());

        let mut temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&file_content)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush temp file");

        let soa = PlyLoader::load_file(temp_file.path()).expect("Failed to load PLY file");
        assert_eq!(soa.count, 1);
        assert_eq!(soa.position(0), [10.0, -5.0, 3.5]);
        assert_eq!(&soa.sh_coeffs_at(0)[..3], &[0.5, 0.6, 0.7]);

        let raw_slice =
            PlyLoader::parse_raw_vertices(&file_content).expect("Failed to parse raw vertices");
        assert_eq!(raw_slice.len(), 1);
        assert_eq!(raw_slice[0], vertex);
    }

    #[test]
    fn test_ascii_is_rejected_with_a_clear_error() {
        let header = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n";
        match parse_header(header) {
            Err(PlyError::UnsupportedFormat(message)) => {
                assert!(message.contains("ascii"), "message: {message}");
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_required_property_is_rejected() {
        let text = header_text(
            "binary_little_endian 1.0",
            1,
            &[("float", "x"), ("float", "y"), ("float", "z")],
        );
        match parse_header(text.as_bytes()) {
            Err(PlyError::MissingProperty(name)) => assert_eq!(name, "f_dc_0"),
            other => panic!("expected MissingProperty, got {other:?}"),
        }
    }

    #[test]
    fn test_reordered_and_extra_properties_parse() {
        // Reordered properties plus an ignored integer property.
        let props = vec![
            ("float", "scale_0"),
            ("float", "scale_1"),
            ("float", "scale_2"),
            ("float", "rot_0"),
            ("float", "rot_1"),
            ("float", "rot_2"),
            ("float", "rot_3"),
            ("float", "opacity"),
            ("float", "f_dc_0"),
            ("float", "f_dc_1"),
            ("float", "f_dc_2"),
            ("int", "cluster_id"),
            ("float", "x"),
            ("float", "y"),
            ("float", "z"),
        ];
        let text = header_text("binary_little_endian 1.0", 1, &props);
        let mut bytes = text.into_bytes();
        // Append a payload writing values by the declared order.
        let values = [
            0.0f32, 0.0, 0.0, // scale
            1.0, 0.0, 0.0, 0.0, // rot
            3.0, // opacity
            0.1, 0.2, 0.3, // f_dc
        ];
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&7i32.to_le_bytes()); // cluster_id
        for value in [1.5f32, -2.0, 0.25] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let scene = PlyLoader::load_from_bytes(&bytes).expect("generic layout must parse");
        assert_eq!(scene.count, 1);
        assert_eq!(scene.position(0), [1.5, -2.0, 0.25]);
        assert!((scene.opacity(0) - crate::math::activate_opacity(3.0)).abs() < 1e-6);
        assert!((scene.sh_coeffs_at(0)[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_big_endian_payload_parses() {
        let props = standard_properties();
        let text = header_text("binary_big_endian 1.0", 1, &props);
        let mut bytes = text.into_bytes();

        let values: [f32; 14] = [
            1.0, 2.0, 3.0, // x,y,z
            0.1, 0.2, 0.3, // f_dc
            0.5, // opacity
            -1.0, 0.0, 1.0, // scale
            1.0, 0.0, 0.0, 0.0, // rot
        ];
        for value in values {
            bytes.extend_from_slice(&value.to_be_bytes());
        }

        let scene = PlyLoader::load_from_bytes(&bytes).expect("big-endian PLY must parse");
        assert_eq!(scene.count, 1);
        assert_eq!(scene.position(0), [1.0, 2.0, 3.0]);
        assert_eq!(&scene.sh_coeffs_at(0)[..3], &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_sh_degree_detection() {
        for (rest, expected) in [(0usize, 0u32), (3, 1), (8, 2), (45, 3)] {
            let mut props = standard_properties();
            let dynamic_names: Vec<String> = (0..rest).map(|i| format!("f_rest_{i}")).collect();
            for name in &dynamic_names {
                props.push(("float", Box::leak(name.clone().into_boxed_str())));
            }
            let text = header_text("binary_little_endian 1.0", 1, &props);
            let header = parse_header(text.as_bytes()).expect("header parses");
            assert_eq!(header.sh_degree, expected, "rest count {rest}");
        }
    }

    #[test]
    fn test_unsupported_sh_layout_is_rejected() {
        let mut props = standard_properties();
        props.push(("float", "f_rest_0"));
        props.push(("float", "f_rest_1"));
        let text = header_text("binary_little_endian 1.0", 1, &props);
        // Payload absent: the layout error is reported before truncation.
        match parse_header(text.as_bytes()) {
            Err(PlyError::UnsupportedShLayout { count }) => assert_eq!(count, 2),
            other => panic!("expected UnsupportedShLayout, got {other:?}"),
        }
    }

    #[test]
    fn test_truncated_payload_is_rejected() {
        let props = standard_properties();
        let text = header_text("binary_little_endian 1.0", 2, &props);
        let mut bytes = text.into_bytes();
        bytes.extend_from_slice(&[0u8; 8]);
        match PlyLoader::load_from_bytes(&bytes) {
            Err(PlyError::TruncatedPayload { needed, available }) => {
                assert_eq!(needed, 2 * 14 * 4);
                assert_eq!(available, 8);
            }
            other => panic!("expected TruncatedPayload, got {other:?}"),
        }
    }

    #[test]
    fn test_list_properties_are_rejected() {
        let text = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty list uchar int vertex_indices\nend_header\n";
        assert!(matches!(
            parse_header(text.as_bytes()),
            Err(PlyError::UnsupportedListProperty(_))
        ));
    }

    #[test]
    fn test_non_zero_extra_element_is_rejected() {
        let text = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n";
        let mut bytes = text.as_bytes().to_vec();
        let patched = String::from_utf8(bytes.clone()).expect("utf8");
        bytes = patched
            .replace(
                "element vertex 1",
                "element vertex 1\nelement face 2\nproperty list uchar int vertex_indices",
            )
            .into_bytes();
        match parse_header(&bytes) {
            Err(PlyError::UnsupportedListProperty(_))
            | Err(PlyError::UnsupportedElement { .. }) => {}
            other => panic!("expected extra-element rejection, got {other:?}"),
        }
    }
}
