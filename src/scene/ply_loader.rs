use crate::scene::raw_vertex::RawPlyVertex;
use crate::scene::soa::GaussianSceneSoa;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

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

    #[error("Payload byte size {size} is not a multiple of RawPlyVertex size ({vertex_size} bytes)")]
    InvalidPayloadSize { size: usize, vertex_size: usize },

    #[error("Declared vertex count {declared} does not match binary payload count {actual}")]
    VertexCountMismatch { declared: usize, actual: usize },

    #[error("Payload memory at address {address:#x} is misaligned for RawPlyVertex (requires {align} bytes alignment)")]
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

/// Casts a binary payload slice directly to `&[RawPlyVertex]` zero-copy.
///
/// # Errors
/// Returns `PlyError::MisalignedPayload` if the slice is not aligned to 4 bytes,
/// or `PlyError::InvalidPayloadSize` if the byte length is not a multiple of 248.
pub fn cast_raw_vertices(payload: &[u8]) -> Result<&[RawPlyVertex], PlyError> {
    let align = std::mem::align_of::<RawPlyVertex>();
    let addr = payload.as_ptr() as usize;
    if !addr.is_multiple_of(align) {
        return Err(PlyError::MisalignedPayload {
            address: addr,
            align,
        });
    }

    let vertex_size = std::mem::size_of::<RawPlyVertex>();
    if !payload.len().is_multiple_of(vertex_size) {
        return Err(PlyError::InvalidPayloadSize {
            size: payload.len(),
            vertex_size,
        });
    }

    Ok(bytemuck::cast_slice::<u8, RawPlyVertex>(payload))
}

/// Memory-mapped PLY file loader and parser.
pub struct PlyLoader;

impl PlyLoader {
    /// Loads a 3DGS binary PLY file directly from disk using memory mapping and parallel ingestion.
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<GaussianSceneSoa, PlyError> {
        let file = File::open(path)?;
        // Memory-map file read-only
        let mmap = unsafe { Mmap::map(&file)? };
        Self::load_from_bytes(&mmap)
    }

    /// Parses a byte buffer containing a 3DGS binary PLY file into `GaussianSceneSoa`.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<GaussianSceneSoa, PlyError> {
        // 1. Verify leading 'ply' magic
        if !bytes.starts_with(b"ply\n") && !bytes.starts_with(b"ply\r\n") {
            return Err(PlyError::InvalidMagic);
        }

        // 2. Locate binary payload offset
        let offset = find_end_header(bytes)?;
        let header_bytes = &bytes[..offset];
        let payload = &bytes[offset..];

        let vertex_size = std::mem::size_of::<RawPlyVertex>();
        if !payload.len().is_multiple_of(vertex_size) {
            return Err(PlyError::InvalidPayloadSize {
                size: payload.len(),
                vertex_size,
            });
        }

        let actual_count = payload.len() / vertex_size;

        // 3. Optional header vertex count validation
        if let Ok(header_str) = std::str::from_utf8(header_bytes)
            && let Some(declared_count) = parse_element_vertex_count(header_str)
            && declared_count != actual_count
        {
            return Err(PlyError::VertexCountMismatch {
                declared: declared_count,
                actual: actual_count,
            });
        }

        // 4. Ingestion: zero-copy cast if aligned, safe copy fallback if misaligned
        let align = std::mem::align_of::<RawPlyVertex>();
        if (payload.as_ptr() as usize).is_multiple_of(align) {
            let raw_vertices = bytemuck::cast_slice::<u8, RawPlyVertex>(payload);
            Ok(GaussianSceneSoa::from_raw_vertices_par(raw_vertices))
        } else {
            let mut aligned = vec![RawPlyVertex::ZERO; actual_count];
            let dst_bytes = bytemuck::cast_slice_mut::<RawPlyVertex, u8>(&mut aligned);
            dst_bytes.copy_from_slice(payload);
            Ok(GaussianSceneSoa::from_raw_vertices_par(&aligned))
        }
    }

    /// Zero-copy parser returning a direct `&[RawPlyVertex]` slice from memory-mapped bytes.
    pub fn parse_raw_vertices(bytes: &[u8]) -> Result<&[RawPlyVertex], PlyError> {
        let offset = find_end_header(bytes)?;
        let payload = &bytes[offset..];
        cast_raw_vertices(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_of_header_offset_detection_lf_and_crlf() {
        // Test with standard LF (\n)
        let header_lf = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nend_header\n";
        let offset_lf = find_end_header(header_lf).expect("Failed to find LF end_header");
        assert_eq!(offset_lf, header_lf.len());

        // Test with Windows CRLF (\r\n)
        let header_crlf =
            b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 2\r\nend_header\r\n";
        let offset_crlf = find_end_header(header_crlf).expect("Failed to find CRLF end_header");
        assert_eq!(offset_crlf, header_crlf.len());

        // Test with comments and other content
        let header_with_comments = b"ply\ncomment author: test\ncomment end_header in comment\nelement vertex 10\nend_header\n";
        let offset_comments =
            find_end_header(header_with_comments).expect("Failed with comments");
        assert_eq!(offset_comments, header_with_comments.len());

        // Test missing end_header
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

    #[test]
    fn test_mmap_load_roundtrip() {
        use std::io::Write;

        // Construct aligned header + binary payload
        let mut vertex = RawPlyVertex::ZERO;
        vertex.position = [10.0, -5.0, 3.5];
        vertex.opacity = 2.0;
        vertex.scale = [-0.5, 0.0, 1.2];
        vertex.rot = [1.0, 0.0, 0.0, 0.0];
        vertex.f_dc = [0.5, 0.6, 0.7];

        let mut file_content = Vec::new();
        let base_header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n";
        let end_marker = "end_header\n";
        let pad_len = (4 - (base_header.len() + end_marker.len() + "comment \n".len()) % 4) % 4;
        let pad_str = "x".repeat(pad_len);
        let header_str = format!("{}comment {}\n{}", base_header, pad_str, end_marker);
        assert_eq!(header_str.len() % 4, 0, "Header must be 4-byte aligned for zero-copy");
        file_content.extend_from_slice(header_str.as_bytes());
        file_content.extend_from_slice(vertex.as_bytes());

        // Write to temporary file
        let mut temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&file_content)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush temp file");

        // Load via PlyLoader::load_file
        let soa = PlyLoader::load_file(temp_file.path()).expect("Failed to load PLY file");
        assert_eq!(soa.count, 1);
        assert_eq!(soa.position(0), [10.0, -5.0, 3.5]);
        assert_eq!(&soa.sh_coeffs_at(0)[..3], &[0.5, 0.6, 0.7]);

        // Zero-copy slice test
        let raw_slice =
            PlyLoader::parse_raw_vertices(&file_content).expect("Failed to parse raw vertices");
        assert_eq!(raw_slice.len(), 1);
        assert_eq!(raw_slice[0], vertex);
    }
}
