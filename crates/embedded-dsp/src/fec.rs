//! Forward error detection and short block codes: CRC family and Hamming(7,4).
//!
//! Algorithms follow the reflected bit-at-a-time CRC in *Hacker's Delight* and
//! the Hamming(7,4) generator/decoder tables used by liquid-dsp. Callers own
//! every buffer; there is no heap.

/// Pulse / Hamming encoded length for `n` data bytes packed as nibbles:
/// each byte becomes two 7-bit codewords stored in two output bytes.
#[inline]
pub const fn hamming74_encoded_len(n_bytes: usize) -> usize {
    n_bytes.saturating_mul(2)
}

const fn reverse_bits(mut x: u32, width: u32) -> u32 {
    let mut r = 0u32;
    let mut i = 0u32;
    while i < width {
        r <<= 1;
        r |= x & 1;
        x >>= 1;
        i += 1;
    }
    r
}

const CRC8_POLY: u32 = reverse_bits(0x07, 8);
const CRC16_POLY: u32 = reverse_bits(0x8005, 16);
const CRC24_POLY: u32 = reverse_bits(0x5D6DCB, 24);
const CRC32_POLY: u32 = reverse_bits(0x04C11DB7, 32);

fn crc_reflected(msg: &[u8], width: u32, poly: u32) -> u32 {
    let mask = if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    let mut key = mask;
    for &b in msg {
        key ^= u32::from(b);
        for _ in 0..8 {
            let odd = key & 1;
            key >>= 1;
            if odd != 0 {
                key ^= poly;
            }
        }
    }
    (!key) & mask
}

/// Two's-complement 8-bit checksum: `~(sum) + 1` of the message bytes.
#[inline]
pub fn checksum8(msg: &[u8]) -> u8 {
    let mut sum: u32 = 0;
    for &b in msg {
        sum += u32::from(b);
    }
    (!(sum as u8)).wrapping_add(1)
}

/// CRC-8 (poly `0x07`, reflected, init/xor-out all ones).
#[inline]
pub fn crc8(msg: &[u8]) -> u8 {
    crc_reflected(msg, 8, CRC8_POLY) as u8
}

/// CRC-16 (poly `0x8005`, reflected, init/xor-out all ones).
#[inline]
pub fn crc16(msg: &[u8]) -> u16 {
    crc_reflected(msg, 16, CRC16_POLY) as u16
}

/// CRC-24 (poly `0x5D6DCB`, reflected, init/xor-out all ones).
#[inline]
pub fn crc24(msg: &[u8]) -> u32 {
    crc_reflected(msg, 24, CRC24_POLY)
}

/// CRC-32 (poly `0x04C11DB7`, reflected, init/xor-out all ones; ISO-HDLC / zlib).
#[inline]
pub fn crc32(msg: &[u8]) -> u32 {
    crc_reflected(msg, 32, CRC32_POLY)
}

/// Returns `true` when `crc32(msg)` equals `key`.
#[inline]
pub fn crc32_verify(msg: &[u8], key: u32) -> bool {
    crc32(msg) == key
}

/// Hamming(7,4) encoder table (liquid-dsp `hamming74_enc_gentab`).
const HAMMING74_ENC: [u8; 16] = [
    0x00, 0x69, 0x2a, 0x43, 0x4c, 0x25, 0x66, 0x0f, 0x70, 0x19, 0x5a, 0x33, 0x3c, 0x55, 0x16,
    0x7f,
];

/// Hamming(7,4) decoder table (liquid-dsp `hamming74_dec_gentab`).
const HAMMING74_DEC: [u8; 128] = [
    0x00, 0x00, 0x00, 0x03, 0x00, 0x05, 0x0e, 0x07, 0x00, 0x09, 0x02, 0x07, 0x04, 0x07, 0x07,
    0x07, 0x00, 0x09, 0x0e, 0x0b, 0x0e, 0x0d, 0x0e, 0x0e, 0x09, 0x09, 0x0a, 0x09, 0x0c, 0x09,
    0x0e, 0x07, 0x00, 0x05, 0x02, 0x0b, 0x05, 0x05, 0x06, 0x05, 0x02, 0x01, 0x02, 0x02, 0x0c,
    0x05, 0x02, 0x07, 0x08, 0x0b, 0x0b, 0x0b, 0x0c, 0x05, 0x0e, 0x0b, 0x0c, 0x09, 0x02, 0x0b,
    0x0c, 0x0c, 0x0c, 0x0f, 0x00, 0x03, 0x03, 0x03, 0x04, 0x0d, 0x06, 0x03, 0x04, 0x01, 0x0a,
    0x03, 0x04, 0x04, 0x04, 0x07, 0x08, 0x0d, 0x0a, 0x03, 0x0d, 0x0d, 0x0e, 0x0d, 0x0a, 0x09,
    0x0a, 0x0a, 0x04, 0x0d, 0x0a, 0x0f, 0x08, 0x01, 0x06, 0x03, 0x06, 0x05, 0x06, 0x06, 0x01,
    0x01, 0x02, 0x01, 0x04, 0x01, 0x06, 0x0f, 0x08, 0x08, 0x08, 0x0b, 0x08, 0x0d, 0x06, 0x0f,
    0x08, 0x01, 0x0a, 0x0f, 0x0c, 0x0f, 0x0f, 0x0f,
];

/// Encodes a data nibble (`0..=15`) to a 7-bit Hamming(7,4) codeword in the low bits.
#[inline]
pub fn hamming74_encode_nibble(nibble: u8) -> u8 {
    HAMMING74_ENC[(nibble & 0x0f) as usize]
}

/// Decodes a 7-bit Hamming(7,4) codeword (low 7 bits) to a data nibble, correcting
/// a single-bit error.
#[inline]
pub fn hamming74_decode_nibble(codeword: u8) -> u8 {
    HAMMING74_DEC[(codeword & 0x7f) as usize]
}

/// Encodes each source byte as two Hamming(7,4) codewords (high nibble, then low).
///
/// `dst.len()` must be `2 * src.len()`.
pub fn hamming74_encode_bytes(src: &[u8], dst: &mut [u8]) -> crate::types::Status {
    if dst.len() != src.len() * 2 {
        return crate::types::Status::LengthError;
    }
    for (i, &b) in src.iter().enumerate() {
        dst[2 * i] = hamming74_encode_nibble(b >> 4);
        dst[2 * i + 1] = hamming74_encode_nibble(b);
    }
    crate::types::Status::Success
}

/// Inverse of [`hamming74_encode_bytes`].
pub fn hamming74_decode_bytes(src: &[u8], dst: &mut [u8]) -> crate::types::Status {
    if src.len() != dst.len() * 2 {
        return crate::types::Status::LengthError;
    }
    for (i, out) in dst.iter_mut().enumerate() {
        let hi = hamming74_decode_nibble(src[2 * i]);
        let lo = hamming74_decode_nibble(src[2 * i + 1]);
        *out = (hi << 4) | lo;
    }
    crate::types::Status::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_bits_matches_crc_polys() {
        assert_eq!(reverse_bits(0x07, 8), CRC8_POLY);
        assert_eq!(reverse_bits(0x8005, 16), CRC16_POLY);
        assert_eq!(reverse_bits(0x5D6DCB, 24), CRC24_POLY);
        assert_eq!(reverse_bits(0x04C11DB7, 32), CRC32_POLY);
        assert_eq!(reverse_bits(0, 8), 0);
    }
}
