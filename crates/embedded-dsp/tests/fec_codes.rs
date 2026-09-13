//! CRC, checksum, and Hamming-code tests.

use embedded_dsp::fec::{
    checksum8, crc8, crc16, crc24, crc32, crc32_verify, hamming74_decode_bytes,
    hamming74_decode_nibble, hamming74_encode_bytes, hamming74_encode_nibble,
    hamming74_encoded_len,
};
use embedded_dsp::types::Status;

#[test]
fn crc_and_hamming() {
    let msg = b"123456789";
    assert_eq!(crc32(msg), 0xCBF4_3926);
    assert!(crc32_verify(msg, 0xCBF4_3926));
    assert!(!crc32_verify(msg, 0));
    assert_eq!(crc32(&[]), 0);
    assert_ne!(crc8(msg), 0);
    assert_ne!(crc16(msg), 0);
    assert_ne!(crc24(msg), 0);
    assert_eq!(checksum8(&[1, 2, 3]), (!6u8).wrapping_add(1));

    for n in 0u8..16 {
        let c = hamming74_encode_nibble(n);
        assert_eq!(hamming74_decode_nibble(c), n);
        for bit in 0..7 {
            assert_eq!(hamming74_decode_nibble(c ^ (1 << bit)), n);
        }
    }

    assert_eq!(hamming74_encoded_len(3), 6);
    let src = [0xABu8, 0x00, 0xFF];
    let mut enc = [0u8; 6];
    let mut dec = [0u8; 3];
    assert_eq!(hamming74_encode_bytes(&src, &mut enc), Status::Success);
    enc[1] ^= 1;
    assert_eq!(hamming74_decode_bytes(&enc, &mut dec), Status::Success);
    assert_eq!(dec, src);
    assert_eq!(
        hamming74_encode_bytes(&src, &mut [0u8; 2]),
        Status::LengthError
    );
    assert_eq!(
        hamming74_decode_bytes(&enc, &mut [0u8; 2]),
        Status::LengthError
    );
}
