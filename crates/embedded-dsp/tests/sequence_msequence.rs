//! Maximal-length LFSR sequence tests.

use embedded_dsp::sequence::MSequence;

#[test]
fn msequence_period_balance_and_scramble() {
    assert!(MSequence::default_degree(1).is_err());
    assert!(MSequence::from_genpoly(0).is_err());
    let mut ms = MSequence::default_degree(4).unwrap();
    assert_eq!(ms.degree(), 4);
    assert_eq!(ms.period(), 15);
    assert_eq!(ms.genpoly(), 0x0c);

    let start = ms.state();
    let mut ones = 0u32;
    let mut period = 0u32;
    for _ in 0..32 {
        ones += ms.advance();
        period += 1;
        if ms.state() == start {
            break;
        }
    }
    assert_eq!(period, 15);
    assert_eq!(ones, 8);

    ms.reset();
    assert_eq!(ms.state(), start);
    assert!(ms.generate_symbol(4) < 16);

    let mut ms7 = MSequence::default_degree(7).unwrap();
    let mut buf = [0xAAu8, 0x55, 0x00, 0xFF];
    let orig = buf;
    ms7.scramble_bytes(&mut buf);
    assert_ne!(buf, orig);
    ms7.reset();
    ms7.scramble_bytes(&mut buf);
    assert_eq!(buf, orig);

    ms.set_state(0);
    assert_eq!(ms.advance(), 0);
    assert_eq!(ms.state(), 0);

    assert!(MSequence::new(1, 0x3, 1).is_err());
    assert!(MSequence::new(32, 0x3, 1).is_err());
    let mut m2 = MSequence::new(2, 0x3, 1).unwrap();
    assert_eq!(m2.period(), 3);
    assert_eq!(m2.advance() | m2.advance() | m2.advance(), 1);
}
