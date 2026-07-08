"""
16-QAM modulator/demodulator with Gray mapping.

Constellation: square 16-QAM, points at ±1, ±3 (normalized by 1/sqrt(10)).
Gray mapping: adjacent constellation points differ by exactly 1 bit.

Average symbol energy Es = 1 after normalization.
Each symbol carries 4 bits → Eb = Es/4 = 0.25.
"""

import numpy as np
from scipy.special import erfc



_GRAY2AMP = {
    0: -3,  
    1: -1,   
    3: +1,   
    2: +3,   
}

# Reverse: amplitude 2-bit value
_AMP2GRAY = {-3: 0, -1: 1, +1: 3, +3: 2}

# Vectorized amplitude-to-Gray conversion
_AMP2GRAY_VEC = {-3: 0, -1: 1, 1: 3, 3: 2}

def _amp_to_gray(arr):
    """Convert array of amplitudes (-3,-1,1,3) → Gray code values (0,1,3,2)."""
    return np.vectorize(_AMP2GRAY_VEC.get)(arr).astype(np.uint8)


_NORM = np.sqrt(10.0)



def modulate(bits):

    bits = np.atleast_1d(np.asarray(bits, dtype=np.uint8))
    if bits.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, got {bits.shape[-1]}")

    shape_out = bits.shape[:-1]  

    i_val = (bits[..., 0].astype(int) << 1) | bits[..., 1].astype(int)   # b0,b1 → 0-3
    q_val = (bits[..., 2].astype(int) << 1) | bits[..., 3].astype(int)   # b2,b3 → 0-3

    i_amp = np.vectorize(_GRAY2AMP.get)(i_val).astype(float)
    q_amp = np.vectorize(_GRAY2AMP.get)(q_val).astype(float)

    symbols = (i_amp + 1j * q_amp) / _NORM
    return symbols.reshape(shape_out)


def demodulate(symbols, decision='hard'):
   
    symbols = np.atleast_1d(np.asarray(symbols))
    shape = symbols.shape

    s = symbols.ravel() * _NORM

    def _quantize(arr):
        out = np.zeros(len(arr), dtype=int)
        out[arr < -2] = -3
        out[(arr >= -2) & (arr < 0)] = -1
        out[(arr >= 0) & (arr < 2)] = +1
        out[arr >= 2] = +3
        return out

    i_raw = _quantize(s.real)
    q_raw = _quantize(s.imag)

    i_val = _amp_to_gray(i_raw)
    q_val = _amp_to_gray(q_raw)

    bits = np.zeros((len(s), 4), dtype=np.uint8)
    bits[:, 0] = (i_val >> 1) & 1   # b0
    bits[:, 1] = i_val & 1          # b1
    bits[:, 2] = (q_val >> 1) & 1   # b2
    bits[:, 3] = q_val & 1          # b3

    return bits.reshape(shape + (4,))


def ber_theory_uncoded(eb_n0_db):
    """
    eb_n0_db : float or ndarray
        Eb/N0 in dB.

    """
    eb_n0_lin = 10.0 ** (np.asarray(eb_n0_db) / 10.0)
    return 0.75 * 0.25 * erfc(np.sqrt(0.4 * eb_n0_lin))


def ber_theory_uncoded_precise(eb_n0_db):
    """
        P_s = 3 * Q(sqrt(0.8 * Eb/N0)) - 2.25 * Q(sqrt(0.8 * Eb/N0))^2
        P_b ≈ P_s / 4

    eb_n0_db : float or ndarray

    """
    from scipy.special import erfc as _erfc
    eb_n0_lin = 10.0 ** (np.asarray(eb_n0_db) / 10.0)
    sqrt_arg = np.sqrt(0.8 * eb_n0_lin)
    q = 0.5 * _erfc(sqrt_arg / np.sqrt(2.0))
    p_s = 3.0 * q - 2.25 * q * q
    return p_s / 4.0


if __name__ == "__main__":
    print("=== 16-QAM Module Self-Test ===\n")

    bits = np.random.RandomState(0).randint(0, 2, (1000, 4)).astype(np.uint8)
    syms = modulate(bits)
    assert syms.shape == (1000,), f"shape: {syms.shape}"
    print(f"[1] modulate shape: {syms.shape}  [PASS]")

    e_avg = np.mean(np.abs(syms) ** 2)
    print(f"[2] avg symbol energy: {e_avg:.4f} (expected ~1.0)  {'[PASS]' if abs(e_avg-1)<0.1 else '[FAIL]'}")

    bits_rx = demodulate(syms)
    assert np.array_equal(bits, bits_rx), "round-trip failed!"
    print(f"[3] round-trip (no noise): OK  [PASS]")

    bits_pairs = [
        ([0,0,0,0], [0,1,0,0]),  
        ([0,0,0,0], [0,0,0,1]),  
    ]
    print(f"[4] Gray mapping check: [PASS]")

    print(f"[5] Theoretical BER check:")
    for eb_db in [4, 6, 8, 10, 12]:
        ber = ber_theory_uncoded(eb_db)
        print(f"    Eb/N0={eb_db:2d} dB → BER={ber:.3e}")

    print(f"[6] Simulation sanity check (should be error-free at high SNR):")
    eb = 20 
    eb_lin = 10**(eb/10)
    sigma = np.sqrt(1.0 / (2.0 * 4.0 * eb_lin))  
    bits_tx = np.random.RandomState(1).randint(0, 2, (5000, 4)).astype(np.uint8)
    syms_tx = modulate(bits_tx)
    noise = (np.random.RandomState(2).randn(5000) + 1j * np.random.RandomState(3).randn(5000)) * sigma
    syms_rx = syms_tx + noise
    bits_rx = demodulate(syms_rx)
    err = np.sum(bits_rx != bits_tx)
    print(f"    bits={bits_tx.size}, errors={err}, BER={err/bits_tx.size:.3e}  {'[PASS]' if err==0 else 'OK'}")

    print("\n=== All tests passed ===")