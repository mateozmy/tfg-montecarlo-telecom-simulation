# -*- coding: utf-8 -*-
"""
Rayleigh flat fading channel model with ideal CSI (Channel State Information)

Model: y = h * x + n
x: transmitted BPSK symbol (+1 or -1)
h: Rayleigh fading coefficient, h ~ CN(0,1), |h| ~ Rayleigh(σ=1/√2)
n: AWGN noise, n ~ CN(0, N0)

With ideal CSI, receiver knows h and performs zero-forcing equalization:
    y_eq = y / h = x + n/h

The effective noise variance becomes N0/|h|^2, which causes SNR degradation.

Theoretical BER for BPSK with ideal CSI in Rayleigh flat fading:
    P_b = (1/2) * (1 - sqrt(γ/(1+γ)))
where γ = Eb/N0 (average SNR per bit)
"""

import numpy as np
from scipy.special import erfc


def rayleigh_channel(sig, EbN0_dB, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    
    N = len(sig)
    
    h_real = rng.standard_normal(N)
    h_imag = rng.standard_normal(N)
    h = (h_real + 1j * h_imag) / np.sqrt(2)
    
    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * EbN0_lin))
    
    n_real = rng.standard_normal(N) * sigma
    n_imag = rng.standard_normal(N) * sigma
    n = (n_real + 1j * n_imag)
    
    y = h * sig + n
    
    y_eq = y / h
    
    return y_eq, h


def ber_uncoded_rayleigh(EbN0_dB):
    """
    
    P_b = (1/2) * (1 - sqrt(γ/(1+γ)))
    
    EbN0_dB : float or ndarray
        Eb/N0 in dB
    
    """
    gamma = 10 ** (EbN0_dB / 10.0)
    return 0.5 * (1 - np.sqrt(gamma / (1 + gamma)))


def ber_uncoded_awgn(EbN0_dB):
    gamma = 10 ** (EbN0_dB / 10.0)
    return 0.5 * erfc(np.sqrt(gamma))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("Testing Rayleigh fading channel model...")
    
    EbN0_range = np.arange(0, 26, 2)  
    N_bits = 500000
    rng = np.random.default_rng(200)
    
    ber_sim = []
    ber_teo = []
    
    for EbN0 in EbN0_range:
        bits = rng.integers(0, 2, N_bits)
        sig = 2 * bits - 1  # BPSK: 0 -> -1, 1 -> +1
        
        y_eq, h = rayleigh_channel(sig, EbN0, rng)
        
        bits_dec = (y_eq.real > 0).astype(int)
        
        err = np.sum(bits != bits_dec)
        ber = err / N_bits
        ber_sim.append(ber)
        
        ber_teo.append(ber_uncoded_rayleigh(EbN0))
        
        print(f"  Eb/N0 = {EbN0:2d} dB: BER_sim = {ber:.2e}, BER_teo = {ber_uncoded_rayleigh(EbN0):.2e}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.semilogy(EbN0_range, ber_sim, 'bo-', label='Simulation', markersize=8)
    plt.semilogy(EbN0_range, ber_teo, 'r--', label='Theory', linewidth=2)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Uncoded BPSK in Rayleigh Flat Fading (Ideal CSI)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim([1e-5, 1])
    plt.savefig('rayleigh_test.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to rayleigh_test.png")
