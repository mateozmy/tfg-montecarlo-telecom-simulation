# -*- coding: utf-8 -*-
"""
sim_compare_all.py
Comparativa completa: Uncoded BPSK vs Hamming(7,4) vs Hamming(15,11)
                     vs Conv K=3 vs Conv K=5 vs Conv K=7
Canal: AWGN
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from scipy.interpolate import interp1d
import time

from hamming74 import encode as h74_enc, decode as h74_dec
from hamming1511 import encode as h1511_enc, decode as h1511_dec
from conv_code import ConvolutionalEncoder, ViterbiDecoder, PRESETS
from ldpc import LDPCCodec


EbN0_dB_range = np.arange(0, 11, 1)   # 0..10 dB
MIN_ERR  = 80                           # minimo errores por punto
MAX_BITS = int(3e6)                    # maximo bits simulados
BLOCK_SZ = 4000                        # bloque para Hamming
CONV_MSG = 500                         # longitud mensaje convolucional
LDPC_CODEC = LDPCCodec(300, 3, 6, seed=100)  


def sim_uncoded():
    ber_sim = np.zeros(len(EbN0_dB_range))
    ber_teo = np.zeros(len(EbN0_dB_range))
    for idx, eb in enumerate(EbN0_dB_range):
        eb_lin = 10**(eb / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * eb_lin))
        n_err, n_tot = 0, 0
        while n_err < MIN_ERR and n_tot < MAX_BITS:
            nb = min(BLOCK_SZ, MAX_BITS - n_tot)
            tx = np.random.randint(0, 2, nb).astype(np.uint8)
            sym = 1.0 - 2.0 * tx.astype(float)
            rx = ((sym + sigma * np.random.randn(nb)) < 0).astype(np.uint8)
            n_err += np.sum(rx != tx)
            n_tot += nb
        ber_sim[idx] = n_err / n_tot
        ber_teo[idx] = 0.5 * erfc(np.sqrt(eb_lin))
        print(f"  Uncoded    {eb:2d} dB  BER={ber_sim[idx]:.3e}  "
              f"err={n_err} bits={n_tot}")
    return ber_sim, ber_teo


def sim_hamming(k, n, rate, enc_fn, dec_fn, label):
    ber = np.zeros(len(EbN0_dB_range))
    for idx, eb in enumerate(EbN0_dB_range):
        eb_lin = 10**(eb / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * rate * eb_lin))
        n_err, n_tot, n_corr = 0, 0, 0
        while n_err < MIN_ERR and n_tot < MAX_BITS:
            nb = min(BLOCK_SZ, MAX_BITS - n_tot)
            nb = (nb // k) * k
            if nb == 0:
                break
            nblk = nb // k
            msg = np.random.randint(0, 2, (nblk, k)).astype(np.uint8)
            coded = enc_fn(msg)
            sym = 1.0 - 2.0 * coded.astype(float)
            rx = ((sym + sigma * np.random.randn(*sym.shape)) < 0).astype(np.uint8)
            dec, corr = dec_fn(rx)
            n_err += np.sum(dec != msg)
            n_tot += msg.size
            n_corr += corr
        ber[idx] = n_err / n_tot
        print(f"  {label:16s} {eb:2d} dB  BER={ber[idx]:.3e}  "
              f"err={n_err} bits={n_tot} corr={n_corr}")
    return ber


def sim_conv(preset_name, label):
    K, g1, g2, _ = PRESETS[preset_name]
    enc = ConvolutionalEncoder(K, g1, g2)
    dec = ViterbiDecoder(K, g1, g2)

    ber = np.zeros(len(EbN0_dB_range))
    for idx, eb in enumerate(EbN0_dB_range):
        eb_lin = 10**(eb / 10.0)
        # Tasa efectiva
        rate_eff = CONV_MSG / (2.0 * (CONV_MSG + enc.memory))
        sigma = np.sqrt(1.0 / (2.0 * rate_eff * eb_lin))
        n_err, n_tot = 0, 0
        while n_err < MIN_ERR and n_tot < MAX_BITS:
            # Procesar un mensaje a la vez (mas estable para Viterbi)
            nb = CONV_MSG
            nblk = max(1, 200 // nb)  # hasta 200 bits por lote
            msg = np.random.randint(0, 2, (nblk, nb)).astype(np.uint8)
            coded = enc.encode(msg)
            sym = 1.0 - 2.0 * coded.astype(float)
            shape = sym.shape
            rx = ((sym + sigma * np.random.randn(*shape)) < 0).astype(np.uint8)
            dec_msg, _ = dec.decode(rx)
            e = np.sum(dec_msg != msg)
            n_err += e
            n_tot += msg.size
        ber[idx] = n_err / n_tot
        print(f"  {label:16s} {eb:2d} dB  BER={ber[idx]:.3e}  "
              f"err={n_err} bits={n_tot}")
    return ber


def sim_ldpc(codec, label):
    """Simulacion LDPC con BP decoding."""
    k, n = codec.k, codec.n
    rate = codec.rate

    ber = np.zeros(len(EbN0_dB_range))
    for idx, eb in enumerate(EbN0_dB_range):
        eb_lin = 10**(eb / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * rate * eb_lin))
        n_err, n_tot = 0, 0
        while n_err < MIN_ERR and n_tot < MAX_BITS:
            msg = np.random.randint(0, 2, k).astype(np.uint8)
            cw = codec.encode(msg)
            sym = 2.0 * cw.astype(float) - 1.0   
            rx_soft = sym + sigma * np.random.randn(n)
            dec = codec.decode(rx_soft, noise_sigma=sigma, max_iter=100)
            n_err += np.sum(dec != msg)
            n_tot += k
        ber[idx] = n_err / n_tot
        print(f"  {label:16s} {eb:2d} dB  BER={ber[idx]:.3e}  "
              f"err={n_err} bits={n_tot}")
    return ber


def main():
    print("=" * 72)
    print("SIMULACION COMPARATIVA: TODOS los esquemas completados")
    print("Canal: AWGN   |   Eb/N0: 0..10 dB")
    print("=" * 72)

    t0 = time.time()

    print("\n[1/6] BPSK sin codificacion...")
    ber_unc_sim, ber_unc_teo = sim_uncoded()

    print("\n[2/6] Hamming (7,4)...")
    ber_h74 = sim_hamming(4, 7, 4/7, h74_enc, h74_dec, "Hamming (7,4)")

    print("\n[3/6] Hamming (15,11)...")
    ber_h1511 = sim_hamming(11, 15, 11/15, h1511_enc, h1511_dec, "Hamming(15,11)")

    print("\n[4/6] Convolutional K=3 (7,5)...")
    ber_conv3 = sim_conv('k3', "Conv K=3 (7,5)")

    print("\n[5/6] Convolutional K=5 (23,35)...")
    ber_conv5 = sim_conv('k5', "Conv K=5 (23,35)")

    print("\n[6/7] Convolutional K=7 (171,133)...")
    ber_conv7 = sim_conv('k7', "Conv K=7 (171,133)")

    print("\n[7/7] LDPC (n=300, rate=0.5)...")
    ber_ldpc = sim_ldpc(LDPC_CODEC, "LDPC (300,152)")

    elapsed = time.time() - t0
    print(f"\n[OK] Completado en {elapsed:.1f} s\n")

    print("=" * 130)
    hdr = f"{'Eb/N0':>5}  {'Uncoded(sim)':>14}  {'Uncoded(teo)':>14}"
    hdr += f"  {'H74':>14}  {'H1511':>14}  {'ConvK3':>14}"
    hdr += f"  {'ConvK5':>14}  {'ConvK7':>14}  {'LDPC':>14}"
    print(hdr)
    print("-" * 130)
    for i, eb in enumerate(EbN0_dB_range):
        row = f"{eb:3d} dB"
        row += f"  {ber_unc_sim[i]:14.3e}  {ber_unc_teo[i]:14.3e}"
        row += f"  {ber_h74[i]:14.3e}  {ber_h1511[i]:14.3e}"
        row += f"  {ber_conv3[i]:14.3e}"
        row += f"  {ber_conv5[i]:14.3e}  {ber_conv7[i]:14.3e}"
        row += f"  {ber_ldpc[i]:14.3e}"
        print(row)
    print("=" * 130)

    print("\n--- Coding Gain (BER = 1e-4) ---")

    def eb_at_ber(ber_arr, eb_arr, target=1e-4):
        f = interp1d(np.log10(np.maximum(ber_arr, 1e-12)),
                     eb_arr, kind='linear', fill_value='extrapolate')
        return float(f(np.log10(target)))

    e_unc = eb_at_ber(ber_unc_teo, EbN0_dB_range)
    e_h74 = eb_at_ber(ber_h74, EbN0_dB_range)
    e_h15 = eb_at_ber(ber_h1511, EbN0_dB_range)
    e_c3 = eb_at_ber(ber_conv3, EbN0_dB_range)
    e_c5 = eb_at_ber(ber_conv5, EbN0_dB_range)
    e_c7 = eb_at_ber(ber_conv7, EbN0_dB_range)
    e_ldpc = eb_at_ber(ber_ldpc, EbN0_dB_range)

    print(f"  Uncoded BPSK:        {e_unc:.2f} dB")
    print(f"  Hamming (7,4):       {e_h74:.2f} dB  (gain: {e_unc-e_h74:+.2f} dB)")
    print(f"  Hamming (15,11):     {e_h15:.2f} dB  (gain: {e_unc-e_h15:+.2f} dB)")
    print(f"  Conv K=3  (7,5):    {e_c3:.2f} dB  (gain: {e_unc-e_c3:+.2f} dB)")
    print(f"  Conv K=5  (23,35):  {e_c5:.2f} dB  (gain: {e_unc-e_c5:+.2f} dB)")
    print(f"  Conv K=7  (171,133):{e_c7:.2f} dB  (gain: {e_unc-e_c7:+.2f} dB)")
    print(f"  LDPC (300,152):     {e_ldpc:.2f} dB  (gain: {e_unc-e_ldpc:+.2f} dB)")

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogy(EbN0_dB_range, ber_unc_teo, '--', color='#555555',
                lw=2, label='Uncoded (teorica)')
    ax.semilogy(EbN0_dB_range, ber_unc_sim, 'o', color='#1f77b4',
                mfc='none', ms=6, label='Uncoded (sim)')
    ax.semilogy(EbN0_dB_range, ber_h74, 's-', color='#d62728',
                lw=2, ms=6, label='Hamming (7,4)  R=4/7')
    ax.semilogy(EbN0_dB_range, ber_h1511, 'D-', color='#2ca02c',
                lw=2, ms=6, label='Hamming (15,11) R=11/15')
    ax.semilogy(EbN0_dB_range, ber_conv3, '^-', color='#9467bd',
                lw=2, ms=7, label='Conv K=3 (7,5)  R=1/2')
    ax.semilogy(EbN0_dB_range, ber_conv5, 'P-', color='#ff7f0e',
                lw=2, ms=7, label='Conv K=5 (23,35) R=1/2')
    ax.semilogy(EbN0_dB_range, ber_conv7, 'X-', color='#8c564b',
                lw=2, ms=7, label='Conv K=7 (171,133) R=1/2')
    ax.semilogy(EbN0_dB_range, ber_ldpc, 'h-', color='#e377c2',
                lw=2, ms=7, label='LDPC (300,152) R=0.5')

    ax.set_xlabel('Eb/N0 (dB)', fontsize=13)
    ax.set_ylabel('BER', fontsize=13)
    ax.set_title('Comparacion de Esquemas de Codificacion de Canal\n'
                 'BPSK + AWGN',
                 fontsize=14, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_ylim([1e-6, 1e-1])
    ax.set_xlim([0, 10])

    plt.tight_layout()
    fig.savefig('ber_all_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[FIG] Guardado: ber_all_comparison.png")


if __name__ == '__main__':
    main()
