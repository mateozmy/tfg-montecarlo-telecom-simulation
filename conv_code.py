# -*- coding: utf-8 -*-
"""
Codigo convolucional (tasa 1/2) + decodificador Viterbi (hard-decision).

  - K=3, generadores (7, 5) octal
  - K=5, generadores (23, 35) octal
  - K=7, generadores (171, 133) octal
"""

import numpy as np


PRESETS = {
    # (constraint_length, g1_octal, g2_octal, label)
    'k3':  (3,  0o7,   0o5,   'Conv K=3, R=1/2'),
    'k5':  (5,  0o23,  0o35,  'Conv K=5, R=1/2'),
    'k7':  (7,  0o171, 0o133, 'Conv K=7, R=1/2'),
}


class ConvolutionalEncoder:
    """
    Codificador convolucional tasa 1/2.

    Convencion de estado:
      state = s_0 + 2*s_1 + ... + 2^(K-2)*s_{K-2}
      s_0 = bit mas reciente (sin incluir el actual)
      s_{K-2} = bit mas antiguo

    Al llegar nuevo bit u:
      reg = (state << 1) | u           (K bits)
      out = XOR taps definidos por g1, g2
      next_state = reg & ((1<<(K-1))-1)  (K-1 lower bits)
    """

    def __init__(self, K, g1, g2):
        self.K = K
        self.memory = K - 1
        self.num_states = 1 << self.memory
        self.g1 = g1
        self.g2 = g2
        self.mask = (1 << self.memory) - 1

        self._build_trellis_table()

    def _build_trellis_table(self):
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.out1 = np.zeros((self.num_states, 2), dtype=np.uint8)
        self.out2 = np.zeros((self.num_states, 2), dtype=np.uint8)

        for state in range(self.num_states):
            for inp in range(2):
                reg = (state << 1) | inp 
                self.next_state[state, inp] = reg & self.mask  # [FIXED]
                self.out1[state, inp] = bin(reg & self.g1).count('1') % 2
                self.out2[state, inp] = bin(reg & self.g2).count('1') % 2

    def encode(self, msg_bits):
       
        msg_bits = np.atleast_2d(msg_bits).astype(np.uint8)
        batch, msg_len = msg_bits.shape

        padded = np.hstack([
            msg_bits,
            np.zeros((batch, self.memory), dtype=np.uint8)
        ])
        total_steps = msg_len + self.memory

        coded = np.zeros((batch, total_steps * 2), dtype=np.uint8)

        for b in range(batch):
            state = 0
            for i in range(total_steps):
                inp = padded[b, i]
                coded[b, 2*i]     = self.out1[state, inp]
                coded[b, 2*i + 1] = self.out2[state, inp]
                state = self.next_state[state, inp]

        return coded

    def encode_single(self, bits):
        return self.encode(bits.reshape(1, -1)).flatten()

    def __repr__(self):
        return f"ConvolutionalEncoder(K={self.K}, g1={oct(self.g1)}, g2={oct(self.g2)})"


class ViterbiDecoder:
    def __init__(self, K, g1, g2):
        self.K = K
        self.memory = K - 1
        self.num_states = 1 << self.memory
        self.g1 = g1
        self.g2 = g2
        self.mask = (1 << self.memory) - 1

        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.out1 = np.zeros((self.num_states, 2), dtype=np.uint8)
        self.out2 = np.zeros((self.num_states, 2), dtype=np.uint8)

        for state in range(self.num_states):
            for inp in range(2):
                reg = (state << 1) | inp
                self.next_state[state, inp] = reg & self.mask
                self.out1[state, inp] = bin(reg & self.g1).count('1') % 2
                self.out2[state, inp] = bin(reg & self.g2).count('1') % 2

        self.ns_0 = self.next_state[:, 0]
        self.ns_1 = self.next_state[:, 1]
        self.out_sym_0 = (self.out1[:, 0].astype(np.int32) << 1) | self.out2[:, 0].astype(np.int32)
        self.out_sym_1 = (self.out1[:, 1].astype(np.int32) << 1) | self.out2[:, 1].astype(np.int32)

        self._hdist = np.array([
            [0, 1, 1, 2],
            [1, 0, 2, 1],
            [1, 2, 0, 1],
            [2, 1, 1, 0],
        ], dtype=np.int32)

    def decode(self, rx_coded):
        rx = np.atleast_2d(rx_coded).astype(np.uint8)
        batch, rx_len = rx.shape
        total_steps = rx_len // 2
        msg_len = total_steps - self.memory
        NS = self.num_states

        decoded = np.zeros((batch, msg_len), dtype=np.uint8)
        total_corrections = 0

        for b in range(batch):
            pm = np.full(NS, np.inf, dtype=np.float64)
            pm[0] = 0.0
            survivor = np.zeros((total_steps, NS), dtype=np.int32)

            for t in range(total_steps):
                rx_sym = (int(rx[b, 2*t]) << 1) | int(rx[b, 2*t + 1])

                bm0 = self._hdist[rx_sym, self.out_sym_0]
                bm1 = self._hdist[rx_sym, self.out_sym_1]
                cand0 = pm + bm0
                cand1 = pm + bm1

                new_pm = np.full(NS, np.inf, dtype=np.float64)

                for s in range(NS):
                    ns = self.ns_0[s]
                    c = cand0[s]
                    if c < new_pm[ns]:
                        new_pm[ns] = c
                        survivor[t, ns] = s

                    ns = self.ns_1[s]
                    c = cand1[s]
                    if c < new_pm[ns]:
                        new_pm[ns] = c
                        survivor[t, ns] = s

                pm = new_pm

            best_s = int(np.argmin(pm))
            dec_seq = np.zeros(total_steps, dtype=np.uint8)
            for t in range(total_steps - 1, -1, -1):
                prev_s = survivor[t, best_s]
                dec_seq[t] = 0 if self.ns_0[prev_s] == best_s else 1
                best_s = prev_s

            decoded[b] = dec_seq[:msg_len]

            re_enc = self._re_encode(dec_seq)
            total_corrections += np.sum(re_enc != rx[b])

        return decoded, int(total_corrections)

    def decode_single(self, rx_coded):
        dec, corr = self.decode(rx_coded.reshape(1, -1))
        return dec.flatten(), corr

    def _re_encode(self, bits):
        state = 0
        result = np.zeros(len(bits) * 2, dtype=np.uint8)
        for i in range(len(bits)):
            inp = int(bits[i])
            reg = (state << 1) | inp
            result[2*i]     = bin(reg & self.g1).count('1') % 2
            result[2*i + 1] = bin(reg & self.g2).count('1') % 2
            state = reg & self.mask  # [FIXED]
        return result


_encoder = None
_decoder = None
_K = None
_memory = None


def _init(K, g1, g2):
    global _encoder, _decoder, _K, _memory
    _K = K
    _memory = K - 1
    _encoder = ConvolutionalEncoder(K, g1, g2)
    _decoder = ViterbiDecoder(K, g1, g2)


def init_from_preset(name='k7'):
    K, g1, g2, label = PRESETS[name]
    _init(K, g1, g2)
    print(f"[CONV] {label} (g1={oct(g1)}, g2={oct(g2)})")


def encode(msg_bits):
    if _encoder is None:
        init_from_preset()
    return _encoder.encode(msg_bits)


def decode(rx_coded):
    if _decoder is None:
        init_from_preset()
    return _decoder.decode(rx_coded)


def run_tests():
    print("=" * 65)
    print("PRUEBAS UNITARIAS: modulos conv_code.py")
    print("=" * 65)

    print("\n[1] Verificacion de trellis (no degenerada)...")
    for preset_name in ['k3', 'k5', 'k7']:
        K, g1, g2, label = PRESETS[preset_name]
        enc = ConvolutionalEncoder(K, g1, g2)
        n_diff = np.count_nonzero(
            enc.next_state != np.arange(enc.num_states)[:, None]
        )
        expected_min = enc.num_states 
        assert n_diff >= expected_min, \
            f"{label}: solo {n_diff} transiciones distintas del estado actual"
        print(f"  {label:22s} | {n_diff}/{2*enc.num_states} transiciones distintas -> OK")

    print("\n[2] Vector de prueba conocido (K=3, g=7,5)...")
    enc7_5 = ConvolutionalEncoder(3, 0o7, 0o5)
    dec7_5 = ViterbiDecoder(3, 0o7, 0o5)

    msg = np.array([1, 1, 0, 1, 0], dtype=np.uint8)
    coded = enc7_5.encode_single(msg)

    expected_without_tail = [
        1, 1,   
        0, 1,   
        0, 1,  
        0, 0,  
        1, 0,  
    ]
    assert np.array_equal(coded[:10], expected_without_tail), \
        f"Vector conocido incorrecto: {coded[:10]} != {expected_without_tail}"
    print(f"  K=3(7,5): output correcto -> OK")

    print("\n[3] Round-trip encode/decode (sin errores)...")
    np.random.seed(123)
    for preset_name in ['k3', 'k5', 'k7']:
        K, g1, g2, label = PRESETS[preset_name]
        enc = ConvolutionalEncoder(K, g1, g2)
        dec = ViterbiDecoder(K, g1, g2)
        for msg_len in [5, 10, 20, 50, 100, 200]:
            msg = np.random.randint(0, 2, msg_len).astype(np.uint8)
            coded = enc.encode_single(msg)
            decoded, _ = dec.decode_single(coded)
            assert np.array_equal(decoded, msg), \
                f"{label}: round-trip FAIL at len={msg_len}"
    print("  Todos OK")

    print("\n[4] Correccion de error de un solo bit...")
    for preset_name in ['k3', 'k5', 'k7']:
        K, g1, g2, label = PRESETS[preset_name]
        enc = ConvolutionalEncoder(K, g1, g2)
        dec = ViterbiDecoder(K, g1, g2)
        n_trials = 100
        n_corrected = 0
        for _ in range(n_trials):
            msg = np.random.randint(0, 2, 40).astype(np.uint8)
            coded = enc.encode_single(msg)
            err_pos = np.random.randint(0, len(coded))
            coded_err = coded.copy()
            coded_err[err_pos] ^= 1
            decoded, _ = dec.decode_single(coded_err)
            if np.array_equal(decoded, msg):
                n_corrected += 1
        pct = n_corrected / n_trials * 100
        print(f"  {label:22s} | {n_corrected}/{n_trials} ({pct:.0f}%)")
        assert pct >= 90, f"{label}: solo {pct:.0f}% (esperado >= 90%)"

    print("\n[5] Procesamiento por lotes (batch)...")
    for preset_name in ['k3', 'k5', 'k7']:
        K, g1, g2, label = PRESETS[preset_name]
        enc = ConvolutionalEncoder(K, g1, g2)
        dec = ViterbiDecoder(K, g1, g2)
        msgs = np.random.randint(0, 2, (50, 30)).astype(np.uint8)
        coded = enc.encode(msgs)
        decoded, _ = dec.decode(coded)
        assert np.array_equal(decoded, msgs), f"Batch {label} FAIL"
    print("  Todos OK")

    print("\n[6] Distinguibilidad del codigo...")
    enc = ConvolutionalEncoder(3, 0o7, 0o5)
    c1 = enc.encode_single(np.array([1,0,1,0,1], dtype=np.uint8))
    c2 = enc.encode_single(np.array([0,1,0,1,0], dtype=np.uint8))
    diff = np.count_nonzero(c1 != c2)
    print(f"  Diferentes mensajes -> {diff} bits distintos (esperado > 0) -> OK")
    assert diff > 0

    print("\n" + "=" * 65)
    print("[OK] TODAS las pruebas unitarias pasadas.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()