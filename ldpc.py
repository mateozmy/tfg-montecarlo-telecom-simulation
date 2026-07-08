"""
LDPC (Low-Density Parity-Check) encoder/decoder using pyldpc.

Encoding:  systematic,  G @ msg  (mod 2);  G is sparse (n, k).
Decoding:  log-domain belief propagation (sum-product algorithm).
"""

import numpy as np
from pyldpc import make_ldpc, decode as ldpc_decode, get_message


class LDPCCodec:

    def __init__(self, n, d_v, d_c, seed=40, sparse=True):
        """
        n : int
            Target codeword length.
        d_v : int
            Variable-node degree (column weight), typically 3.
        d_c : int
            Check-node degree (row weight), must divide n, > d_v.
        seed : int
            RNG seed for reproducible H.
        sparse : bool
            Use scipy sparse matrices (recommended for speed).
        """
        self.H, self.G = make_ldpc(n, d_v, d_c, systematic=True,
                                    sparse=sparse, seed=seed)
        self.n = int(self.G.shape[0])   
        self.k = int(self.G.shape[1])   
        self.rate = self.k / self.n

    def encode(self, msg):
   
        msg = np.atleast_1d(np.asarray(msg, dtype=np.uint8))
        if msg.ndim == 1:
            return np.mod(self.G @ msg, 2).astype(np.uint8)

        return np.mod(self.G @ msg.T, 2).T.astype(np.uint8)

    def decode(self, rx, noise_sigma, max_iter=100):
        """
        rx : ndarray, float, shape (n,) or (batch, n)
            Soft received samples (channel convention: 0->-1, 1->+1).
        noise_sigma : float
            AWGN noise standard deviation sigma.
        max_iter : int
            Max BP iterations.
        """
        rx = np.atleast_1d(np.asarray(rx, dtype=np.float64))
        snr = 1.0 / (noise_sigma ** 2)  
        y_ldpc = -rx                     

        if rx.ndim == 1:
            dec_cw = ldpc_decode(self.H, y_ldpc, snr, maxiter=max_iter)
            return get_message(self.G, dec_cw).astype(np.uint8)

        decoded = []
        for row in y_ldpc:
            dec_cw = ldpc_decode(self.H, row, snr, maxiter=max_iter)
            decoded.append(get_message(self.G, dec_cw))
        return np.array(decoded, dtype=np.uint8)

    def __repr__(self):
        return (f"LDPCCodec(n={self.n}, k={self.k}, rate={self.rate:.3f})")



def make_ldpc_r12_n300():
    """Regular (3,6) LDPC, n=300, rate=1/2  (fast, good for simulation)."""
    return LDPCCodec(300, 3, 6, seed=40)


def make_ldpc_r12_n600():
    """Regular (3,6) LDPC, n=600, rate=1/2  (better perf, slower)."""
    return LDPCCodec(600, 3, 6, seed=40)