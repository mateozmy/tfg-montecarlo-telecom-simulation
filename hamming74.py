# -*- coding: utf-8 -*-
"""
Codificador y decodificador Hamming (7,4)

Hamming (7,4): código de bloque lineal sistemático
  - k = 4 bits de datos  →  n = 7 bits codificados
  - Capacidad: corregir 1 error por palabra-código
  - Tasa: R = 4/7 ≈ 0.5714

Matriz generadora G (forma sistemática [I₄ | P]):
  Los primeros 4 bits son los datos, los últimos 3 son paridad.

Matriz de verificación H:
  Cada columna de H es la representación binaria del índice (1..7),
  ordenadas para que el síndrome indique directamente el bit erróneo.
"""

import numpy as np


G = np.array([
    [1, 0, 0, 0, 1, 1, 0],   
    [0, 1, 0, 0, 1, 0, 1],   
    [0, 0, 1, 0, 1, 1, 1],  
    [0, 0, 0, 1, 0, 1, 1],   
], dtype=np.uint8)


H = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)


SYNDROME_TABLE = {
    0:  None,   
    1:  6,      
    2:  5,     
    3:  3,      
    4:  4,      
    5:  1,      
    6:  0,     
    7:  2,     
}


def encode(message_bits):
    message_bits = np.asarray(message_bits, dtype=np.uint8)
    return (message_bits @ G) % 2


def decode(received_bits):
    received_bits = np.asarray(received_bits, dtype=np.uint8)

    if received_bits.ndim == 1:
        received_bits = received_bits.reshape(1, -1)

    n_blocks = received_bits.shape[0]
    decoded = np.zeros((n_blocks, 4), dtype=np.uint8)
    n_corrected = 0

    for i in range(n_blocks):
        r = received_bits[i]

        s_bits = (r @ H.T) % 2

        s_val = int(s_bits[0] * 4 + s_bits[1] * 2 + s_bits[2])

        if s_val != 0:
            error_pos = SYNDROME_TABLE[s_val]
            r[error_pos] ^= 1   
            n_corrected += 1

        decoded[i] = r[:4]

    return decoded, n_corrected


def compute_syndrome(received_bits):
    received_bits = np.asarray(received_bits, dtype=np.uint8).flatten()
    s_bits = (received_bits @ H.T) % 2
    return int(s_bits[0] * 4 + s_bits[1] * 2 + s_bits[2])


if __name__ == "__main__":
    print("=" * 60)
    print("Prueba del módulo Hamming (7,4)")
    print("=" * 60)

    msg = np.array([1, 0, 1, 1], dtype=np.uint8)
    cw = encode(msg)
    print(f"\nMensaje original:     {msg}")
    print(f"Palabra-código:       {cw}")
    print(f"m @ G (mod 2):        {(msg @ G) % 2}  ← debe coincidir con la fila de arriba")

    dec, nc = decode(cw)
    print(f"\nSin errores -> decodificado: {dec.flatten()} (correcciones: {nc})")
    assert np.array_equal(dec.flatten(), msg), "ERROR: decodificacion fallo"

    print("\n--- Correccion de errores en cada bit ---")
    for pos in range(7):
        r_err = cw.copy()
        r_err[pos] ^= 1  
        dec, nc = decode(r_err)
        ok = "OK" if np.array_equal(dec.flatten(), msg) else "FAIL"
        print(f"  Error en bit {pos+1}: sindrome calculado, corregido -> {ok}")

    print("\n--- Dos errores (Hamming no puede corregir 2) ---")
    r_double = cw.copy()
    r_double[0] ^= 1
    r_double[5] ^= 1
    dec, nc = decode(r_double)
    ok = "OK" if np.array_equal(dec.flatten(), msg) else "FAIL (esperado)"
    print(f"  Errores en bits 1 y 6: {ok}")
    print(f"  Original: {msg}, Decodificado: {dec.flatten()}")

    print("\n[OK] Todas las pruebas completadas.")