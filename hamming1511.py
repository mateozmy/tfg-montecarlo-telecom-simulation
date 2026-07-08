# -*- coding: utf-8 -*-
"""
Codificador y decodificador Hamming (15,11)

Hamming (15,11): codigo de bloque lineal sistematico
  - k = 11 bits de datos  ->  n = 15 bits codificados
  - Capacidad: corregir 1 error por palabra-codigo
  - Tasa: R = 11/15 ~= 0.7333
  - 4 bits de paridad (n - k = 4)

Matriz generadora G (forma sistematica [I11 | P]):
  Los primeros 11 bits son los datos, los ultimos 4 son paridad.

Matriz de verificacion H (4x15):
  Cada columna es la representacion binaria de su posicion (1..15),
  ordenada para que las ultimas 4 columnas formen I4.
"""

import numpy as np


G = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,   0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,   1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,   1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,   1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   1, 1, 1, 1],
], dtype=np.uint8)

H = np.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,   1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,   0, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,   0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,   0, 0, 0, 1],
], dtype=np.uint8)


SYNDROME_TABLE = {
    0:  None,   
    1:  11,     
    2:  12,     
    3:  7,      
    4:  13,     
    5:  5,      
    6:  2,      
    7:  9,      
    8:  14,     
    9:  4,     
    10: 1,     
    11: 8,     
    12: 0,     
    13: 6,     
    14: 3,      
    15: 10,     
}


def encode(msg_bits):
    msg_bits = np.asarray(msg_bits, dtype=np.uint8)
    return (msg_bits @ G) % 2


def decode(rx_bits):
    rx_bits = np.asarray(rx_bits, dtype=np.uint8)

    s_bits = (rx_bits @ H.T) % 2 
    powers = np.array([1, 2, 4, 8], dtype=np.int32)
    s_int = np.sum(s_bits.astype(np.int32) * powers, axis=-1)  

    corrected = rx_bits.copy()
    n_corrected = 0

    shape = s_int.shape
    s_flat = s_int.ravel()
    corrected_flat = corrected.reshape(-1, 15)

    for idx, syn in enumerate(s_flat):
        pos = SYNDROME_TABLE.get(syn)
        if pos is not None:
            corrected_flat[idx, pos] ^= 1  
            n_corrected += 1

    decoded = corrected[..., :11]

    return decoded, n_corrected


if __name__ == "__main__":
    print("=" * 60)
    print("Prueba del modulo Hamming (15,11)")
    print("=" * 60)

    msg = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    codeword = encode(msg)
    codeword_manual = (msg @ G) % 2
    print(f"\nMensaje original:    {msg}")
    print(f"Palabra-codigo:      {codeword}")
    print(f"msg @ G (mod 2):     {codeword_manual}  <-- debe coincidir")

    syndrome_check = (H @ codeword) % 2
    assert np.all(syndrome_check == 0), "H * c^T NO es cero!"

    decoded, corr = decode(codeword)
    assert np.array_equal(decoded, msg), "Decodificacion sin errores fallo!"
    print(f"\nSin errores -> decodificado: {decoded} (correcciones: {corr})")

    print("\n--- Correccion de errores en cada bit ---")
    all_ok = True
    for bit in range(15):
        rx = codeword.copy()
        rx[bit] ^= 1  
        decoded, corr = decode(rx)
        ok = np.array_equal(decoded, msg) and corr == 1
        status = "[OK]" if ok else "[FAIL]"
        if not ok:
            all_ok = False
        print(f"  Error en bit {bit+1:2d}: sindrome calculado, corregido -> {status}")

    print("\n--- Dos errores (Hamming no puede corregir 2) ---")
    rx2 = codeword.copy()
    rx2[0] ^= 1
    rx2[7] ^= 1
    decoded2, corr2 = decode(rx2)
    ok2 = np.array_equal(decoded2, msg)
    status2 = "[OK]" if ok2 else "[FAIL] (esperado)"
    print(f"  Errores en bits 1 y 8: {status2}")
    if not ok2:
        print(f"  Original: {msg}, Decodificado: {decoded2}")

    if all_ok:
        print(f"\n[OK] Todas las pruebas completadas -- 15/15 bits OK")
    else:
        print(f"\n[FAIL] Algunas pruebas fallaron")