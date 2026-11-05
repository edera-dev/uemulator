#!/usr/bin/env python3
"""Test vectors for uemulator-crypto's TPM primitives, from Python's hmac/hashlib and the
`cryptography` package (pip install cryptography).

Writes the randomized fixtures into crates/uemulator-crypto/src/testdata (a fresh RSA-2048 key
and an OAEP ciphertext under it, an ECDSA signature with a fixed P-256 key; every run changes
them) and prints the deterministic vectors (HMAC, KDFa, KDFe, AES-CFB, P-256 points, ECDH),
which the tests in hmac.rs, kdf.rs, aes.rs and ecc.rs carry as hex. Run it with `python3 -I`.
"""

import hashlib
import hmac
import pathlib
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

try:
    from cryptography.hazmat.decrepit.ciphers.modes import CFB
except ImportError:
    CFB = modes.CFB

OUT = pathlib.Path(__file__).resolve().parents[2] / 'crates/uemulator-crypto/src/testdata'


def print_vector(name, value):
    print(f'{name} = {value.hex()}')


def mask_top(data, bits):
    if bits % 8:
        data = bytes([data[0] & ((1 << (bits % 8)) - 1)]) + data[1:]
    return data


def kdfa(algorithm, key, label, context_u, context_v, bits):
    """TPM 2.0 Part 1 11.4.10.2: SP800-108 counter mode with HMAC."""
    out, counter = b'', 1
    while len(out) * 8 < bits:
        message = counter.to_bytes(4, 'big') + label + b'\0' + context_u + context_v \
            + bits.to_bytes(4, 'big')
        out += hmac.new(key, message, algorithm).digest()
        counter += 1
    return mask_top(out[:(bits + 7) // 8], bits)


def kdfe(algorithm, z, label, party_u, party_v, bits):
    """TPM 2.0 Part 1 11.4.10.3: SP800-56A concatenation KDF."""
    out, counter = b'', 1
    while len(out) * 8 < bits:
        out += hashlib.new(algorithm, counter.to_bytes(4, 'big') + z + label + b'\0'
                           + party_u + party_v).digest()
        counter += 1
    return mask_top(out[:(bits + 7) // 8], bits)


def point(public_key):
    numbers = public_key.public_numbers()
    return numbers.x.to_bytes(32, 'big') + numbers.y.to_bytes(32, 'big')


def main():
    print('# HMAC: RFC 4231 cases 1, 2, 6, 7 and RFC 2202 cases 1, 2, 6')
    rfc4231 = [
        (b'\x0b' * 20, b'Hi There'),
        (b'Jefe', b'what do ya want for nothing?'),
        (b'\xaa' * 131, b'Test Using Larger Than Block-Size Key - Hash Key First'),
        (b'\xaa' * 131, b'This is a test using a larger than block-size key and a larger than '
         b'block-size data. The key needs to be hashed before being used by the HMAC '
         b'algorithm.'),
    ]
    for number, (key, data) in zip([1, 2, 6, 7], rfc4231):
        for algorithm in ['sha256', 'sha384', 'sha512']:
            print_vector(f'rfc4231 case {number} {algorithm}', hmac.new(key, data, algorithm).digest())
    rfc2202 = [
        (b'\x0b' * 20, b'Hi There'),
        (b'Jefe', b'what do ya want for nothing?'),
        (b'\xaa' * 80, b'Test Using Larger Than Block-Size Key - Hash Key First'),
    ]
    for number, (key, data) in zip([1, 2, 6], rfc2202):
        print_vector(f'rfc2202 case {number} sha1', hmac.new(key, data, 'sha1').digest())
    print_vector('empty key sha256', hmac.new(b'', b'', 'sha256').digest())

    print('# KDFa')
    print_vector('sha256 key "session key" ATH nonce-caller nonce-tpm 256',
                 kdfa('sha256', b'session key', b'ATH', b'nonce-caller', b'nonce-tpm', 256))
    print_vector('sha1 key "seed" CFB "name" "" 300',
                 kdfa('sha1', b'seed', b'CFB', b'name', b'', 300))
    print_vector('sha384 key "" STORAGE "" "" 131',
                 kdfa('sha384', b'', b'STORAGE', b'', b'', 131))
    print('# KDFe')
    z = bytes(range(32))
    print_vector('sha256 z 00..1f SECRET "party u" "party v" 256',
                 kdfe('sha256', z, b'SECRET', b'party u', b'party v', 256))
    print_vector('sha512 z 00..1f DUPLICATE "" "v" 777',
                 kdfe('sha512', z, b'DUPLICATE', b'', b'v', 777))

    print('# AES-CFB128 with partial last blocks')
    data = bytes((index * 7 + 3) & 0xff for index in range(37))
    for key in [bytes.fromhex('2b7e151628aed2a6abf7158809cf4f3c'), bytes(range(32))]:
        iv = bytes(range(0xf0, 0x100))
        encryptor = Cipher(algorithms.AES(key), CFB(iv)).encryptor()
        print_vector(f'key {key.hex()} iv f0..ff data (7i+3)*37',
                     encryptor.update(data) + encryptor.finalize())
    # SP800-38A F.3.13 and FIPS-197 C, checked here so a typo in the tests shows up.
    key = bytes.fromhex('2b7e151628aed2a6abf7158809cf4f3c')
    encryptor = Cipher(algorithms.AES(key), CFB(bytes(range(16)))).encryptor()
    print_vector('sp800-38a f.3.13', encryptor.update(bytes.fromhex(
        '6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51'
        '30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710')))
    for size in [16, 24, 32]:
        encryptor = Cipher(algorithms.AES(bytes(range(size))), modes.ECB()).encryptor()
        print_vector(f'fips-197 aes-{size * 8}',
                     encryptor.update(bytes.fromhex('00112233445566778899aabbccddeeff')))

    print('# P-256')
    curve = ec.SECP256R1()
    order = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
    for scalar in [1, 2, order - 1]:
        print_vector(f'{scalar:x} * G', point(ec.derive_private_key(scalar, curve).public_key()))
    # NIST CAVS 14.1 ECC CDH Primitive, P-256 COUNT = 0.
    private = ec.derive_private_key(
        0x7d7dc5f71eb29ddaf80d6214632eeae03d9058af1fb6d22ed80badb62bc1a534, curve)
    peer = ec.EllipticCurvePublicNumbers(
        0x700c48f77f56584c5cc632ca65640db91b6bacce3a4df6b42ce7cc838833d287,
        0xdb71e509e3fd9b060ddb20ba5c51dcc5948d46fbf640dfe0441782cab85fa4ac, curve).public_key()
    print_vector('cavs count 0 QIUT', point(private.public_key()))
    print_vector('cavs count 0 ZIUT', private.exchange(ec.ECDH(), peer))

    signer = ec.derive_private_key(
        0xc9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721, curve)
    print_vector('ecdsa key public', point(signer.public_key()))
    der = signer.sign(b'uemulator', ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    (OUT / 'p256-ecdsa-sha256.sig').write_bytes(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = key.private_numbers()
    (OUT / 'rsa2048-modulus.bin').write_bytes(numbers.public_numbers.n.to_bytes(256, 'big'))
    (OUT / 'rsa2048-prime.bin').write_bytes(numbers.p.to_bytes(128, 'big'))
    ciphertext = key.public_key().encrypt(b'uemulator salt', padding.OAEP(
        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=b'SECRET\0'))
    (OUT / 'rsa2048-oaep-sha256.bin').write_bytes(ciphertext)
    print(f'# wrote the RSA and ECDSA fixtures to {OUT}', file=sys.stderr)


if __name__ == '__main__':
    main()
