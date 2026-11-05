#!/bin/sh
# Regenerates the test fixtures in crates/uemulator-crypto/src/testdata with openssl and
# python3: a throwaway CA and leaf, an RSA-1024 signature, detached CMS signatures like an
# authenticated variable's, and a small PE32+ image signed with hand-built Authenticode (no
# sbsign or osslsigncode needed). The private keys are thrown away; only public material and
# signatures are kept. Every run makes new keys, so all fixtures change together.
set -eu

cd "$(dirname "$0")/../.."
out=crates/uemulator-crypto/src/testdata
mkdir -p "$out"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 36500 \
    -subj "/O=uemulator tests/CN=uemulator Test CA" \
    -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,keyCertSign \
    -keyout "$work/ca.key" -out "$work/ca.pem" 2>/dev/null
openssl req -newkey rsa:2048 -nodes -sha256 -subj "/O=uemulator tests/CN=uemulator Test Signer" \
    -keyout "$work/leaf.key" -out "$work/leaf.csr" 2>/dev/null
openssl x509 -req -in "$work/leaf.csr" -CA "$work/ca.pem" -CAkey "$work/ca.key" \
    -set_serial 0x00a1b2c3d4 -days 36500 -sha256 -out "$work/leaf.pem" 2>/dev/null
openssl x509 -in "$work/ca.pem" -outform der -out "$out/test-ca.der"
openssl x509 -in "$work/leaf.pem" -outform der -out "$out/test-leaf.der"
# The TBSCertificate starts 4 bytes in (SEQUENCE with a two-byte long length).
openssl asn1parse -inform der -in "$out/test-ca.der" -strparse 4 -noout -out "$work/tbs"
openssl dgst -sha256 -binary "$work/tbs" > "$out/test-ca-tbs.sha256"

openssl genrsa -out "$work/rsa1024.key" 1024 2>/dev/null
openssl rsa -in "$work/rsa1024.key" -pubout -outform der -out "$out/rsa1024-public.der" 2>/dev/null
printf uemulator > "$work/message"
openssl dgst -sha256 -sign "$work/rsa1024.key" -out "$out/rsa1024-sha256.sig" "$work/message"

printf 'uemulator authenticated variable' > "$work/variable"
openssl cms -sign -binary -outform der -md sha256 -signer "$work/leaf.pem" -inkey "$work/leaf.key" \
    -in "$work/variable" -out "$out/detached-attributes.p7"
openssl cms -sign -binary -noattr -outform der -md sha256 -signer "$work/leaf.pem" \
    -inkey "$work/leaf.key" -in "$work/variable" -out "$out/detached-noattributes.p7"

python3 -I - "$work" "$out" <<'EOF'
import hashlib, struct, subprocess, sys

work, out = sys.argv[1], sys.argv[2]

def tlv(tag, body):
    n = len(body)
    if n < 0x80:
        length = bytes([n])
    else:
        raw = n.to_bytes((n.bit_length() + 7) // 8, 'big')
        length = bytes([0x80 | len(raw)]) + raw
    return bytes([tag]) + length + body

def read_tlv(data, at):
    tag, first = data[at], data[at + 1]
    if first < 0x80:
        return tag, at + 2, first
    count = first & 0x7f
    return tag, at + 2 + count, int.from_bytes(data[at + 2:at + 2 + count], 'big')

def children(data, start, length):
    at, items = start, []
    while at < start + length:
        tag, body, size = read_tlv(data, at)
        items.append((tag, at, body, size))
        at = body + size
    return items

def oid(text):
    parts = [int(p) for p in text.split('.')]
    body = bytes([40 * parts[0] + parts[1]])
    for value in parts[2:]:
        chunk = [value & 0x7f]
        value >>= 7
        while value:
            chunk.append(0x80 | (value & 0x7f))
            value >>= 7
        body += bytes(reversed(chunk))
    return tlv(0x06, body)

SHA256 = oid('2.16.840.1.101.3.4.2.1')
NULL = b'\x05\x00'

# A PE32+ EFI application: two sections and a little data after them.
FILE_ALIGNMENT = 0x200
SECTION_ALIGNMENT = 0x1000
text = bytes((i * 7) & 0xff for i in range(0x300))
data = b'uemulator test image data'.ljust(0x200, b'\0')
headers_size = 0x400
dos = bytearray(0x80)
dos[0:2] = b'MZ'
struct.pack_into('<I', dos, 0x3c, 0x80)
coff = struct.pack('<HHIIIHH', 0x8664, 2, 0, 0, 0, 240, 0x22)
optional = bytearray(240)
struct.pack_into('<HBB', optional, 0, 0x20b, 1, 0)
struct.pack_into('<III', optional, 4, len(text), len(data), 0)
struct.pack_into('<II', optional, 16, 0x1000, 0x1000)
struct.pack_into('<Q', optional, 24, 0x10000000)
struct.pack_into('<II', optional, 32, SECTION_ALIGNMENT, FILE_ALIGNMENT)
struct.pack_into('<II', optional, 56, 0x3000, headers_size)
struct.pack_into('<I', optional, 64, 0x1234)  # CheckSum, left out of the digest
struct.pack_into('<HH', optional, 68, 10, 0)  # EFI application
struct.pack_into('<I', optional, 108, 16)
def section(name, virtual, raw_size, raw_offset, flags):
    return struct.pack('<8sIIIIIIHHI', name, raw_size, virtual, raw_size, raw_offset, 0, 0, 0, 0, flags)
sections = section(b'.text', 0x1000, len(text), 0x400, 0x60000020) + \
    section(b'.data', 0x2000, len(data), 0x400 + len(text), 0xc0000040)
header = bytes(dos) + b'PE\0\0' + coff + bytes(optional) + sections
image = bytearray(header.ljust(headers_size, b'\0') + text + data + b'trailing!')
cert_entry = 0x80 + 4 + 20 + 112 + 4 * 8

def authenticode(image, certificate_size):
    h = hashlib.sha256()
    checksum = 0x80 + 4 + 20 + 64
    h.update(image[:checksum])
    h.update(image[checksum + 4:cert_entry])
    h.update(image[cert_entry + 8:headers_size])
    hashed = headers_size
    for offset, size in [(0x400, len(text)), (0x400 + len(text), len(data))]:
        h.update(image[offset:offset + size])
        hashed += size
    h.update(image[hashed:len(image) - certificate_size])
    return h.digest()

# Pad to 8 so the certificate table starts aligned.
while len(image) % 8:
    image += b'\0'
digest = authenticode(bytes(image), 0)
open(f'{out}/unsigned.efi', 'wb').write(image)
open(f'{out}/image.sha256', 'wb').write(digest)

# SpcIndirectDataContent { SpcAttributeTypeAndOptionalValue { SPC_PE_IMAGE_DATAOBJ,
# SpcPeImageData { flags, file: <<<Obsolete>>> } }, DigestInfo }.
obsolete = '<<<Obsolete>>>'.encode('utf-16-be')
pe_image_data = tlv(0x30, b'\x03\x01\x00' + tlv(0xa0, tlv(0xa2, tlv(0x80, obsolete))))
attribute = tlv(0x30, oid('1.3.6.1.4.1.311.2.1.15') + pe_image_data)
digest_info = tlv(0x30, tlv(0x30, SHA256 + NULL) + tlv(0x04, digest))
indirect_body = attribute + digest_info
indirect = tlv(0x30, indirect_body)
SPC_INDIRECT = oid('1.3.6.1.4.1.311.2.1.4')

attributes = tlv(0x30, oid('1.2.840.113549.1.9.3') + tlv(0x31, SPC_INDIRECT)) + \
    tlv(0x30, oid('1.2.840.113549.1.9.4') + tlv(0x31, tlv(0x04, hashlib.sha256(indirect_body).digest())))
open(f'{work}/attributes', 'wb').write(tlv(0x31, attributes))
signature = subprocess.run(['openssl', 'dgst', '-sha256', '-sign', f'{work}/leaf.key',
                            f'{work}/attributes'], check=True, capture_output=True).stdout

leaf = open(f'{out}/test-leaf.der', 'rb').read()
ca = open(f'{out}/test-ca.der', 'rb').read()
_, body, size = read_tlv(leaf, 0)
_, tbs_body, tbs_size = read_tlv(leaf, body)
fields = children(leaf, tbs_body, tbs_size)
# TBSCertificate: [0] version, serial, signature, issuer, ...
serial = leaf[fields[1][1]:fields[1][2] + fields[1][3]]
issuer = leaf[fields[3][1]:fields[3][2] + fields[3][3]]

signer_info = tlv(0x30, b'\x02\x01\x01' + tlv(0x30, issuer + serial) + tlv(0x30, SHA256 + NULL)
                  + tlv(0xa0, attributes) + tlv(0x30, oid('1.2.840.113549.1.1.1') + NULL)
                  + tlv(0x04, signature))
signed_data = tlv(0x30, b'\x02\x01\x01' + tlv(0x31, tlv(0x30, SHA256 + NULL))
                  + tlv(0x30, SPC_INDIRECT + tlv(0xa0, indirect))
                  + tlv(0xa0, leaf + ca) + tlv(0x31, signer_info))
content_info = tlv(0x30, oid('1.2.840.113549.1.7.2') + tlv(0xa0, signed_data))

certificate = content_info
while (len(certificate) + 8) % 8:
    certificate += b'\0'
entry = struct.pack('<IHH', len(certificate) + 8, 0x0200, 0x0002) + certificate
struct.pack_into('<II', image, cert_entry, len(image), len(entry))
signed = bytes(image) + entry
assert authenticode(signed, len(entry)) == digest
open(f'{out}/signed.efi', 'wb').write(signed)
EOF
ls -l "$out"
