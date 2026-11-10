"""Minimal Parquet reader for flat files with BYTE_ARRAY (string) columns. Stdlib only.

Supports: thrift-compact footer, codecs UNCOMPRESSED/SNAPPY/GZIP, data pages v1/v2, PLAIN and (RLE_)DICTIONARY encodings,
optional columns (definition level 0/1). Enough for Hugging Face dataset exports of text columns.
"""
import struct, zlib

class Thrift:
    def __init__(self, b, o=0):
        self.b, self.o = b, o
    def byte(self):
        v = self.b[self.o]; self.o += 1; return v
    def varint(self):
        r = s = 0
        while True:
            x = self.byte(); r |= (x & 0x7f) << s; s += 7
            if not x & 0x80:
                return r
    def zigzag(self):
        n = self.varint(); return (n >> 1) ^ -(n & 1)
    def value(self, t):
        if t in (1, 2): return t == 1                       # bool true/false (in struct field type)
        if t == 3: return struct.unpack_from("b", self.b, self._adv(1))[0]
        if t in (4, 5, 6): return self.zigzag()
        if t == 7: v = struct.unpack_from("<d", self.b, self.o)[0]; self.o += 8; return v
        if t == 8:
            n = self.varint(); v = self.b[self.o:self.o + n]; self.o += n; return v
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0f
            if n == 15: n = self.varint()
            return [self.value(et) if et not in (1, 2) else (self.byte() == 1) for _ in range(n)]
        if t == 11: raise ValueError("map not supported")
        if t == 12: return self.struct()
        raise ValueError(f"thrift type {t}")
    def _adv(self, n):
        o = self.o; self.o += n; return o
    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0: return out
            d, t = h >> 4, h & 0x0f
            fid = last + d if d else self.zigzag()
            last = fid
            out[fid] = self.value(t)

def snappy(src):
    o = 0; n = s = 0
    while True:
        x = src[o]; o += 1; n |= (x & 0x7f) << s; s += 7
        if not x & 0x80: break
    out = bytearray()
    while o < len(src):
        tag = src[o]; o += 1; kind = tag & 3
        if kind == 0:
            ln = tag >> 2
            if ln >= 60:
                k = ln - 59; ln = int.from_bytes(src[o:o + k], "little"); o += k
            ln += 1; out += src[o:o + ln]; o += ln
        else:
            if kind == 1: ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[o]; o += 1
            elif kind == 2: ln = (tag >> 2) + 1; off = int.from_bytes(src[o:o + 2], "little"); o += 2
            else: ln = (tag >> 2) + 1; off = int.from_bytes(src[o:o + 4], "little"); o += 4
            for _ in range(ln): out.append(out[-off])
    return bytes(out)

def decompress(codec, data, size):
    if codec == 0: return data
    if codec == 1: return snappy(data)
    if codec == 2: return zlib.decompress(data, 16 + zlib.MAX_WBITS)
    raise ValueError(f"codec {codec} not supported (6 = ZSTD)")

def rle_hybrid(b, o, end, bit_width, count):
    out = []
    bw = (bit_width + 7) // 8
    while len(out) < count and o < end:
        h = 0; s = 0
        while True:
            x = b[o]; o += 1; h |= (x & 0x7f) << s; s += 7
            if not x & 0x80: break
        if h & 1:                                   # bit-packed groups of 8
            groups = h >> 1; nbytes = groups * bit_width
            v = int.from_bytes(b[o:o + nbytes], "little"); o += nbytes
            mask = (1 << bit_width) - 1
            for i in range(groups * 8): out.append((v >> (i * bit_width)) & mask)
        else:
            ln = h >> 1; val = int.from_bytes(b[o:o + bw], "little"); o += bw
            out += [val] * ln
    return out[:count]

def plain_strings(b, o, end, n):
    out = []
    while len(out) < n and o + 4 <= end:
        ln = struct.unpack_from("<I", b, o)[0]; o += 4
        out.append(b[o:o + ln].decode("utf-8", "replace")); o += ln
    return out

def read_columns(path, wanted):
    b = open(path, "rb").read()
    assert b[:4] == b"PAR1" and b[-4:] == b"PAR1"
    flen = struct.unpack_from("<I", b, len(b) - 8)[0]
    meta = Thrift(b, len(b) - 8 - flen).struct()
    schema = meta[2]
    # flat schema: element 0 is the root; leaf names with repetition (1 = optional)
    leaves = [(s[4].decode(), s.get(3, 0)) for s in schema[1:]]
    cols = {w: [] for w in wanted}
    for rg in meta[4]:
        for ci, cc in enumerate(rg[1]):
            md = cc[3]
            name = b".".join(md[3]).decode()
            if name not in cols: continue
            optional = dict(leaves).get(name, 0) == 1
            codec = md[4]
            o = md.get(11, md[9])                   # dictionary_page_offset if present, else data_page_offset
            end = md[9] + md[7] if 11 not in md else md[11] + md[7]
            dictionary, values = [], []
            total = md[5]
            while len(values) < total and o < end:
                t = Thrift(b, o); ph = t.struct(); o = t.o
                ptype, usize, csize = ph[1], ph[2], ph[3]
                page = b[o:o + csize]; o += csize
                if ptype == 2:                      # dictionary page
                    raw = decompress(codec, page, usize)
                    dictionary = plain_strings(raw, 0, len(raw), ph[7][1])
                    continue
                if ptype == 0:                      # data page v1
                    raw = decompress(codec, page, usize); dh = ph[5]; n = dh[1]; enc = dh[2]; p = 0
                    defs = [1] * n
                    if optional:
                        ln = struct.unpack_from("<I", raw, p)[0]; p += 4
                        defs = rle_hybrid(raw, p, p + ln, 1, n); p += ln
                elif ptype == 3:                    # data page v2
                    dh = ph[8]; n = dh[1]; enc = dh[4]; dl = dh[5]; rl = dh[6]
                    levels = page[:rl + dl]
                    body = page[rl + dl:]
                    raw = levels + (decompress(codec, body, usize - rl - dl) if dh.get(7, True) else body)
                    defs = rle_hybrid(raw, rl, rl + dl, 1, n) if optional and dl else [1] * n
                    p = rl + dl
                else:
                    continue
                present = sum(defs)
                if enc in (2, 8):                   # PLAIN_DICTIONARY / RLE_DICTIONARY
                    bw = raw[p]; idx = rle_hybrid(raw, p + 1, len(raw), bw, present)
                    vals = [dictionary[i] for i in idx]
                else:
                    vals = plain_strings(raw, p, len(raw), present)
                it = iter(vals)
                values += [next(it) if d else "" for d in defs]
            cols[name] += values
    return cols

if __name__ == "__main__":
    import sys
    c = read_columns(sys.argv[1], sys.argv[2:])
    for k, v in c.items():
        print(k, len(v), repr(v[0][:120]) if v else "")
