#!/usr/bin/env python3
"""Build the fixed-384 centroid matrix/vector kernel with only Python's stdlib.

Each SIMD lane belongs to a different centroid. Separate Float64 multiply and
add instructions preserve the scalar dot product's dimension order exactly.
The caller adds its existing Float64 L2 bias after this kernel returns.

Matrix layout: [row block][384 dimensions][block_size Float32 rows]. Count must
include the caller's zero padding to a whole block. Query/output are Float64.
Only stable WebAssembly SIMD instructions are used; no compiler is required.
"""

import argparse
from pathlib import Path


DEFAULT_BLOCK = 32
DEFAULT_UNROLL = 2
DIMENSIONS = 384

# Binary instruction mnemonics from the WebAssembly core/SIMD specifications.
I32_ADD = b'\x6a'
I32_SHL = b'\x74'
I32_LT_U = b'\x49'
I32_GE_U = b'\x4f'
F64_LOAD = b'\x2b'
BLOCK = b'\x02\x40'
LOOP = b'\x03\x40'
END = b'\x0b'
V128_CONST = 12
V128_STORE = 11
V128_LOAD64_ZERO = 93
F64X2_SPLAT = 20
F64X2_PROMOTE_LOW_F32X4 = 95
F64X2_MUL = 242
F64X2_ADD = 240


def unsigned(value):
    """Unsigned LEB128 for sizes, indices and SIMD opcodes."""
    result = bytearray()
    while True:
        byte = value & 127
        value >>= 7
        result.append(byte | (128 if value else 0))
        if not value:
            return bytes(result)


def integer(value):
    """i32.const with signed LEB128."""
    result = bytearray(b'\x41')
    while True:
        byte = value & 127
        value >>= 7
        finished = (value == 0 and not byte & 64) or (value == -1 and byte & 64)
        result.append(byte | (0 if finished else 128))
        if finished:
            return bytes(result)


def vector(parts):
    return unsigned(len(parts)) + b''.join(parts)


def section(identifier, payload):
    return bytes([identifier]) + unsigned(len(payload)) + payload


def name(value):
    payload = value.encode('utf-8')
    return unsigned(len(payload)) + payload


def get(local):
    return b'\x20' + unsigned(local)


def put(local):
    return b'\x21' + unsigned(local)


def simd(opcode):
    return b'\xfd' + unsigned(opcode)


def build_module(block_size=DEFAULT_BLOCK, unroll=DEFAULT_UNROLL):
    if block_size not in (8, 16, 24, 32) or unroll not in (1, 2):
        raise ValueError('Supported blocks are 8/16/24/32 and unroll factors 1/2')

    # Parameters: count, matrix pointer, query pointer, output pointer (all i32).
    function_type = b'\x60' + vector([b'\x7f'] * 4) + vector([])
    module = b'\0asm\x01\0\0\0'
    module += section(1, vector([function_type]))
    module += section(3, vector([unsigned(0)]))
    module += section(5, vector([b'\0' + unsigned(1)]))  # One growable memory.
    module += section(6, vector([b'\x7f\0' + integer(block_size) + END]))
    module += section(7, vector([
        name('memory') + b'\x02' + unsigned(0),
        name('matrix384') + b'\0' + unsigned(0),
        name('block_size') + b'\x03' + unsigned(0),
    ]))

    # i32 locals: row=4, matrix_cursor=5, query_cursor=6, query_end=7.
    # v128 locals: one F64x2 accumulator per two rows, then the query splat.
    pairs = block_size // 2
    query_local = 8 + pairs
    body = vector([unsigned(4) + b'\x7f', unsigned(pairs + 1) + b'\x7b'])
    body += integer(0) + put(4) + get(1) + put(5)
    body += get(2) + integer(DIMENSIONS * 8) + I32_ADD + put(7)

    body += BLOCK + LOOP
    body += get(4) + get(0) + I32_GE_U + b'\x0d\x01'  # br_if 1: also handles count=0.
    for accumulator in range(8, query_local):
        body += simd(V128_CONST) + bytes(16) + put(accumulator)
    body += get(2) + put(6) + LOOP
    for _ in range(unroll):
        body += get(6) + F64_LOAD + unsigned(3) + unsigned(0)
        body += simd(F64X2_SPLAT) + put(query_local)
        for pair in range(pairs):
            # Load two adjacent Float32 rows and promote them exactly to F64.
            body += get(8 + pair) + get(5)
            body += simd(V128_LOAD64_ZERO) + unsigned(3) + unsigned(pair * 8)
            body += simd(F64X2_PROMOTE_LOW_F32X4) + get(query_local)
            # Explicit multiply then add: neither FMA nor a tree reduction.
            body += simd(F64X2_MUL) + simd(F64X2_ADD) + put(8 + pair)
        body += get(5) + integer(block_size * 4) + I32_ADD + put(5)
        body += get(6) + integer(8) + I32_ADD + put(6)
    body += get(6) + get(7) + I32_LT_U + b'\x0d\0' + END  # br_if 0.

    for pair in range(pairs):
        body += get(3) + get(4) + integer(3) + I32_SHL + I32_ADD
        body += get(8 + pair) + simd(V128_STORE) + unsigned(4) + unsigned(pair * 16)
    body += get(4) + integer(block_size) + I32_ADD + put(4)
    body += b'\x0c\0' + END + END + END  # Outer br 0; loop, block, function ends.
    return module + section(10, vector([unsigned(len(body)) + body]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=int, choices=(8, 16, 24, 32), default=DEFAULT_BLOCK)
    parser.add_argument('--unroll', type=int, choices=(1, 2), default=DEFAULT_UNROLL)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'centroid-simd.wasm')
    args = parser.parse_args()
    payload = build_module(args.block, args.unroll)
    args.output.write_bytes(payload)
    print(f'{args.output}: {len(payload)} bytes; block={args.block}, unroll={args.unroll}')
