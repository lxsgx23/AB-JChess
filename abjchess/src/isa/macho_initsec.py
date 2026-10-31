#!/usr/bin/env python3
"""macho_initsec.py OBJ NEWNAME -- for the macOS universal build (build_macos.sh): in a 64-bit Mach-O relocatable
object (one tier, after 'ld -r'), turn the static-initializer section __mod_init_func (S_MOD_INIT_FUNC_POINTERS) into
a plain data section NEWNAME (S_REGULAR, relocations kept), so dyld does not run this tier's initializers at load;
the dispatcher (isa/dispatch_macos.cpp) runs the selected tier's ones itself (getsectiondata). Refuses objects with
termination-function sections or init-offset sections (nothing else may run implicitly). Prints the count."""
import struct, sys
path, newname = sys.argv[1], sys.argv[2].encode()
assert len(newname) <= 16
b = bytearray(open(path, 'rb').read())
magic, cputype, cpusub, ftype, ncmds, sizeofcmds, flags, _res = struct.unpack_from('<IiiIIIII', b, 0)
if magic != 0xfeedfacf or ftype != 1: sys.exit(f'{path}: not a 64-bit Mach-O object (magic {magic:#x} type {ftype})')
off, n = 32, 0
for _ in range(ncmds):
    cmd, cmdsize = struct.unpack_from('<II', b, off)
    if cmd == 0x19:  # LC_SEGMENT_64
        nsects = struct.unpack_from('<I', b, off + 64)[0]
        s = off + 72
        for _ in range(nsects):
            sect = bytes(b[s:s + 16]).rstrip(b'\0'); seg = bytes(b[s + 16:s + 32]).rstrip(b'\0')
            fl = struct.unpack_from('<I', b, s + 64)[0]
            typ = fl & 0xff
            if typ in (0x0a, 0x16):  # S_MOD_TERM_FUNC_POINTERS, S_INIT_FUNC_OFFSETS
                sys.exit(f'{path}: unexpected section {seg.decode()},{sect.decode()} type {typ:#x}')
            if typ == 0x09:  # S_MOD_INIT_FUNC_POINTERS
                if n: sys.exit(f'{path}: more than one initializer section')
                b[s:s + 16] = newname.ljust(16, b'\0')
                struct.pack_into('<I', b, s + 64, fl & ~0xff)  # S_REGULAR, attributes kept
                n += 1
                cnt = struct.unpack_from('<Q', b, s + 40)[0] // 8
                print(f'{path}: {seg.decode()},{sect.decode()} -> {seg.decode()},{newname.decode()} ({cnt} initializers)')
            s += 80
    off += cmdsize
if not n: print(f'{path}: no initializer section')
open(path, 'wb').write(b)
