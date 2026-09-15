# Reversetta

**Reverse It.**

Reversetta is the reverse Rosetta: it runs real macOS **arm64** binaries on **x86-64** Macs. It takes a genuine arm64 Mach-O executable, maps its segments into virtual memory, translates ARM64 code into native x86-64 basic blocks on the fly (via [AsmJit]), and executes it. Calls into libSystem (`printf` and friends) are intercepted and routed to host implementations.

```
$ ./build/reversetta test/fib_arm
=== Reversetta v0.4 ===
Fibonacci(40) = 102334155
======== end ========
```

## How fast is it?

On the recursive `fib(40)` benchmark (12th-gen Intel i7-12700):

| | Time | |
|---|---|---|
| Native x86-64 build | 0.48 s | baseline |
| **Reversetta v0.4** | **0.60 s** | **1.25x of native** |
| Reversetta v0.2 (naive dispatch) | 61.9 s | 130x of native |

v0.4 executes fib(40) — 1.3 billion block entries — with only **10 round-trips into the dispatcher**. The guest code runs as a self-contained web of directly linked native blocks.

## How it works

```
arm64 Mach-O ──> Loader ──> Guest window ──> Block JIT ──> x86-64 code ──> CPU
                    │              │              │
              segments,       guard pages    chaining, pinning,
              fixups, GOT     + faults       lazy flags, DMC
```

1. **Mach-O loader** — parses segments, the entry point, symbol tables and `LC_DYLD_CHAINED_FIXUPS`; binds imports to host thunks.
2. **Guest memory** — one flat 256 MiB window; `host_ptr = mem_bias + guest_addr`. The window is surrounded by 2 GiB `PROT_NONE` guard regions, so out-of-bounds guest accesses are caught by a `SIGSEGV`/`SIGBUS` handler — generated code carries **no inline bounds checks**.
3. **Block JIT** — translates one basic block at a time into position-independent x86-64.
4. **Dispatcher** — spins blocks, services "magic" host-function PCs and syscalls.

### The speed tricks

- **Block chaining** — `b`/`bl`/`b.cond` exits are emitted as `jmp rel32` slots that get patched to the target block body once it is compiled. Hot paths never return to the dispatcher.
- **Direct-mapped branch cache (DMC)** — indirect branches (`br`/`blr`/`ret`) resolve through a pc → body cache embedded in the generated code itself.
- **Global register pinning** — guest `x0`, `x8`, `x29`, `x30`, `sp` permanently live in host registers (`r10`, `r11`, `rbx`, `rbp`, `r14`) across chained blocks; they are synced with the CPU context only on dispatcher entry/exit.
- **Lazy flags** — `subs` followed by `b.cond`/`csel`/`ccmp` consumes x86 `RFLAGS` directly via `jcc`/`setcc`/`cmovcc`; NZCV round-trips through memory happen only when actually needed.
- **Peepholes** — memory-form ALU (`add qword[ctx], imm`), `lea` fusion for `add`, base-register caching for `ldur`/`str` series, `Wn` stores without redundant `mov r32, r32` (a 32-bit x86 op already zeroes the upper half).

## Building

Requirements: macOS, CMake ≥ 3.24, a clang that can cross-compile for arm64 (Xcode toolchain).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

AsmJit is vendored in `asmjit/` and built as a static library — no external dependencies.

## Usage

```bash
./build/reversetta [options] <arm64-binary> [program args...]
```

| Option | Effect |
|---|---|
| `-v` | show instruction translation |
| `-vv` | also trace blocks and host calls |
| `-vvv` | also dump generated x86-64 code bytes |
| `-h` | help |

Engine logs go to **stderr**, guest program output to **stdout**, framed by the banner and `======== end ========`. So `./build/reversetta helloworld_arm 2>/dev/null` prints exactly what the binary would print on a real ARM Mac.

Guest exit codes are propagated: the process exits with the guest's code (e.g. `139` for an out-of-bounds memory access, mirroring a segfault).

## Trying it

```bash
# cross-compile a test program for arm64
clang -arch arm64 -O2 -o test/fib_arm test/fib.c

# run it on your Intel Mac
./build/reversetta test/fib_arm

# golden-diff torture test: x64 output vs emulated arm64 output
clang -O2 -fno-vectorize -o /tmp/torture_x64 test/torture.c
/tmp/torture_x64 > /tmp/golden.txt
./build/reversetta test/torture_arm 2>/dev/null | sed '1d;$d' | diff /tmp/golden.txt -
```

`test/torture.c` exercises the whole integer ISA subset — ALU ops, shifts, division edge cases, `umulh`/`smulh`, bitfields, `csel`, `rbit`/`rev`, indirect calls, stack frames — at both `-O0` and `-O2`.

## What is supported

- Integer ARM64: data processing (immediate/register/extended), logical ops, bitfields, `extr`, conditional select/compare, multiplication/division (all edge cases incl. division by zero and `INT_MIN / -1`), `rbit`/`rev16`/`rev32`/`rev`/`clz`/`cls`
- Loads/stores: all addressing modes, `ldp`/`stp`, sign/zero extension, register offsets, literal loads
- Branches: `b`/`bl`/`b.cond`/`cbz`/`tbz`/`br`/`blr`/`ret`, plus `svc` (a minimal `exit`/`write` handler)
- C library bridge: `printf`-family (including fortified `_chk` variants), `puts`/`fwrite`, `malloc`/`calloc`/`realloc`/`free`, `str*`/`mem*`, `exit`/`abort`, `__stack_chk_fail`, `__assert_rtn`
- Fat binaries (the arm64 slice is picked automatically)

**Not supported (yet):** SIMD/FP instructions, C++ iostream binaries, threads, exceptions, dynamic libraries (`dyld`).

## Project layout

```
src/
  common.h          — layout constants, GuestCpu state, bit helpers, DecodeBitMasks
  mach-o.h          — packed Mach-O 64 structures
  emu.h / emu.cpp   — Emu: guest memory, loader, dispatcher, fault handling
  translator.h/.cpp — BlockBuilder: ARM64 -> x86-64 code generation
  host-fns.h/.cpp   — libSystem bridge
  main.cpp          — CLI entry point
test/               — benchmarks and the torture suite
asmjit/             — vendored AsmJit
```

## License

AGPL v3.

[AsmJit]: https://github.com/asmjit/asmjit