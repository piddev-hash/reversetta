// ============================================================================
//  common.h — shared constants, virtual CPU state and bit helpers
//
//  Reversetta — the reverse Rosetta: runs real macOS arm64 binaries on
//  x86-64 by translating ARM64 code to native x86-64 basic blocks.
// ============================================================================

#pragma once

#include <cstdint>
#include <cstring>

// ----------------------------------------------------------------------------
//  Guest memory layout
// ----------------------------------------------------------------------------

// Guest memory window: [image_base, image_base + GUEST_WINDOW).
// Mach-O segments, the heap and the stack live inside it.
constexpr uint64_t GUEST_WINDOW  = 0x10000000ull;   // 256 MiB
constexpr uint64_t HEAP_OFF      = 0x08000000ull;   // heap       +128 MiB
constexpr uint64_t HEAP_SIZE     = 0x02000000ull;   //                 32 MiB
constexpr uint64_t STACK_TOP_OFF = 0x0F000000ull;   // stack top
constexpr uint64_t STACK_SIZE    = 0x00800000ull;   //                  8 MiB
constexpr uint64_t ARGV_AREA     = 0x1000ull;       // argc/argv/envp

// "Magic" host-function addresses. The guest never executes them as code:
// the dispatcher recognizes such a PC and calls the native implementation.
constexpr uint64_t THUNK_BASE = 0xF00D000000000000ull;
constexpr uint64_t THUNK_EXIT = THUNK_BASE;         // index 0 — return from main

// Magic FILE* values for stdin/stdout/stderr.
constexpr uint64_t FILE_STDIN  = 0xF11E0001ull;
constexpr uint64_t FILE_STDOUT = 0xF11E0002ull;
constexpr uint64_t FILE_STDERR = 0xF11E0003ull;

constexpr int MAX_BLOCK_INSNS = 512;

// ----------------------------------------------------------------------------
//  Virtual CPU state
// ----------------------------------------------------------------------------

enum ExitReason : uint32_t {
    kExitNone = 0,
    kExitMemFault,
    kExitUndefInsn,
    kExitSyscall,
};

// Field layout matters: generated code addresses fields via offsetof().
struct GuestCpu {
    uint64_t x[32];            // x0..x30; [31] — dummy slot for XZR (never used)
    uint64_t sp;
    uint64_t pc;
    uint8_t  fN, fZ, fC, fV;   // NZCV, one byte per flag — setcc-friendly
    uint8_t  pad[4];
    uint64_t mem_bias;         // host_ptr = mem_bias + guest_addr
    uint64_t guest_base;       // low bound of the guest memory window
    uint64_t fault_addr;
    uint64_t scratch;
    uint32_t exit_reason;
    uint32_t exit_info;
};

using BlockFn = void (*)(GuestCpu*);

// ----------------------------------------------------------------------------
//  Bit helpers + the ARM64 logical-immediate mask decoder
// ----------------------------------------------------------------------------

inline uint32_t bits(uint32_t v, unsigned hi, unsigned lo) {
    unsigned w = hi - lo + 1;
    return (v >> lo) & (w >= 32 ? 0xFFFFFFFFu : ((1u << w) - 1u));
}
inline uint32_t bit(uint32_t v, unsigned n) { return (v >> n) & 1u; }

// Sign-extend a value that is n bits wide to int64_t.
inline int64_t sext(uint64_t v, unsigned n) {
    const uint64_t m = 1ull << (n - 1);
    return (int64_t)((v & ((n >= 64) ? ~0ull : ((1ull << n) - 1))) ^ m) - (int64_t)m;
}

inline uint64_t ones_mask(unsigned n) { return n >= 64 ? ~0ull : ((1ull << n) - 1ull); }

// DecodeBitMasks from the ARM ARM: turns (N, imms, immr) into a mask pair.
// wmask is needed by every logical-immediate operation, tmask by bitfields
// (SBFM/BFM/UBFM).
inline bool decode_bit_masks(uint32_t N, uint32_t imms, uint32_t immr,
                             bool immediate, int datasize,
                             uint64_t& wmask, uint64_t& tmask) {
    uint32_t combined = (N << 6) | ((~imms) & 0x3Fu);
    if (combined == 0) return false;
    int len = 31 - __builtin_clz(combined);
    if (len < 1) return false;
    if (datasize < (1 << len)) return false;

    uint32_t levels = (uint32_t)ones_mask((unsigned)len);
    if (immediate && ((imms & levels) == levels)) return false;

    uint32_t S = imms & levels;
    uint32_t R = immr & levels;
    uint32_t diff = (S - R) & levels;
    unsigned esize = 1u << len;

    uint64_t welem = ones_mask(S + 1);
    uint64_t telem = ones_mask(diff + 1);
    uint64_t emask = ones_mask(esize);

    uint64_t wrot = (R == 0) ? welem
                             : (((welem >> R) | (welem << (esize - R))) & emask);

    auto replicate = [&](uint64_t e) {
        uint64_t r = 0;
        for (unsigned i = 0; i < 64; i += esize) r |= (e & emask) << i;
        return r;
    };
    wmask = replicate(wrot);
    tmask = replicate(telem);
    return true;
}
