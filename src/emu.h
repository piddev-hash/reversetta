// ============================================================================
//  emu.h — the emulator: guest memory, Mach-O loader, block dispatcher
// ============================================================================

#pragma once

#include "common.h"
#include "mach-o.h"
#include "host-fns.h"

#include <cstdio>
#include <cstdarg>

#include <vector>
#include <string>
#include <unordered_map>

#include <sys/mman.h>

#include "asmjit/asmjit/core.h"

struct Emu {
    uint8_t*  mem   = nullptr;
    uint64_t  base  = 0;               // image base == low bound of the window
    uint64_t  image_end = 0;
    uint8_t*  guard_lo = nullptr;      // whole mapping: guard | window | guard
    uint8_t*  guard_hi = nullptr;
    GuestCpu  cpu{};

    asmjit::JitRuntime rt;
    std::unordered_map<uint64_t, BlockFn> blocks;

    // ---- block chaining ----------------------------------------------------
    // Direct block links: b/bl/b.cond exit as a jmp rel32 that gets patched
    // to the target block body once it is compiled. Before the patch
    // rel32 == 0 and execution falls into the miss path (exit to the
    // dispatcher with pc already stored in the context).
    //
    // DMC (direct-mapped cache): pc -> block body for indirect branches
    // (br/blr/ret). Filled in by the block prologue on first execution.
    static constexpr uint32_t DMC_BITS = 16;
    static constexpr uint32_t DMC_SIZE = 1u << DMC_BITS;
    static constexpr uint32_t DMC_MASK = DMC_SIZE - 1;
    static constexpr uint64_t DMC_TAG_OFF  = 0;
    static constexpr uint64_t DMC_BODY_OFF = (uint64_t)DMC_SIZE * 8;

    std::vector<uint64_t> chain_tbl;    // [tags | bodies], base lives in r15

    struct PatchSite { uint64_t holder; uint32_t disp_off; };
    std::unordered_map<uint64_t, uint64_t> block_entry;   // pc -> body address
    std::unordered_map<uint64_t, std::vector<PatchSite>> pending_patches;

    void init_chains() {
        chain_tbl.assign(2 * (size_t)DMC_SIZE, 0);        // address stays stable for life
    }

    std::vector<std::string> thunks;   // index -> import name
    std::unordered_map<std::string, uint64_t> thunk_by_name;

    uint64_t heap_cur = 0, heap_end = 0;
    bool     running  = true;
    int      exit_code = 0;
    int      verbose  = 0;
    uint64_t insn_translated = 0;
    uint64_t blocks_compiled = 0;
    uint64_t block_execs = 0;

    ~Emu();

    // ---- memory ------------------------------------------------------------
    // The guest memory window is surrounded by PROT_NONE guards: out-of-bounds
    // accesses are caught by a signal handler, so generated code performs NO
    // inline address checks — every guest load/store is a single access of
    // the form [BIAS + base + disp].
    static constexpr uint64_t GUARD = 0x80000000ull;      // 2 GiB on each side

    bool map_memory(uint64_t image_base) {
        base = image_base;
        uint64_t total = GUARD + GUEST_WINDOW + GUARD;
        void* p = mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) return false;
        guard_lo = (uint8_t*)p;
        guard_hi = guard_lo + total;
        mem = guard_lo + GUARD;
        if (mprotect(mem, GUEST_WINDOW, PROT_READ | PROT_WRITE) != 0) return false;
        cpu.mem_bias   = (uint64_t)mem - base;
        cpu.guest_base = base;
        heap_cur = base + HEAP_OFF;
        heap_end = base + HEAP_OFF + HEAP_SIZE;
        return true;
    }

    bool in_range(uint64_t addr, uint64_t len) const {
        if (addr < base) return false;
        uint64_t off = addr - base;
        return off <= GUEST_WINDOW && len <= GUEST_WINDOW - off;
    }
    void* host_ptr(uint64_t addr, uint64_t len) const {
        return in_range(addr, len) ? (void*)(mem + (addr - base)) : nullptr;
    }
    template <typename T> T read(uint64_t addr) const {
        T v{};
        if (in_range(addr, sizeof(T))) std::memcpy(&v, mem + (addr - base), sizeof(T));
        return v;
    }
    template <typename T> void write(uint64_t addr, T v) {
        if (in_range(addr, sizeof(T))) std::memcpy(mem + (addr - base), &v, sizeof(T));
    }
    const char* guest_cstr(uint64_t addr) const {
        if (!in_range(addr, 1)) return nullptr;
        const char* s = (const char*)(mem + (addr - base));
        uint64_t max = GUEST_WINDOW - (addr - base);
        return memchr(s, 0, max) ? s : nullptr;
    }

    uint64_t heap_alloc(uint64_t size) {
        uint64_t need = (size + 15) & ~15ull;
        if (heap_cur + need + 16 > heap_end) return 0;
        uint64_t hdr = heap_cur;
        write<uint64_t>(hdr, need);
        uint64_t p = hdr + 16;
        heap_cur = p + need;
        return p;
    }

    // ---- logging -----------------------------------------------------------
    void log(int level, const char* fmt, ...) __attribute__((format(printf, 3, 4))) {
        if (verbose < level) return;
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
    }

    // ---- imports -----------------------------------------------------------
    uint64_t thunk_for(const std::string& name);
    uint64_t resolve_import(const std::string& raw_name);

    // ---- loader ------------------------------------------------------------
    bool load(const std::string& path, uint64_t& entry_out);
    bool bind_chained_fixups(const uint8_t* file, uint64_t file_size,
                             const linkedit_data_command& lc);

    // ---- execution ---------------------------------------------------------
    BlockFn block_for(uint64_t pc);
    int run(uint64_t entry, const std::vector<std::string>& argv);
    bool handle_thunk(uint64_t pc);
    void handle_syscall();

    // ---- printf bridge -----------------------------------------------------
    std::string format_guest(uint64_t fmt_addr, uint64_t va_sp);
};
