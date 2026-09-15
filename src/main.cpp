// ============================================================================
//  REVESETTA — ARM64 (Mach-O) -> x86-64 JIT
//  "Nizhnepodmyshinsk Monolith", v0.4
//
//  What it is: the reverse Rosetta. Takes a real macOS arm64 binary, maps its
//  segments into virtual memory, translates ARM64 code into native x86-64
//  basic blocks (via AsmJit) and executes it. Calls into libSystem (printf
//  and friends) are intercepted and routed to host implementations.
//
//  Architecture:
//    1. Mach-O loader  — segments, entry point, symbols, fixups.
//    2. Guest memory   — one flat window, host = mem_bias + guest_addr.
//    3. Block JIT      — ARM64 -> x86-64, one basic block at a time.
//    4. Dispatcher     — spins blocks, catches "magic" host-function PCs.
//
//  Speed (v0.4):
//    - block chaining: b/bl/b.cond jump straight to the target block body
//      (a patched jmp rel32), no return to the dispatcher;
//    - indirect branches (br/blr/ret) go through a direct-mapped cache
//      embedded right into the generated code;
//    - global pinning: x0/x8/x29/x30/sp permanently live in host registers
//      (r10/r11/rbx/rbp/r14); they sync with GuestCpu only on dispatcher
//      entry/exit, chained transitions inherit them;
//    - lazy flags: subs + b.cond/csel consume x86 RFLAGS directly, no
//      NZCV round-trips through memory;
//    - the guest memory is surrounded by guard pages: bounds are caught by
//      a SIGSEGV/SIGBUS handler, there are no inline checks in the hot path;
//    - a base-register cache for ldur/str series + peepholes (lea,
//      memory-form ALU, Wn stores without a redundant mov r32,r32).
//
//  Engine logs go to stderr, guest program output to stdout.
// ============================================================================

#include "emu.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char* prog) {
    fprintf(stderr,
        "Reversetta — run macOS arm64 binaries on x86-64\n"
        "\n"
        "Usage: %s [options] <arm64-binary> [program args...]\n"
        "  -v     show instruction translation\n"
        "  -vv    also trace blocks and host calls\n"
        "  -vvv   also dump generated x86 code bytes\n"
        "  -h     this help\n"
        "\n"
        "Engine logs go to stderr, guest output to stdout.\n", prog);
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    int i = 1;
    for (; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v") verbose = 1;
        else if (a == "-vv") verbose = 2;
        else if (a == "-vvv") verbose = 3;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else break;
    }
    if (i >= argc) { usage(argv[0]); return 1; }

    std::string path = argv[i];
    std::vector<std::string> guest_args;
    for (int k = i; k < argc; k++) guest_args.push_back(argv[k]);

    Emu emu;
    emu.verbose = verbose;
    emu.init_chains();

    // Index 0 is always taken by "return from main": LR points here.
    emu.thunk_for("__reversetta_exit");

    uint64_t entry = 0;
    if (!emu.load(path, entry)) return 1;

    printf("=== Reversetta v0.4 ===\n");
    fflush(stdout);

    int rc = emu.run(entry, guest_args);

    fflush(stdout);
    printf("======== end ========\n");

    emu.log(1, "[+] Done. Guest exit code: %d "
               "(blocks: %llu, instructions translated: %llu, dispatcher entries: %llu)\n",
            rc, (unsigned long long)emu.blocks_compiled,
            (unsigned long long)emu.insn_translated,
            (unsigned long long)emu.block_execs);
    return rc;
}
