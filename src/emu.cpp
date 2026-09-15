// ============================================================================
//  emu.cpp — Mach-O loader, block dispatcher and fault handling
// ============================================================================

#include "emu.h"
#include "translator.h"
#include "host-fns.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>

#include <signal.h>
#include <setjmp.h>
#include <libkern/OSCacheControl.h>

using namespace asmjit;

// ----------------------------------------------------------------------------
//  Catching guest faults via guard pages
//
//  The guest memory is surrounded by PROT_NONE pages: any access outside the
//  window turns into SIGSEGV/SIGBUS. The handler converts it into
//  kExitMemFault and jumps back into the dispatcher. Foreign faults (engine
//  bugs) are handed back to the system.
// ----------------------------------------------------------------------------
static Emu*          g_fault_emu   = nullptr;
static sigjmp_buf    g_fault_jmp;
static bool          g_fault_armed = false;
static struct sigaction g_fault_old[2];
static const int     g_fault_sigs[2] = { SIGSEGV, SIGBUS };

static void guest_fault_handler(int, siginfo_t* si, void*) {
    if (g_fault_armed && g_fault_emu) {
        uintptr_t a = (uintptr_t)si->si_addr;
        if (a >= (uintptr_t)g_fault_emu->guard_lo && a < (uintptr_t)g_fault_emu->guard_hi) {
            g_fault_emu->cpu.fault_addr = (uint64_t)a - g_fault_emu->cpu.mem_bias;
            g_fault_emu->cpu.exit_reason = kExitMemFault;
            siglongjmp(g_fault_jmp, 1);
        }
    }
    // not our fault (engine bug) — restore the default handler and crash honestly
    for (int i = 0; i < 2; i++)
        sigaction(g_fault_sigs[i], &g_fault_old[i], nullptr);
}

Emu::~Emu() {
    if (guard_lo) munmap(guard_lo, GUARD + GUEST_WINDOW + GUARD);
}

// ----------------------------------------------------------------------------
//  Mach-O loader
// ----------------------------------------------------------------------------
bool Emu::bind_chained_fixups(const uint8_t* file, uint64_t file_size,
                              const linkedit_data_command& lc) {
    if (lc.dataoff + lc.datasize > file_size) return false;
    const uint8_t* blob = file + lc.dataoff;
    auto* hdr = (const dyld_chained_fixups_header*)blob;

    // Import table: ordinal -> name.
    std::vector<uint64_t> import_values;
    const char* symbols = (const char*)(blob + hdr->symbols_offset);
    const uint32_t* imports = (const uint32_t*)(blob + hdr->imports_offset);
    for (uint32_t i = 0; i < hdr->imports_count; i++) {
        uint32_t ent, name_off;
        if (hdr->imports_format == 1) {          // DYLD_CHAINED_IMPORT
            ent = imports[i];
            name_off = ent >> 9;
        } else if (hdr->imports_format == 2) {   // DYLD_CHAINED_IMPORT_ADDEND
            ent = ((const uint32_t*)imports)[i * 2];
            name_off = ent >> 9;
        } else {
            log(1, "[!] Unknown chained fixups import format (%u)\n", hdr->imports_format);
            return false;
        }
        import_values.push_back(resolve_import(symbols + name_off));
    }

    auto* starts = (const uint32_t*)(blob + hdr->starts_offset);
    uint32_t seg_count = starts[0];
    for (uint32_t s = 0; s < seg_count; s++) {
        uint32_t off = starts[1 + s];
        if (!off) continue;
        auto* seg = (const dyld_chained_starts_in_segment*)(blob + hdr->starts_offset + off);
        if (seg->pointer_format != DYLD_CHAINED_PTR_64 &&
            seg->pointer_format != DYLD_CHAINED_PTR_64_OFFSET) {
            log(1, "[!] Unsupported pointer_format=%u — skipping segment\n",
                seg->pointer_format);
            continue;
        }
        bool offset_form = (seg->pointer_format == DYLD_CHAINED_PTR_64_OFFSET);

        for (uint16_t pg = 0; pg < seg->page_count; pg++) {
            uint16_t start = seg->page_start[pg];
            if (start == DYLD_CHAINED_PTR_START_NONE) continue;
            uint64_t addr = base + seg->segment_offset + (uint64_t)pg * seg->page_size + start;
            while (true) {
                uint64_t raw = read<uint64_t>(addr);
                uint32_t next = (uint32_t)((raw >> 51) & 0xFFF);
                bool is_bind = (raw >> 63) & 1;
                if (is_bind) {
                    uint32_t ordinal = (uint32_t)(raw & 0xFFFFFF);
                    if (ordinal < import_values.size()) write<uint64_t>(addr, import_values[ordinal]);
                } else {
                    uint64_t target = raw & 0xFFFFFFFFFull;      // 36 bits
                    uint64_t high8 = (raw >> 36) & 0xFF;
                    uint64_t value = offset_form ? (base + target) : (target | (high8 << 56));
                    write<uint64_t>(addr, value);
                }
                if (!next) break;
                addr += (uint64_t)next * 4;
            }
        }
    }
    return true;
}

bool Emu::load(const std::string& path, uint64_t& entry_out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[-] Cannot open file: %s\n", path.c_str()); return false; }
    uint64_t size = (uint64_t)f.tellg();
    std::vector<uint8_t> file(size);
    f.seekg(0);
    f.read((char*)file.data(), (std::streamsize)size);

    uint64_t slice = 0;
    uint32_t magic = *(const uint32_t*)file.data();
    if (magic == FAT_MAGIC || magic == FAT_CIGAM) {
        auto be32 = [](uint32_t v) { return __builtin_bswap32(v); };
        auto* fh = (const fat_header*)file.data();
        uint32_t n = be32(fh->nfat_arch);
        auto* arch = (const fat_arch*)(file.data() + sizeof(fat_header));
        bool found = false;
        for (uint32_t i = 0; i < n; i++) {
            if ((int32_t)be32((uint32_t)arch[i].cputype) == CPU_TYPE_ARM64) {
                slice = be32(arch[i].offset); found = true; break;
            }
        }
        if (!found) { fprintf(stderr, "[-] No arm64 slice in the fat binary.\n"); return false; }
        log(1, "[*] Fat binary: using the arm64 slice at offset 0x%llx\n",
            (unsigned long long)slice);
    }

    auto* mh = (const mach_header_64*)(file.data() + slice);
    if (mh->magic != MH_MAGIC_64) { fprintf(stderr, "[-] Not a 64-bit Mach-O.\n"); return false; }
    if (mh->cputype != CPU_TYPE_ARM64) {
        fprintf(stderr, "[-] cputype=0x%x — arm64 (0x%x) required.\n", mh->cputype, CPU_TYPE_ARM64);
        return false;
    }

    // --- first pass: find the image base ---
    uint64_t image_base = ~0ull;
    const uint8_t* cmds = file.data() + slice + sizeof(mach_header_64);
    const uint8_t* p = cmds;
    for (uint32_t i = 0; i < mh->ncmds; i++) {
        auto* lc = (const load_command*)p;
        if (lc->cmd == LC_SEGMENT_64) {
            auto* seg = (const segment_command_64*)p;
            if (seg->filesize > 0 && seg->vmaddr < image_base) image_base = seg->vmaddr;
        }
        p += lc->cmdsize;
    }
    if (image_base == ~0ull) { fprintf(stderr, "[-] No segments found.\n"); return false; }

    if (!map_memory(image_base)) { fprintf(stderr, "[-] Guest memory mmap failed.\n"); return false; }
    log(1, "[*] Guest memory: 0x%llx .. 0x%llx (%llu MiB), host base %p\n",
        (unsigned long long)base, (unsigned long long)(base + GUEST_WINDOW),
        (unsigned long long)(GUEST_WINDOW >> 20), (void*)mem);

    // --- second pass: segments, entry point, symbol tables ---
    const symtab_command* symtab = nullptr;
    const dysymtab_command* dysym = nullptr;
    const linkedit_data_command* chained = nullptr;
    std::vector<const section_64*> sections;
    uint64_t entry = 0;

    p = cmds;
    for (uint32_t i = 0; i < mh->ncmds; i++) {
        auto* lc = (const load_command*)p;
        switch (lc->cmd) {
        case LC_SEGMENT_64: {
            auto* seg = (const segment_command_64*)p;
            std::string nm(seg->segname, strnlen(seg->segname, 16));
            if (seg->filesize > 0) {
                if (!in_range(seg->vmaddr, seg->vmsize)) {
                    fprintf(stderr, "[-] Segment %s does not fit into the guest memory window.\n", nm.c_str());
                    return false;
                }
                std::memcpy(mem + (seg->vmaddr - base),
                            file.data() + slice + seg->fileoff, seg->filesize);
                if (seg->vmaddr + seg->vmsize > image_end) image_end = seg->vmaddr + seg->vmsize;
                log(1, "[*] Segment %-12s 0x%011llx +0x%llx\n", nm.c_str(),
                    (unsigned long long)seg->vmaddr, (unsigned long long)seg->filesize);
            }
            auto* sect = (const section_64*)(p + sizeof(segment_command_64));
            for (uint32_t k = 0; k < seg->nsects; k++) sections.push_back(&sect[k]);
            break;
        }
        case LC_SYMTAB:   symtab = (const symtab_command*)p; break;
        case LC_DYSYMTAB: dysym  = (const dysymtab_command*)p; break;
        case LC_MAIN:     entry = image_base + ((const entry_point_command*)p)->entryoff; break;
        case LC_DYLD_CHAINED_FIXUPS: chained = (const linkedit_data_command*)p; break;
        default: break;
        }
        p += lc->cmdsize;
    }

    if (!entry) { fprintf(stderr, "[-] No LC_MAIN — entry point unknown.\n"); return false; }
    entry_out = entry;

    // --- fixups and imports ---
    if (chained) {
        log(1, "[*] Applying LC_DYLD_CHAINED_FIXUPS...\n");
        bind_chained_fixups(file.data() + slice, size - slice, *chained);
    }
    if (symtab && dysym && dysym->nindirectsyms) {
        auto* nl = (const nlist_64*)(file.data() + slice + symtab->symoff);
        const char* strs = (const char*)(file.data() + slice + symtab->stroff);
        auto* indirect = (const uint32_t*)(file.data() + slice + dysym->indirectsymoff);

        for (const section_64* s : sections) {
            uint32_t type = s->flags & 0xFF;
            if (type != S_NON_LAZY_SYMBOL_POINTERS && type != S_LAZY_SYMBOL_POINTERS) continue;
            uint64_t count = s->size / 8;
            for (uint64_t k = 0; k < count; k++) {
                uint32_t idx = indirect[s->reserved1 + k];
                if (idx & (INDIRECT_SYMBOL_LOCAL | INDIRECT_SYMBOL_ABS)) continue;
                if (idx >= symtab->nsyms) continue;
                write<uint64_t>(s->addr + k * 8, resolve_import(strs + nl[idx].n_strx));
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
//  Block dispatcher
// ----------------------------------------------------------------------------
BlockFn Emu::block_for(uint64_t pc) {
    auto it = blocks.find(pc);
    if (it != blocks.end()) return it->second;

    log(1, "[*] Translating block @0x%llx\n", (unsigned long long)pc);
    CodeHolder code;
    if (code.init(rt.environment()) != Error::kOk) return nullptr;
    x86::Assembler a(&code);

    BlockBuilder bb(*this, a, pc);
    if (!bb.build()) return nullptr;

    BlockFn fn = nullptr;
    Error err = rt.add(&fn, &code);
    if (err != Error::kOk) {
        fprintf(stderr, "[-] AsmJit failed to assemble block @0x%llx (error %u)\n",
                (unsigned long long)pc, (uint32_t)err);
        return nullptr;
    }
    if (verbose >= 3) {
        // Dump the generated block bytes (dtrace/otool-friendly).
        CodeBuffer& buf = code.section_by_id(0)->buffer();
        fprintf(stderr, "[CODE] block @0x%llx size=%zu\n", (unsigned long long)pc, buf.size());
        for (size_t i = 0; i < buf.size(); i += 16) {
            fprintf(stderr, "  %04zx:", i);
            for (size_t j = i; j < std::min(i + 16, buf.size()); j++)
                fprintf(stderr, " %02x", buf.data()[j]);
            fputc('\n', stderr);
        }
    }

    // Patch the waiting sites of this block and register its entry (body).
    auto patch_site = [](uint64_t holder, uint32_t disp_off, uint64_t body) {
        int64_t rel = (int64_t)body - ((int64_t)holder + disp_off + 4);
        if (rel < INT32_MIN || rel > INT32_MAX) return false;   // too far — keep the miss path
        int32_t rel32 = (int32_t)rel;
        memcpy((uint8_t*)holder + disp_off, &rel32, 4);
        sys_icache_invalidate((uint8_t*)holder + disp_off, 4);
        return true;
    };

    {
        auto it = pending_patches.find(pc);
        if (it != pending_patches.end()) {
            for (const PatchSite& s : it->second)
                patch_site(s.holder, s.disp_off, (uint64_t)fn + bb.body_offset());
            pending_patches.erase(it);
        }
    }
    for (const BlockBuilder::Patch& s : bb.take_patches()) {
        auto it = block_entry.find(s.target);
        if (it != block_entry.end())
            patch_site((uint64_t)fn, s.disp_off, it->second);
        else
            pending_patches[s.target].push_back({(uint64_t)fn, s.disp_off});
    }
    block_entry[pc] = (uint64_t)fn + bb.body_offset();

    blocks_compiled++;
    blocks[pc] = fn;
    return fn;
}

bool Emu::handle_thunk(uint64_t pc) {
    uint64_t idx = (pc - THUNK_BASE) / 4;
    if (idx >= thunks.size()) {
        fprintf(stderr, "[-] Jump to unknown thunk 0x%llx\n", (unsigned long long)pc);
        return false;
    }
    const std::string& name = thunks[idx];
    if (name == "__reversetta_exit") {            // main returned
        running = false;
        exit_code = (int)(int32_t)(uint32_t)cpu.x[0];
        return true;
    }
    auto it = g_host_fns.find(name);
    if (it == g_host_fns.end()) {
        fprintf(stderr, "[-] Import '%s' is not implemented in Reversetta.\n", name.c_str());
        running = false; exit_code = 127;
        return false;
    }
    log(2, "    [HOST] %s(x0=0x%llx, x1=0x%llx)\n", name.c_str(),
        (unsigned long long)cpu.x[0], (unsigned long long)cpu.x[1]);
    it->second(*this);
    cpu.pc = cpu.x[30];                          // return via LR
    return true;
}

// A minimal SVC handler: enough for statically linked programs that talk to
// the kernel directly (the syscall number travels in x16).
void Emu::handle_syscall() {
    uint64_t nr = cpu.x[16] & 0xFFFFFF;
    switch (nr) {
        case 1:                                   // exit
            running = false; exit_code = (int)(uint32_t)cpu.x[0];
            break;
        case 4: {                                 // write
            void* p = host_ptr(cpu.x[1], cpu.x[2]);
            FILE* f = (cpu.x[0] == 2) ? stderr : stdout;
            if (p) fwrite(p, 1, cpu.x[2], f);
            cpu.x[0] = cpu.x[2];
            break;
        }
        default:
            fprintf(stderr, "[-] Unsupported syscall #%llu\n", (unsigned long long)nr);
            running = false; exit_code = 127;
            break;
    }
    cpu.exit_reason = kExitNone;
}

int Emu::run(uint64_t entry, const std::vector<std::string>& args) {
    // --- stack, argv, envp ---
    uint64_t stack_top = base + STACK_TOP_OFF;
    uint64_t info = stack_top - ARGV_AREA;
    uint64_t cursor = info;

    std::vector<uint64_t> argv_ptrs;
    for (const auto& s : args) {
        uint64_t at = cursor;
        // The argv area's tail abuts the stack top — long arguments get cut.
        if (cursor + s.size() + 1 >= stack_top) {
            fprintf(stderr, "[!] Arguments do not fit into the argv area — truncating the list.\n");
            break;
        }
        std::memcpy(mem + (at - base), s.c_str(), s.size() + 1);
        cursor += (s.size() + 1 + 7) & ~7ull;
        argv_ptrs.push_back(at);
    }
    cursor = (cursor + 15) & ~15ull;
    uint64_t argv_addr = cursor;
    for (uint64_t a : argv_ptrs) { write<uint64_t>(cursor, a); cursor += 8; }
    write<uint64_t>(cursor, 0); cursor += 8;      // argv[argc] = NULL
    uint64_t envp_addr = cursor;
    write<uint64_t>(cursor, 0); cursor += 8;      // empty environ
    uint64_t apple_addr = cursor;
    write<uint64_t>(cursor, 0);

    std::memset(&cpu.x, 0, sizeof(cpu.x));
    cpu.x[0] = argv_ptrs.size();
    cpu.x[1] = argv_addr;
    cpu.x[2] = envp_addr;
    cpu.x[3] = apple_addr;
    cpu.x[30] = THUNK_EXIT;                       // LR: main returns here
    cpu.sp = info & ~15ull;
    cpu.pc = entry;
    cpu.exit_reason = kExitNone;

    log(1, "[*] Start: PC=0x%llx, SP=0x%llx, argc=%zu\n",
        (unsigned long long)entry, (unsigned long long)cpu.sp, args.size());

    // Install fault handlers: guest faults through guard pages arrive here
    // (macOS raises SIGSEGV or SIGBUS depending on the access type).
    struct sigaction sa{};
    sa.sa_sigaction = guest_fault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    for (int i = 0; i < 2; i++) sigaction(g_fault_sigs[i], &sa, &g_fault_old[i]);
    g_fault_emu = this;

    bool faulted = false;
    if (sigsetjmp(g_fault_jmp, 1) == 0) {
        g_fault_armed = true;

        while (running) {
            if (cpu.pc >= THUNK_BASE) {
                if (!handle_thunk(cpu.pc)) break;
                continue;
            }
            if (verbose >= 2) {
                log(2, "    [PC] 0x%llx  x0=%llx x1=%llx x8=%llx sp=%llx\n",
                    (unsigned long long)cpu.pc, (unsigned long long)cpu.x[0],
                    (unsigned long long)cpu.x[1], (unsigned long long)cpu.x[8],
                    (unsigned long long)cpu.sp);
            }
            BlockFn fn = block_for(cpu.pc);
            if (!fn) { exit_code = 70; break; }

            block_execs++;
            fn(&cpu);

            if (cpu.exit_reason == kExitMemFault) { faulted = true; break; }
            if (cpu.exit_reason == kExitUndefInsn) {
                fflush(stdout);
                fprintf(stderr, "[-] Cannot translate instruction 0x%08x at address 0x%llx\n",
                        cpu.exit_info, (unsigned long long)cpu.pc);
                fprintf(stderr, "    (most likely SIMD/FP — not supported yet)\n");
                exit_code = 4; break;
            }
            if (cpu.exit_reason == kExitSyscall) handle_syscall();
        }
    } else {
        faulted = true;                     // siglongjmp from the fault handler
    }
    g_fault_armed = false;
    for (int i = 0; i < 2; i++) sigaction(g_fault_sigs[i], &g_fault_old[i], nullptr);

    if (faulted) {
        fflush(stdout);
        fprintf(stderr, "[-] Guest memory access out of bounds: address 0x%llx\n",
                (unsigned long long)cpu.fault_addr);
        exit_code = 139;
    }

    fflush(stdout);
    return exit_code;
}
