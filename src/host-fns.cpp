// ============================================================================
//  host-fns.cpp — host implementations of libSystem imports
//
//  Arguments arrive in x0..x7, variadic ones on the guest stack.
// ============================================================================

#include "emu.h"
#include "host-fns.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <algorithm>

static FILE* file_from_guest(uint64_t h) {
    switch (h) {
        case FILE_STDIN:  return stdin;
        case FILE_STDOUT: return stdout;
        case FILE_STDERR: return stderr;
        default: return nullptr;
    }
}

// Parses a printf-style format and substitutes arguments from the guest
// stack. Every conversion is printed by a separate host snprintf call — no
// variadic trampoline is needed and the behaviour stays C-like.
std::string Emu::format_guest(uint64_t fmt_addr, uint64_t va) {
    std::string out;
    const char* f = guest_cstr(fmt_addr);
    if (!f) { out = "<reversetta: bad format pointer>"; return out; }

    auto next_slot = [&]() -> uint64_t { uint64_t v = read<uint64_t>(va); va += 8; return v; };
    std::vector<char> buf(256);

    for (const char* p = f; *p; ) {
        if (*p != '%') { out.push_back(*p++); continue; }
        const char* start = p++;
        if (*p == '%') { out.push_back('%'); p++; continue; }

        std::string spec = "%";
        while (*p && strchr("-+ #0'", *p)) spec.push_back(*p++);

        long width = 0, prec = 0;
        if (*p == '*') { width = (int32_t)(uint32_t)next_slot(); spec += std::to_string(width); p++; }
        else while (isdigit((unsigned char)*p)) { width = width * 10 + (*p - '0'); spec.push_back(*p++); }

        if (*p == '.') {
            spec.push_back(*p++);
            if (*p == '*') { prec = (int32_t)(uint32_t)next_slot(); spec += std::to_string(prec); p++; }
            else while (isdigit((unsigned char)*p)) { prec = prec * 10 + (*p - '0'); spec.push_back(*p++); }
        }

        std::string len;
        while (*p && strchr("hljztLq", *p)) len.push_back(*p++);
        char conv = *p ? *p++ : '\0';
        if (!conv) { out.append(start, (size_t)(p - start)); break; }

        bool wide = (len == "l" || len == "ll" || len == "j" || len == "z" ||
                     len == "t" || len == "q" || len == "L");

        size_t need = (size_t)width + (size_t)prec + 64;
        if (buf.size() < need) buf.resize(need);
        int n = 0;

        switch (conv) {
        case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': {
            uint64_t raw = next_slot();
            if (wide) {
                spec += "ll"; spec.push_back(conv);
                n = snprintf(buf.data(), buf.size(), spec.c_str(), (long long)raw);
            } else {
                spec += len; spec.push_back(conv);
                int v = (conv == 'd' || conv == 'i') ? (int)(int32_t)(uint32_t)raw
                                                     : (int)(uint32_t)raw;
                n = snprintf(buf.data(), buf.size(), spec.c_str(), v);
            }
            break;
        }
        case 'c': {
            uint64_t raw = next_slot();
            spec.push_back('c');
            n = snprintf(buf.data(), buf.size(), spec.c_str(), (int)(uint32_t)raw);
            break;
        }
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A': {
            uint64_t raw = next_slot();
            double d; std::memcpy(&d, &raw, 8);
            spec.push_back(conv);
            n = snprintf(buf.data(), buf.size(), spec.c_str(), d);
            break;
        }
        case 's': {
            uint64_t gp = next_slot();
            const char* s = gp ? guest_cstr(gp) : nullptr;
            if (!s) s = gp ? "<reversetta: bad pointer>" : "(null)";
            if (buf.size() < strlen(s) + need) buf.resize(strlen(s) + need);
            spec.push_back('s');
            n = snprintf(buf.data(), buf.size(), spec.c_str(), s);
            break;
        }
        case 'p': {
            uint64_t gp = next_slot();
            n = snprintf(buf.data(), buf.size(), "0x%llx", (unsigned long long)gp);
            break;
        }
        case 'n': {
            uint64_t gp = next_slot();
            write<uint32_t>(gp, (uint32_t)out.size());
            n = 0;
            break;
        }
        default:
            out.append(start, (size_t)(p - start));
            continue;
        }
        if (n > 0) out.append(buf.data(), (size_t)n);
    }
    return out;
}

// --- the implementations themselves: arguments in x0..x7, variadic ones on
//     the guest stack ---
#define HOST_FN(name) static void hf_##name(Emu& e)
#define ARG(i) (e.cpu.x[i])
#define RET(v) (e.cpu.x[0] = (uint64_t)(v))

HOST_FN(printf) {
    std::string s = e.format_guest(ARG(0), e.cpu.sp);
    fwrite(s.data(), 1, s.size(), stdout);
    RET(s.size());
}
HOST_FN(printf_chk) {                         // __printf_chk(flag, fmt, ...)
    std::string s = e.format_guest(ARG(1), e.cpu.sp);
    fwrite(s.data(), 1, s.size(), stdout);
    RET(s.size());
}
HOST_FN(fprintf) {
    FILE* f = file_from_guest(ARG(0));
    std::string s = e.format_guest(ARG(1), e.cpu.sp);
    if (f) fwrite(s.data(), 1, s.size(), f);
    RET(s.size());
}
HOST_FN(fprintf_chk) {                        // __fprintf_chk(stream, flag, fmt, ...)
    FILE* f = file_from_guest(ARG(0));
    std::string s = e.format_guest(ARG(2), e.cpu.sp);
    if (f) fwrite(s.data(), 1, s.size(), f);
    RET(s.size());
}
HOST_FN(sprintf) {
    std::string s = e.format_guest(ARG(1), e.cpu.sp);
    if (void* d = e.host_ptr(ARG(0), s.size() + 1)) {
        std::memcpy(d, s.c_str(), s.size() + 1);
    }
    RET(s.size());
}
HOST_FN(snprintf) {
    std::string s = e.format_guest(ARG(2), e.cpu.sp);
    uint64_t n = ARG(1);
    if (n) {
        uint64_t copy = std::min<uint64_t>(n - 1, s.size());
        if (void* d = e.host_ptr(ARG(0), copy + 1)) {
            std::memcpy(d, s.data(), copy);
            ((char*)d)[copy] = '\0';
        }
    }
    RET(s.size());
}
// Fortified variants from -D_FORTIFY_SOURCE (Apple appends the buffer size).
// For sprintf/snprintf the argument order differs:
// (str[, size], flag, dstlen, fmt, ...).
HOST_FN(sprintf_chk) {                       // __sprintf_chk(str, flag, dstlen, fmt, ...)
    std::string s = e.format_guest(ARG(3), e.cpu.sp);
    if (void* d = e.host_ptr(ARG(0), s.size() + 1)) std::memcpy(d, s.c_str(), s.size() + 1);
    RET(s.size());
}
HOST_FN(snprintf_chk) {                      // __snprintf_chk(str, maxlen, flag, dstlen, fmt, ...)
    std::string s = e.format_guest(ARG(4), e.cpu.sp);
    uint64_t n = ARG(1);
    if (n) {
        uint64_t copy = std::min<uint64_t>(n - 1, s.size());
        if (void* d = e.host_ptr(ARG(0), copy + 1)) {
            std::memcpy(d, s.data(), copy);
            ((char*)d)[copy] = '\0';
        }
    }
    RET(s.size());
}
HOST_FN(puts) {
    const char* s = e.guest_cstr(ARG(0));
    if (s) { fputs(s, stdout); fputc('\n', stdout); }
    RET(1);
}
HOST_FN(putchar) { fputc((int)(uint32_t)ARG(0), stdout); RET((uint32_t)ARG(0)); }
HOST_FN(fputc)   { FILE* f = file_from_guest(ARG(1)); if (f) fputc((int)(uint32_t)ARG(0), f); RET((uint32_t)ARG(0)); }
HOST_FN(fputs)   {
    const char* s = e.guest_cstr(ARG(0));
    FILE* f = file_from_guest(ARG(1));
    if (s && f) fputs(s, f);
    RET(1);
}
HOST_FN(fwrite) {
    uint64_t total = ARG(1) * ARG(2);
    FILE* f = file_from_guest(ARG(3));
    void* p = e.host_ptr(ARG(0), total);
    if (p && f) fwrite(p, 1, total, f);
    RET(ARG(2));
}
HOST_FN(fflush)  { FILE* f = file_from_guest(ARG(0)); fflush(f ? f : nullptr); RET(0); }
HOST_FN(exit)    { e.running = false; e.exit_code = (int)(uint32_t)ARG(0); }
HOST_FN(abort)   { fprintf(stderr, "[!] Guest called abort()\n"); e.running = false; e.exit_code = 134; }

HOST_FN(malloc)  { RET(e.heap_alloc(ARG(0))); }
HOST_FN(calloc)  {
    uint64_t n = ARG(0) * ARG(1);
    uint64_t p = e.heap_alloc(n);
    if (p) if (void* h = e.host_ptr(p, n)) std::memset(h, 0, n);
    RET(p);
}
HOST_FN(free)    { (void)e; }
HOST_FN(realloc) {
    uint64_t old = ARG(0), n = ARG(1);
    uint64_t p = e.heap_alloc(n);
    if (p && old) {
        uint64_t oldsz = e.read<uint64_t>(old - 16);
        uint64_t copy = std::min(oldsz, n);
        if (void* s = e.host_ptr(old, copy)) if (void* d = e.host_ptr(p, copy))
            std::memcpy(d, s, copy);
    }
    RET(p);
}
HOST_FN(strdup) {
    const char* s = e.guest_cstr(ARG(0));
    if (!s) { RET(0); return; }
    uint64_t n = strlen(s) + 1;
    uint64_t p = e.heap_alloc(n);
    if (p) if (void* d = e.host_ptr(p, n)) std::memcpy(d, s, n);
    RET(p);
}

HOST_FN(strlen)  { const char* s = e.guest_cstr(ARG(0)); RET(s ? strlen(s) : 0); }
HOST_FN(strcmp)  {
    const char* x = e.guest_cstr(ARG(0)); const char* y = e.guest_cstr(ARG(1));
    RET((int64_t)(x && y ? strcmp(x, y) : 0));
}
HOST_FN(strncmp) {
    const char* x = e.guest_cstr(ARG(0)); const char* y = e.guest_cstr(ARG(1));
    RET((int64_t)(x && y ? strncmp(x, y, ARG(2)) : 0));
}
HOST_FN(strcpy)  {
    const char* s = e.guest_cstr(ARG(1));
    if (s) if (void* d = e.host_ptr(ARG(0), strlen(s) + 1)) std::memcpy(d, s, strlen(s) + 1);
    RET(ARG(0));
}
HOST_FN(strncpy) {
    const char* s = e.guest_cstr(ARG(1));
    if (s) if (void* d = e.host_ptr(ARG(0), ARG(2))) strncpy((char*)d, s, ARG(2));
    RET(ARG(0));
}
HOST_FN(strcat)  {
    const char* d = e.guest_cstr(ARG(0)); const char* s = e.guest_cstr(ARG(1));
    if (d && s) {
        uint64_t at = ARG(0) + strlen(d);
        if (void* h = e.host_ptr(at, strlen(s) + 1)) std::memcpy(h, s, strlen(s) + 1);
    }
    RET(ARG(0));
}
HOST_FN(strchr)  {
    const char* s = e.guest_cstr(ARG(0));
    if (!s) { RET(0); return; }
    const char* h = strchr(s, (int)(uint32_t)ARG(1));
    RET(h ? ARG(0) + (uint64_t)(h - s) : 0);
}
HOST_FN(memcpy)  {
    void* d = e.host_ptr(ARG(0), ARG(2)); void* s = e.host_ptr(ARG(1), ARG(2));
    if (d && s) std::memcpy(d, s, ARG(2));
    RET(ARG(0));
}
HOST_FN(memmove) {
    void* d = e.host_ptr(ARG(0), ARG(2)); void* s = e.host_ptr(ARG(1), ARG(2));
    if (d && s) std::memmove(d, s, ARG(2));
    RET(ARG(0));
}
HOST_FN(memset)  {
    void* d = e.host_ptr(ARG(0), ARG(2));
    if (d) std::memset(d, (int)(uint32_t)ARG(1), ARG(2));
    RET(ARG(0));
}
HOST_FN(memcmp)  {
    void* x = e.host_ptr(ARG(0), ARG(2)); void* y = e.host_ptr(ARG(1), ARG(2));
    RET((int64_t)(x && y ? std::memcmp(x, y, ARG(2)) : 0));
}
HOST_FN(atoi)    { const char* s = e.guest_cstr(ARG(0)); RET((int64_t)(s ? atoi(s) : 0)); }
HOST_FN(abs_)    { int32_t v = (int32_t)(uint32_t)ARG(0); RET((uint32_t)(v < 0 ? -v : v)); }
HOST_FN(rand_)   { (void)e; RET((uint32_t)rand()); }
HOST_FN(srand_)  { srand((unsigned)ARG(0)); (void)e; }
HOST_FN(time_)   {
    time_t t = time(nullptr);
    if (ARG(0)) e.write<uint64_t>(ARG(0), (uint64_t)t);
    RET((uint64_t)t);
}
HOST_FN(stack_chk_fail) {
    fprintf(stderr, "[!] Guest: stack canary tripped (__stack_chk_fail)\n");
    e.running = false; e.exit_code = 134;
}
HOST_FN(assert_rtn) {
    const char* fn = e.guest_cstr(ARG(0));
    const char* fl = e.guest_cstr(ARG(1));
    const char* ex = e.guest_cstr(ARG(3));
    fprintf(stderr, "[!] Guest: assert(%s) failed in %s:%llu (%s)\n",
            ex ? ex : "?", fl ? fl : "?", (unsigned long long)ARG(2), fn ? fn : "?");
    e.running = false; e.exit_code = 134;
}

const std::unordered_map<std::string, HostFn> g_host_fns = {
    {"printf", hf_printf}, {"fprintf", hf_fprintf}, {"sprintf", hf_sprintf},
    {"snprintf", hf_snprintf}, {"puts", hf_puts}, {"putchar", hf_putchar},
    {"putc", hf_fputc}, {"fputc", hf_fputc}, {"fputs", hf_fputs},
    {"fwrite", hf_fwrite}, {"fflush", hf_fflush},
    {"exit", hf_exit}, {"_exit", hf_exit}, {"abort", hf_abort},
    {"malloc", hf_malloc}, {"calloc", hf_calloc}, {"realloc", hf_realloc},
    {"free", hf_free}, {"strdup", hf_strdup},
    {"strlen", hf_strlen}, {"strcmp", hf_strcmp}, {"strncmp", hf_strncmp},
    {"strcpy", hf_strcpy}, {"strncpy", hf_strncpy}, {"strcat", hf_strcat},
    {"strchr", hf_strchr},
    {"memcpy", hf_memcpy}, {"memmove", hf_memmove}, {"memset", hf_memset},
    {"memcmp", hf_memcmp},
    {"atoi", hf_atoi}, {"abs", hf_abs_}, {"rand", hf_rand_}, {"srand", hf_srand_},
    {"time", hf_time_},
    {"__stack_chk_fail", hf_stack_chk_fail}, {"__assert_rtn", hf_assert_rtn},

    // Fortified twins (-D_FORTIFY_SOURCE, enabled at -O1 and above).
    // Most of them just append a "buffer size" argument, so the plain
    // handlers read exactly the same registers.
    {"__strcpy_chk", hf_strcpy}, {"__strncpy_chk", hf_strncpy},
    {"__strcat_chk", hf_strcat}, {"__strncat_chk", hf_strcat},
    {"__memcpy_chk", hf_memcpy}, {"__memmove_chk", hf_memmove},
    {"__memset_chk", hf_memset},
    {"__printf_chk", hf_printf_chk}, {"__fprintf_chk", hf_fprintf_chk},
    {"__sprintf_chk", hf_sprintf_chk}, {"__snprintf_chk", hf_snprintf_chk},
};

// Data (not function) symbols imported by nearly every C program.
const std::unordered_map<std::string, uint64_t> g_host_data = {
    {"__stdinp", FILE_STDIN}, {"__stdoutp", FILE_STDOUT}, {"__stderrp", FILE_STDERR},
    {"__stack_chk_guard", 0x0000DEADC0DE0000ull},
};

uint64_t Emu::thunk_for(const std::string& name) {
    auto it = thunk_by_name.find(name);
    if (it != thunk_by_name.end()) return it->second;
    uint64_t addr = THUNK_BASE + thunks.size() * 4;
    thunks.push_back(name);
    thunk_by_name[name] = addr;
    return addr;
}

// Mach-O symbol name -> the value placed into the GOT slot.
uint64_t Emu::resolve_import(const std::string& raw) {
    std::string name = (!raw.empty() && raw[0] == '_') ? raw.substr(1) : raw;

    auto d = g_host_data.find(name);
    if (d != g_host_data.end()) {
        // A data symbol: allocate a cell in the heap and return its address.
        uint64_t cell = heap_alloc(8);
        write<uint64_t>(cell, d->second);
        log(1, "    [BIND] %s -> data @0x%llx\n", raw.c_str(), (unsigned long long)cell);
        return cell;
    }
    uint64_t t = thunk_for(name);
    log(1, "    [BIND] %s -> thunk 0x%llx%s\n", raw.c_str(), (unsigned long long)t,
        g_host_fns.count(name) ? "" : "  (!!! no implementation)");
    return t;
}
