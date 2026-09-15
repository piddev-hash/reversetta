// ============================================================================
//  host-fns.h — bridge from guest imports to host implementations
//
//  g_host_fns maps an import name (printf, malloc, ...) to a native function.
//  g_host_data maps imported data symbols (__stdoutp, __stack_chk_guard, ...)
//  to the values exposed to the guest.
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

struct Emu;
using HostFn = void (*)(Emu&);

extern const std::unordered_map<std::string, HostFn> g_host_fns;
extern const std::unordered_map<std::string, uint64_t> g_host_data;
