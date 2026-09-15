// ============================================================================
//  mach-o.h — Mach-O 64-bit on-disk structures
//
//  Reversetta parses these directly from the binary: no <mach-o/loader.h>
//  dependency, everything is self-contained and packed.
// ============================================================================

#pragma once

#include <cstdint>

#define MH_MAGIC_64   0xFEEDFACFu
#define FAT_MAGIC     0xCAFEBABEu
#define FAT_CIGAM     0xBEBAFECAu
#define CPU_TYPE_ARM64 0x0100000C

#define LC_REQ_DYLD   0x80000000u
#define LC_SEGMENT_64 0x19u
#define LC_SYMTAB     0x02u
#define LC_DYSYMTAB   0x0Bu
#define LC_MAIN       (0x28u | LC_REQ_DYLD)
#define LC_UNIXTHREAD 0x05u
#define LC_DYLD_CHAINED_FIXUPS (0x34u | LC_REQ_DYLD)

#define S_NON_LAZY_SYMBOL_POINTERS 0x6
#define S_LAZY_SYMBOL_POINTERS     0x7

#define INDIRECT_SYMBOL_LOCAL 0x80000000u
#define INDIRECT_SYMBOL_ABS   0x40000000u

#pragma pack(push, 1)
struct mach_header_64 {
    uint32_t magic; int32_t cputype; int32_t cpusubtype; uint32_t filetype;
    uint32_t ncmds; uint32_t sizeofcmds; uint32_t flags; uint32_t reserved;
};
struct load_command { uint32_t cmd; uint32_t cmdsize; };
struct segment_command_64 {
    uint32_t cmd; uint32_t cmdsize; char segname[16];
    uint64_t vmaddr; uint64_t vmsize; uint64_t fileoff; uint64_t filesize;
    int32_t maxprot; int32_t initprot; uint32_t nsects; uint32_t flags;
};
struct section_64 {
    char sectname[16]; char segname[16]; uint64_t addr; uint64_t size;
    uint32_t offset; uint32_t align; uint32_t reloff; uint32_t nreloc;
    uint32_t flags; uint32_t reserved1; uint32_t reserved2; uint32_t reserved3;
};
struct symtab_command {
    uint32_t cmd, cmdsize, symoff, nsyms, stroff, strsize;
};
struct dysymtab_command {
    uint32_t cmd, cmdsize, ilocalsym, nlocalsym, iextdefsym, nextdefsym;
    uint32_t iundefsym, nundefsym, tocoff, ntoc, modtaboff, nmodtab;
    uint32_t extrefsymoff, nextrefsyms, indirectsymoff, nindirectsyms;
    uint32_t extreloff, nextrel, locreloff, nlocrel;
};
struct entry_point_command { uint32_t cmd, cmdsize; uint64_t entryoff, stacksize; };
struct linkedit_data_command { uint32_t cmd, cmdsize, dataoff, datasize; };
struct nlist_64 {
    uint32_t n_strx; uint8_t n_type; uint8_t n_sect; uint16_t n_desc; uint64_t n_value;
};
struct fat_header { uint32_t magic, nfat_arch; };
struct fat_arch { int32_t cputype, cpusubtype; uint32_t offset, size, align; };

// LC_DYLD_CHAINED_FIXUPS
struct dyld_chained_fixups_header {
    uint32_t fixups_version, starts_offset, imports_offset, symbols_offset;
    uint32_t imports_count, imports_format, symbols_format;
};
struct dyld_chained_starts_in_segment {
    uint32_t size; uint16_t page_size; uint16_t pointer_format;
    uint64_t segment_offset; uint32_t max_valid_pointer;
    uint16_t page_count; uint16_t page_start[1];
};
#pragma pack(pop)

#define DYLD_CHAINED_PTR_64        2
#define DYLD_CHAINED_PTR_64_OFFSET 6
#define DYLD_CHAINED_PTR_START_NONE 0xFFFF
