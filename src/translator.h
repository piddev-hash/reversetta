// ============================================================================
//  translator.h — ARM64 -> x86-64 basic block translator
//
//  BlockBuilder emits one basic block at a time. Key optimizations:
//    - guest registers x0/x8/x29/x30/sp are pinned to host registers and
//      survive across chained block links;
//    - lazy flags: subs + b.cond/csel consume x86 RFLAGS directly;
//    - block exits are patchable jmp rel32 / DMC lookups;
//    - a one-entry base-register cache serves ldr/str series.
// ============================================================================

#pragma once

#include "emu.h"

#include "asmjit/asmjit/x86.h"

enum class Step { kNext, kEnd, kBad };

// x86 registers reserved for the block execution environment.
inline const asmjit::x86::Gp CTX   = asmjit::x86::r12;  // GuestCpu*
inline const asmjit::x86::Gp BIAS  = asmjit::x86::r13;  // mem_bias
inline const asmjit::x86::Gp CHAIN = asmjit::x86::r15;  // block-chain table base (DMC)

// Global guest register pinning: pinned registers live in host registers for
// the whole duration of a block chain. They are synced with GuestCpu only in
// the prologue (dispatcher entry) and epilogue (dispatcher exit); chained
// transitions (jmp rel32 / DMC) simply inherit them.
//   rbx -> x29 (FP)   rbp -> x30 (LR)   r14 -> SP   r10 -> x0   r11 -> x8
inline const asmjit::x86::Gp PIN_X29 = asmjit::x86::rbx;
inline const asmjit::x86::Gp PIN_X30 = asmjit::x86::rbp;
inline const asmjit::x86::Gp PIN_SP  = asmjit::x86::r14;
inline const asmjit::x86::Gp PIN_X0  = asmjit::x86::r10;
inline const asmjit::x86::Gp PIN_X8  = asmjit::x86::r11;

// Returns the host register if guest register r is pinned, else an empty Gp.
inline const asmjit::x86::Gp& pin_of(int r) {
    static const asmjit::x86::Gp none = asmjit::x86::Gp{};
    switch (r) {
        case 0:  return PIN_X0;
        case 8:  return PIN_X8;
        case 29: return PIN_X29;
        case 30: return PIN_X30;
        default: return none;
    }
}

class BlockBuilder {
public:
    BlockBuilder(Emu& e, asmjit::x86::Assembler& asmb, uint64_t start)
        : e(e), a(asmb), pc_start(start) {}

    bool build();

    // Offset of the body label from the block start (for chaining entries).
    uint32_t body_offset() const { return body_off; }
    // Direct-link sites awaiting a patch: {rel32 offset, guest target}.
    struct Patch { uint32_t disp_off; uint64_t target; };
    std::vector<Patch> take_patches() { return std::move(patches); }

private:
    Emu& e;
    asmjit::x86::Assembler& a;
    uint64_t pc_start;
    uint32_t body_off = 0;
    std::vector<Patch> patches;
    asmjit::Label epilogue;

    // ---- flag fusion ----
    // After an S-instruction the guest NZCV equivalently live in x86 RFLAGS.
    // The nearest consumer (b.cond / csel family / ccmp) can consume them
    // directly via jcc/setcc/cmovcc, bypassing memory.
    enum class FlagSrc : uint8_t { kNone, kFromAdd, kFromSub, kFromLogic };
    FlagSrc flag_src = FlagSrc::kNone;   // what produced the pending flags

    // Inversion of an x86 condition (every one has a pair).
    static asmjit::x86::CondCode inv_cc(asmjit::x86::CondCode c) {
        using CC = asmjit::x86::CondCode;
        switch (c) {
            case CC::kO:  return CC::kNO;  case CC::kNO: return CC::kO;
            case CC::kC:  return CC::kNC;  case CC::kNC: return CC::kC;
            case CC::kE:  return CC::kNE;  case CC::kNE: return CC::kE;
            case CC::kBE: return CC::kA;   case CC::kA:  return CC::kBE;
            case CC::kS:  return CC::kNS;  case CC::kNS: return CC::kS;
            case CC::kP:  return CC::kNP;  case CC::kNP: return CC::kP;
            case CC::kL:  return CC::kGE;  case CC::kGE: return CC::kL;
            case CC::kLE: return CC::kG;   case CC::kG:  return CC::kLE;
        }
        return c;
    }

    // The x86 condition equivalent to an ARM cond, if the flags are currently
    // in RFLAGS. On success CONSUMES the pending flags (they are never
    // materialized).
    bool take_fused_cond(uint32_t cond, asmjit::x86::CondCode& out) {
        if (!flags_pending) return false;
        if ((cond & 0xEu) == 0xEu) return false;          // AL/NV

        // EQ NE CS CC MI PL VS VC HI LS GE LT GT LE
        static const uint8_t sub_map[14] = {
            0x04, 0x05, 0x03, 0x02, 0x08, 0x09, 0x00, 0x01,
            0x07, 0x06, 0x0D, 0x0C, 0x0F, 0x0E,
        };
        // After ADD: ARM C == CF (after SUB it is inverted, hence the
        // CS/CC/HI/LS differences).
        static const uint8_t add_map[14] = {
            0x04, 0x05, 0x02, 0x03, 0x08, 0x09, 0x00, 0x01,
            0xFF, 0xFF, 0x0D, 0x0C, 0x0F, 0x0E,
        };
        // After logic: CF=OF=0. GE->NS, LT->S, GT->G, LE->LE; HI/LS/CS/VS/VC
        // cannot be expressed as a single jcc — report "no fusion".
        static const uint8_t log_map[14] = {
            0x04, 0x05, 0xFF, 0xFF, 0x08, 0x09, 0xFF, 0xFF,
            0xFF, 0xFF, 0x09, 0x08, 0x0F, 0x0E,
        };
        const uint8_t* m = (flag_src == FlagSrc::kFromAdd) ? add_map
                       : (flag_src == FlagSrc::kFromSub)  ? sub_map
                                                          : log_map;
        uint8_t v = m[cond];
        if (v == 0xFF) return false;
        out = (asmjit::x86::CondCode)v;
        flags_pending = false;
        return true;
    }

    // ---- context field access ----
    static int32_t roff(int r) { return (int32_t)(offsetof(GuestCpu, x) + r * 8); }
    asmjit::x86::Mem reg_mem(int r) { return asmjit::x86::qword_ptr(CTX, roff(r)); }
    asmjit::x86::Mem sp_mem() { return asmjit::x86::qword_ptr(CTX, (int32_t)offsetof(GuestCpu, sp)); }
    asmjit::x86::Mem pc_mem() { return asmjit::x86::qword_ptr(CTX, (int32_t)offsetof(GuestCpu, pc)); }
    asmjit::x86::Mem fN() { return asmjit::x86::byte_ptr(CTX, (int32_t)offsetof(GuestCpu, fN)); }
    asmjit::x86::Mem fZ() { return asmjit::x86::byte_ptr(CTX, (int32_t)offsetof(GuestCpu, fZ)); }
    asmjit::x86::Mem fC() { return asmjit::x86::byte_ptr(CTX, (int32_t)offsetof(GuestCpu, fC)); }
    asmjit::x86::Mem fV() { return asmjit::x86::byte_ptr(CTX, (int32_t)offsetof(GuestCpu, fV)); }

    // ---- guest register read/write ----
    // Xn/Wn, where number 31 means the zero register XZR/WZR.
    // Pinned x0/x29/x30 travel in host registers without touching memory.
    void ld_r(const asmjit::x86::Gp& dst, int r, bool sf) {
        if (r == 31) { dirty_flags(); a.xor_(dst.r32(), dst.r32()); return; }
        const asmjit::x86::Gp& pin = pin_of(r);
        if (pin.is_valid()) {
            if (dst.id() == pin.id()) return;            // already in place
            if (sf) a.mov(dst.r64(), pin);
            else    a.mov(dst.r32(), pin.r32());
            return;
        }
        if (sf) a.mov(dst.r64(), reg_mem(r));
        else    a.mov(dst.r32(), asmjit::x86::dword_ptr(CTX, roff(r)));
    }
    // Wn store: any 32-bit x86 operation has already zeroed the upper half
    // of src (an ISA property), so store 64 bits right away — no mov r32,r32.
    void st_r(int r, const asmjit::x86::Gp& src, bool sf) {
        if (r == 31) return;
        base_invalidate(r);
        const asmjit::x86::Gp& pin = pin_of(r);
        if (pin.is_valid()) {
            if (src.id() == pin.id()) return;            // value already in the pin
            if (sf) a.mov(pin, src.r64());
            else    a.mov(pin.r32(), src.r32());         // mov r32 zeroes the pin's upper half
            return;
        }
        a.mov(reg_mem(r), src.r64());
    }
    // Addressing forms: number 31 means SP.
    void ld_rsp(const asmjit::x86::Gp& dst, int r, bool sf) {
        if (r != 31) { ld_r(dst, r, sf); return; }
        if (dst.id() == PIN_SP.id()) return;
        if (sf) a.mov(dst.r64(), PIN_SP);
        else    a.mov(dst.r32(), PIN_SP.r32());
    }
    void st_rsp(int r, const asmjit::x86::Gp& src, bool sf) {
        if (r != 31) { st_r(r, src, sf); return; }
        base_invalidate(31);
        // See st_r: 32-bit operations have already zeroed the upper half of src.
        if (src.id() == PIN_SP.id()) return;
        a.mov(PIN_SP, src.r64());
    }

    // ---- arithmetic with a 64-bit constant ----
    enum class Alu { kAdd, kSub, kCmp, kAnd, kOr, kXor };

    // Base-register cache: rsi holds the value of guest register base_cached
    // since the last load. Invalidated by any write to that guest register
    // (st_r/st_rsp) and by any change to rsi itself.
    int base_cached = -1;
    void base_load(int rn) {
        if (base_cached == rn) return;
        ld_rsp(asmjit::x86::rsi, rn, true);
        base_cached = rn;
    }
    void base_invalidate(int r) { if (r == base_cached) base_cached = -1; }
    void base_clobber_rsi()    { base_cached = -1; }

    void alu_rr(Alu op, const asmjit::x86::Gp& d, const asmjit::x86::Gp& s) {
        dirty_flags();
        switch (op) {
            case Alu::kAdd: a.add(d, s); break;
            case Alu::kSub: a.sub(d, s); break;
            case Alu::kCmp: a.cmp(d, s); break;
            case Alu::kAnd: a.and_(d, s); break;
            case Alu::kOr:  a.or_(d, s);  break;
            case Alu::kXor: a.xor_(d, s); break;
        }
    }
    void alu_ri(Alu op, const asmjit::x86::Gp& dst, uint64_t v, bool sf) {
        asmjit::x86::Gp d = sf ? dst.r64() : dst.r32();
        dirty_flags();
        if (!sf) {
            asmjit::Imm i((int32_t)(uint32_t)v);
            switch (op) {
                case Alu::kAdd: a.add(d, i); break;
                case Alu::kSub: a.sub(d, i); break;
                case Alu::kCmp: a.cmp(d, i); break;
                case Alu::kAnd: a.and_(d, i); break;
                case Alu::kOr:  a.or_(d, i);  break;
                case Alu::kXor: a.xor_(d, i); break;
            }
            return;
        }
        if ((int64_t)v >= INT32_MIN && (int64_t)v <= INT32_MAX) {
            asmjit::Imm i((int32_t)(int64_t)v);
            switch (op) {
                case Alu::kAdd: a.add(d, i); break;
                case Alu::kSub: a.sub(d, i); break;
                case Alu::kCmp: a.cmp(d, i); break;
                case Alu::kAnd: a.and_(d, i); break;
                case Alu::kOr:  a.or_(d, i);  break;
                case Alu::kXor: a.xor_(d, i); break;
            }
        } else {
            // r11 is taken by the x8 pin — carry big constants via rdx.
            a.mov(asmjit::x86::rdx, asmjit::Imm(v));
            alu_rr(op, d, asmjit::x86::rdx);
        }
    }

    // ---- flags ----
    // ARM C for subtraction is "no borrow", i.e. the inversion of x86 CF.
    void flags_sub() { a.sets(fN()); a.sete(fZ()); a.setnc(fC()); a.seto(fV()); }
    void flags_add() { a.sets(fN()); a.sete(fZ()); a.setc(fC());  a.seto(fV()); }
    void flags_logic() { a.sets(fN()); a.sete(fZ()); a.mov(fC(), asmjit::Imm(0)); a.mov(fV(), asmjit::Imm(0)); }

    // Lazy flags: after an S-instruction the guest NZCV equivalently live in
    // x86 RFLAGS. A fused consumer (jcc/setcc/cmovcc) eats them directly; if
    // flag-clobbering code comes next, NZCV is first materialized into the
    // context memory (spill).
    bool flags_pending = false;   // guest NZCV currently in RFLAGS, memory copy stale

    void spill_flags() {
        switch (flag_src) {
            case FlagSrc::kFromSub:   flags_sub();   break;
            case FlagSrc::kFromAdd:   flags_add();   break;
            case FlagSrc::kFromLogic: flags_logic(); break;
            default: break;
        }
        flags_pending = false;
    }
    // Called before any x86 operation that clobbers RFLAGS.
    void dirty_flags() { if (flags_pending) spill_flags(); }
    // Start of a guest S-instruction: the old flags are overwritten,
    // no pending materialization is needed.
    void begin_flag_def() { flags_pending = false; }
    // End of an S-instruction: the new flags are in RFLAGS.
    void mark_flags(FlagSrc src) { flag_src = src; flags_pending = true; }

    // Evaluates an ARM condition (4 bits) into r8b as 0/1.
    void emit_cond(uint32_t cond) {
        uint32_t basec = cond >> 1;
        bool invert = (cond & 1u) && cond != 0x0F;   // 1111 (NV) in AArch64 also means "always"
        switch (basec) {
            case 0: a.mov(asmjit::x86::r8b, fZ()); break;                     // EQ: Z
            case 1: a.mov(asmjit::x86::r8b, fC()); break;                     // CS: C
            case 2: a.mov(asmjit::x86::r8b, fN()); break;                     // MI: N
            case 3: a.mov(asmjit::x86::r8b, fV()); break;                     // VS: V
            case 4:                                                    // HI: C && !Z
                a.mov(asmjit::x86::r8b, fC()); a.mov(asmjit::x86::r9b, fZ());
                a.xor_(asmjit::x86::r9b, asmjit::Imm(1)); a.and_(asmjit::x86::r8b, asmjit::x86::r9b); break;
            case 5:                                                    // GE: N == V
                a.mov(asmjit::x86::r8b, fN()); a.mov(asmjit::x86::r9b, fV());
                a.xor_(asmjit::x86::r8b, asmjit::x86::r9b); a.xor_(asmjit::x86::r8b, asmjit::Imm(1)); break;
            case 6:                                                    // GT: !Z && N == V
                a.mov(asmjit::x86::r8b, fN()); a.mov(asmjit::x86::r9b, fV());
                a.xor_(asmjit::x86::r8b, asmjit::x86::r9b); a.xor_(asmjit::x86::r8b, asmjit::Imm(1));
                a.mov(asmjit::x86::r9b, fZ()); a.xor_(asmjit::x86::r9b, asmjit::Imm(1));
                a.and_(asmjit::x86::r8b, asmjit::x86::r9b); break;
            default: a.mov(asmjit::x86::r8b, asmjit::Imm(1)); break;          // AL/NV
        }
        if (invert) a.xor_(asmjit::x86::r8b, asmjit::Imm(1));
    }

    // ---- block exits ----
    // Exit to a known address: a direct jmp rel32. Until the target is
    // compiled, rel32 == 0 and execution falls into the miss path (dispatcher
    // with pc in the context); once the target exists, Emu::block_for
    // patches rel32 to its body.
    void exit_const(uint64_t target) {
        dirty_flags();                               // nothing live beyond this point
        if (target < THUNK_BASE) {
            asmjit::Label site = a.new_label();
            a.jmp(site);                             // E9 + rel32 (zero for now)
            a.bind(site);
            patches.push_back({(uint32_t)(a.offset() - 4), target});
        }
        a.mov(asmjit::x86::rax, asmjit::Imm(target));
        a.mov(pc_mem(), asmjit::x86::rax);
        a.jmp(epilogue);
    }
    // Register exit (br/blr/ret): an inline direct-mapped pc -> body cache.
    // Index = (pc & (DMC_MASK << 2)); scale 2 in addressing gives *8 like the tag.
    void exit_reg(const asmjit::x86::Gp& r) {
        dirty_flags();
        asmjit::Label miss = a.new_label();
        a.mov(asmjit::x86::rcx, r.r64());
        a.and_(asmjit::x86::ecx, asmjit::Imm(Emu::DMC_MASK << 2));   // 32-bit and also clears the upper half of rcx
        a.cmp(asmjit::x86::qword_ptr(CHAIN, asmjit::x86::rcx, 1, (int32_t)Emu::DMC_TAG_OFF), r.r64());
        a.jne(miss);
        a.jmp(asmjit::x86::qword_ptr(CHAIN, asmjit::x86::rcx, 1, (int32_t)Emu::DMC_BODY_OFF));
        a.bind(miss);
        a.mov(pc_mem(), r.r64());
        a.jmp(epilogue);
    }
    void exit_fault(uint64_t pc, ExitReason why, uint32_t info = 0) {
        dirty_flags();                               // leaving the block — materialize flags
        a.mov(asmjit::x86::rax, asmjit::Imm(pc));
        a.mov(pc_mem(), asmjit::x86::rax);
        a.mov(asmjit::x86::dword_ptr(CTX, (int32_t)offsetof(GuestCpu, exit_reason)), asmjit::Imm((uint32_t)why));
        a.mov(asmjit::x86::dword_ptr(CTX, (int32_t)offsetof(GuestCpu, exit_info)), asmjit::Imm(info));
        a.jmp(epilogue);
    }

    // Guest memory operand [BIAS + addr]. Bounds are guarded by guard pages
    // and the signal handler — there are no inline checks in the hot path.
    asmjit::x86::Mem mem_ref(const asmjit::x86::Gp& addr, uint32_t size, uint64_t /*ipc*/) {
        return mem_at(addr, 0, size);
    }
    asmjit::x86::Mem mem_at(const asmjit::x86::Gp& idx, int32_t disp, uint32_t size) {
        switch (size) {
            case 1:  return asmjit::x86::byte_ptr(BIAS, idx, 0, disp);
            case 2:  return asmjit::x86::word_ptr(BIAS, idx, 0, disp);
            case 4:  return asmjit::x86::dword_ptr(BIAS, idx, 0, disp);
            default: return asmjit::x86::qword_ptr(BIAS, idx, 0, disp);
        }
    }

    Step translate(uint32_t insn, uint64_t pc);
    Step tr_dp_imm(uint32_t insn, uint64_t pc);
    Step tr_branch(uint32_t insn, uint64_t pc);
    Step tr_ldst(uint32_t insn, uint64_t pc);
    Step tr_dp_reg(uint32_t insn, uint64_t pc);

    // An unconditional B inside the window does not end the block: the
    // translator continues at the target (block merging). See tr_branch.
    uint64_t jump_to   = 0;
    bool     has_jump  = false;

    // Register names for logs: 31 is either XZR/WZR or SP.
    static const char* rn_z(int r, bool sf) {
        static thread_local char buf[4][8]; static thread_local int k = 0;
        k = (k + 1) & 3;
        if (r == 31) return sf ? "xzr" : "wzr";
        snprintf(buf[k], sizeof(buf[k]), "%c%d", sf ? 'x' : 'w', r);
        return buf[k];
    }
    static const char* rn_sp(int r, bool sf) {
        static thread_local char buf[4][8]; static thread_local int k = 0;
        k = (k + 1) & 3;
        if (r == 31) return sf ? "sp" : "wsp";
        snprintf(buf[k], sizeof(buf[k]), "%c%d", sf ? 'x' : 'w', r);
        return buf[k];
    }
};
