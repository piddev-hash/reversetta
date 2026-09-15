// ============================================================================
//  translator.cpp — ARM64 -> x86-64 basic block code generation
//
//  One basic block per Emu::block_for() call. See translator.h for the
//  register pinning / lazy flags / block chaining design notes.
// ============================================================================

#include "translator.h"

#include <cstdio>

using namespace asmjit;

// ---------------------------------------------------------------------------
//  Whole-block assembly
// ---------------------------------------------------------------------------
bool BlockBuilder::build() {
    epilogue = a.new_label();
    Label body = a.new_label();

    // Prologue. 7 pushes + the return address = 64 bytes, RSP stays 16-aligned.
    a.push(x86::rbx);
    a.push(x86::rbp);
    a.push(x86::r10);
    a.push(x86::r12);
    a.push(x86::r13);
    a.push(x86::r14);
    a.push(x86::r15);
    a.mov(CTX, x86::rdi);
    a.mov(BIAS,  x86::qword_ptr(CTX, (int32_t)offsetof(GuestCpu, mem_bias)));
    a.mov(CHAIN, Imm((uint64_t)e.chain_tbl.data()));

    // Self-registration: the prologue publishes the block body address in the
    // DMC for indirect branches (br/blr/ret). The stores sit BEFORE the body
    // label — chained entries skip them. Direct entries are patched by
    // Emu::block_for (jmp rel32).
    a.lea(x86::r11, x86::ptr(body));
    a.mov(x86::rax, Imm(pc_start));
    uint32_t h = (uint32_t)((pc_start >> 2) & Emu::DMC_MASK);
    a.mov(x86::qword_ptr(CHAIN, (int32_t)(Emu::DMC_TAG_OFF + (uint64_t)h * 8)), x86::rax);
    a.mov(x86::qword_ptr(CHAIN, (int32_t)(Emu::DMC_BODY_OFF + (uint64_t)h * 8)), x86::r11);

    // Pins: guest x0/x8/x29/x30/sp enter host registers. Chained transitions
    // (jmp rel32 / DMC) skip these movs — the values stay alive.
    a.mov(PIN_X0,  x86::qword_ptr(CTX, roff(0)));
    a.mov(PIN_X8,  x86::qword_ptr(CTX, roff(8)));
    a.mov(PIN_X29, x86::qword_ptr(CTX, roff(29)));
    a.mov(PIN_X30, x86::qword_ptr(CTX, roff(30)));
    a.mov(PIN_SP,  x86::qword_ptr(CTX, (int32_t)offsetof(GuestCpu, sp)));
    a.bind(body);
    body_off = a.offset();

    uint64_t pc = pc_start;
    for (int n = 0; n < MAX_BLOCK_INSNS; n++) {
        if (!e.in_range(pc, 4)) { exit_fault(pc, kExitMemFault); break; }
        uint32_t insn = e.read<uint32_t>(pc);

        Step st = translate(insn, pc);
        e.insn_translated++;

        if (st == Step::kBad) {
            e.log(1, "    [JIT] 0x%016llx: %08x  <unknown instruction>\n",
                  (unsigned long long)pc, insn);
            exit_fault(pc, kExitUndefInsn, insn);
            break;
        }
        if (st == Step::kEnd) break;

        pc += 4;
        if (has_jump) { pc = jump_to; has_jump = false; }   // fell through a B
        if (n == MAX_BLOCK_INSNS - 1) exit_const(pc);  // block overflow — cut here
    }

    // Epilogue: the single return point into the dispatcher. Spill the pins
    // into GuestCpu — host code (thunks, syscalls) reads state from there.
    a.bind(epilogue);
    a.mov(x86::qword_ptr(CTX, roff(0)),  PIN_X0);
    a.mov(x86::qword_ptr(CTX, roff(8)),  PIN_X8);
    a.mov(x86::qword_ptr(CTX, roff(29)), PIN_X29);
    a.mov(x86::qword_ptr(CTX, roff(30)), PIN_X30);
    a.mov(x86::qword_ptr(CTX, (int32_t)offsetof(GuestCpu, sp)), PIN_SP);
    a.pop(x86::r15);
    a.pop(x86::r14);
    a.pop(x86::r13);
    a.pop(x86::r12);
    a.pop(x86::r10);
    a.pop(x86::rbp);
    a.pop(x86::rbx);
    a.ret();
    return true;
}

// ---------------------------------------------------------------------------
//  Top-level decode (by bits 28..25, as in the ARM ARM)
// ---------------------------------------------------------------------------
Step BlockBuilder::translate(uint32_t insn, uint64_t pc) {
    uint32_t op0 = bits(insn, 28, 25);

    if ((op0 & 0b1110) == 0b1000) return tr_dp_imm(insn, pc);   // 100x
    if ((op0 & 0b1110) == 0b1010) return tr_branch(insn, pc);   // 101x
    if ((op0 & 0b0101) == 0b0100) return tr_ldst(insn, pc);     // x1x0
    if ((op0 & 0b0111) == 0b0101) return tr_dp_reg(insn, pc);   // x101
    return Step::kBad;                                          // x111 — SIMD/FP
}

// ---------------------------------------------------------------------------
//  Data Processing -- Immediate
// ---------------------------------------------------------------------------
Step BlockBuilder::tr_dp_imm(uint32_t insn, uint64_t pc) {
    bool sf = bit(insn, 31);
    int rd = (int)bits(insn, 4, 0);
    int rn = (int)bits(insn, 9, 5);
    uint32_t grp = bits(insn, 25, 23);

    switch (grp) {
    case 0b000: case 0b001: {   // ADR / ADRP
        uint64_t immlo = bits(insn, 30, 29);
        uint64_t immhi = bits(insn, 23, 5);
        int64_t  imm   = sext((immhi << 2) | immlo, 21);
        uint64_t res   = bit(insn, 31) ? ((pc & ~0xFFFull) + ((uint64_t)imm << 12))
                                       : (pc + (uint64_t)imm);
        e.log(1, "    [JIT] %s X%d, #0x%llx\n", bit(insn,31) ? "ADRP" : "ADR", rd,
              (unsigned long long)res);
        a.mov(x86::rax, Imm(res));
        st_r(rd, x86::rax, true);
        return Step::kNext;
    }
    case 0b010: case 0b011: {   // ADD/SUB (immediate)
        bool is_sub = bit(insn, 30), setf = bit(insn, 29);
        uint32_t sh = bits(insn, 23, 22);
        uint64_t imm12 = bits(insn, 21, 10);
        if (sh == 1) imm12 <<= 12;
        else if (sh != 0) return Step::kBad;

        e.log(1, "    [JIT] %s%s %s, %s, #%llu\n", is_sub ? "SUB" : "ADD",
              setf ? "S" : "", setf ? rn_z(rd, sf) : rn_sp(rd, sf), rn_sp(rn, sf),
              (unsigned long long)imm12);

        // Rd == Rn (SP included): the result goes straight into the pin, or
        // into memory with a single add/sub qword[ctx], imm — no detour
        // through a register.
        if (rd == rn) {
            const x86::Gp& pin = (rn == 31) ? PIN_SP : pin_of(rn);
            base_invalidate(rn);                      // rn overwritten directly
            if (pin.is_valid()) {
                if (setf) begin_flag_def();
                alu_ri(is_sub ? Alu::kSub : Alu::kAdd, pin, imm12, sf);
                if (setf) mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd);
                return Step::kNext;
            }
            if (sf) {
                x86::Mem dstm = reg_mem(rn);
                if (setf) begin_flag_def(); else dirty_flags();
                if (is_sub) a.sub(dstm, Imm((int32_t)(uint64_t)imm12));
                else        a.add(dstm, Imm((int32_t)(uint64_t)imm12));
                if (setf) mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd);
                return Step::kNext;
            }
            // 32-bit non-pinned — the generic path below
        }
        // SUBS XZR, Xn, #imm == CMP: compare against the pin or memory.
        if (setf && rd == 31) {
            const x86::Gp& pin = (rn == 31) ? PIN_SP : pin_of(rn);
            if (pin.is_valid()) {
                begin_flag_def();
                alu_ri(Alu::kCmp, pin, imm12, sf);
                mark_flags(FlagSrc::kFromSub);
                return Step::kNext;
            }
            if (sf) {
                begin_flag_def();
                a.cmp(reg_mem(rn), Imm((int32_t)(uint64_t)imm12));
                mark_flags(FlagSrc::kFromSub);
                return Step::kNext;
            }
        }

        // ADD (non-S) with a pinned destination and a SP/pin source: one lea.
        if (!is_sub && !setf && rd != rn) {
            const x86::Gp& pd = (rd == 31) ? PIN_SP : pin_of(rd);
            const x86::Gp& ps = (rn == 31) ? PIN_SP : pin_of(rn);
            if (pd.is_valid() && ps.is_valid() && (int64_t)imm12 <= INT32_MAX) {
                dirty_flags();                        // lea preserves RFLAGS, but keep the order
                base_invalidate(rd);                  // rd overwritten directly
                a.lea(pd, x86::ptr(ps, (int32_t)(int64_t)imm12));
                return Step::kNext;
            }
        }

        ld_rsp(x86::rax, rn, sf);
        if (setf) begin_flag_def();
        alu_ri(is_sub ? Alu::kSub : Alu::kAdd, x86::rax, imm12, sf);
        if (setf) { mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd);
                    st_r(rd, x86::rax, sf); }
        else      { st_rsp(rd, x86::rax, sf); }
        return Step::kNext;
    }
    case 0b100: {               // Logical (immediate)
        uint32_t opc = bits(insn, 30, 29);
        uint32_t N = bit(insn, 22), immr = bits(insn, 21, 16), imms = bits(insn, 15, 10);
        int datasize = sf ? 64 : 32;
        if (!sf && N) return Step::kBad;
        uint64_t wmask, tmask;
        if (!decode_bit_masks(N, imms, immr, true, datasize, wmask, tmask)) return Step::kBad;
        if (!sf) wmask &= 0xFFFFFFFFull;

        static const char* nm[] = {"AND", "ORR", "EOR", "ANDS"};
        e.log(1, "    [JIT] %s %s%d, %s%d, #0x%llx\n", nm[opc], sf ? "X" : "W", rd,
              sf ? "X" : "W", rn, (unsigned long long)wmask);

        ld_r(x86::rax, rn, sf);
        Alu op = (opc == 0 || opc == 3) ? Alu::kAnd : (opc == 1 ? Alu::kOr : Alu::kXor);
        if (opc == 3) begin_flag_def();
        alu_ri(op, x86::rax, wmask, sf);
        if (opc == 3) { mark_flags(FlagSrc::kFromLogic); st_r(rd, x86::rax, sf); }
        else          { st_rsp(rd, x86::rax, sf); }   // Rd == 31 means SP here
        return Step::kNext;
    }
    case 0b101: {               // Move wide (immediate)
        uint32_t opc = bits(insn, 30, 29);
        uint32_t hw = bits(insn, 22, 21);
        uint64_t imm16 = bits(insn, 20, 5);
        if (!sf && hw > 1) return Step::kBad;
        unsigned shift = hw * 16;

        if (opc == 3) {          // MOVK — only its own field is rewritten
            uint64_t field = imm16 << shift;
            uint64_t keep  = ~(0xFFFFull << shift);
            e.log(1, "    [JIT] MOVK %s%d, #%llu, LSL #%u\n", sf ? "X" : "W", rd,
                  (unsigned long long)imm16, shift);
            ld_r(x86::rax, rd, sf);
            alu_ri(Alu::kAnd, x86::rax, keep, sf);
            alu_ri(Alu::kOr,  x86::rax, field, sf);
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        if (opc != 0 && opc != 2) return Step::kBad;
        uint64_t val = imm16 << shift;
        if (opc == 0) val = ~val;                       // MOVN
        if (!sf) val &= 0xFFFFFFFFull;
        e.log(1, "    [JIT] %s %s%d, #0x%llx\n", opc == 0 ? "MOVN" : "MOVZ",
              sf ? "X" : "W", rd, (unsigned long long)val);
        a.mov(x86::rax, Imm(val));
        st_r(rd, x86::rax, sf);
        return Step::kNext;
    }
    case 0b110: {               // Bitfield: SBFM / BFM / UBFM
        uint32_t opc = bits(insn, 30, 29);
        uint32_t N = bit(insn, 22), immr = bits(insn, 21, 16), imms = bits(insn, 15, 10);
        int datasize = sf ? 64 : 32;
        if (sf != (bool)N) return Step::kBad;
        uint64_t wmask, tmask;
        if (!decode_bit_masks(N, imms, immr, false, datasize, wmask, tmask)) return Step::kBad;
        uint32_t R = immr & (sf ? 63u : 31u);
        uint64_t m = wmask & tmask;
        if (!sf) m &= 0xFFFFFFFFull;

        static const char* nm[] = {"SBFM", "BFM", "UBFM", "?"};
        e.log(1, "    [JIT] %s %s%d, %s%d, #%u, #%u\n", nm[opc], sf ? "X" : "W", rd,
              sf ? "X" : "W", rn, immr, imms);

        if (opc == 2) {                                  // UBFM = ror & (wmask & tmask)
            ld_r(x86::rax, rn, sf);
            if (R) { dirty_flags(); a.ror(sf ? x86::rax : x86::eax, Imm(R)); }
            alu_ri(Alu::kAnd, x86::rax, m, sf);
            st_r(rd, x86::rax, sf);
        } else if (opc == 0) {                           // SBFM: sign-fill the field
            uint32_t S = imms & (sf ? 63u : 31u);
            ld_r(x86::rax, rn, sf);
            a.mov(sf ? x86::rcx : x86::ecx, sf ? x86::rax : x86::eax);
            if (S) { dirty_flags(); a.shr(sf ? x86::rcx : x86::ecx, Imm(S)); }
            alu_ri(Alu::kAnd, x86::rcx, 1, sf);
            a.neg(sf ? x86::rcx : x86::ecx);             // 0 -> 0, 1 -> all ones
            alu_ri(Alu::kAnd, x86::rcx, ~m, sf);
            if (R) a.ror(sf ? x86::rax : x86::eax, Imm(R));
            alu_ri(Alu::kAnd, x86::rax, m, sf);
            a.or_(sf ? x86::rax : x86::eax, sf ? x86::rcx : x86::ecx);
            st_r(rd, x86::rax, sf);
        } else if (opc == 1) {                           // BFM: field insertion
            ld_r(x86::rax, rn, sf);
            if (R) { dirty_flags(); a.ror(sf ? x86::rax : x86::eax, Imm(R)); }
            alu_ri(Alu::kAnd, x86::rax, m, sf);
            ld_r(x86::rcx, rd, sf);
            alu_ri(Alu::kAnd, x86::rcx, ~m, sf);
            a.or_(sf ? x86::rax : x86::eax, sf ? x86::rcx : x86::ecx);
            st_r(rd, x86::rax, sf);
        } else return Step::kBad;
        return Step::kNext;
    }
    case 0b111: {               // EXTR (including rotate by a constant)
        if (bits(insn, 30, 29) != 0 || bit(insn, 21) != 0) return Step::kBad;
        if (sf != (bool)bit(insn, 22)) return Step::kBad;
        int rm = (int)bits(insn, 20, 16);
        uint32_t lsb = bits(insn, 15, 10);
        e.log(1, "    [JIT] EXTR %s%d, %s%d, %s%d, #%u\n", sf ? "X" : "W", rd,
              sf ? "X" : "W", rn, sf ? "X" : "W", rm, lsb);
        ld_r(x86::rax, rm, sf);                          // low part
        ld_r(x86::rcx, rn, sf);                          // high part
        if (lsb) { dirty_flags(); a.shrd(sf ? x86::rax : x86::eax, sf ? x86::rcx : x86::ecx, Imm(lsb)); }
        st_r(rd, x86::rax, sf);
        return Step::kNext;
    }
    }
    return Step::kBad;
}

// ---------------------------------------------------------------------------
//  Branches, Exception Generating and System
// ---------------------------------------------------------------------------
Step BlockBuilder::tr_branch(uint32_t insn, uint64_t pc) {
    // B / BL
    if ((insn & 0x7C000000u) == 0x14000000u) {
        uint64_t target = pc + (uint64_t)(sext(bits(insn, 25, 0), 26) * 4);
        bool link = bit(insn, 31);
        e.log(1, "    [JIT] %s 0x%llx\n", link ? "BL" : "B", (unsigned long long)target);
        if (link) { a.mov(x86::rax, Imm(pc + 4)); st_r(30, x86::rax, true); exit_const(target); return Step::kEnd; }
        // Unconditional B: instead of ending the block, keep translating at
        // the target.
        if (target < THUNK_BASE && e.in_range(target, 4)) {
            jump_to = target; has_jump = true;
            return Step::kNext;
        }
        exit_const(target);
        return Step::kEnd;
    }
    // CBZ / CBNZ
    if ((insn & 0x7E000000u) == 0x34000000u) {
        bool sf = bit(insn, 31), nz = bit(insn, 24);
        int rt = (int)bits(insn, 4, 0);
        uint64_t target = pc + (uint64_t)(sext(bits(insn, 23, 5), 19) * 4);
        e.log(1, "    [JIT] %s %s%d, 0x%llx\n", nz ? "CBNZ" : "CBZ", sf ? "X" : "W",
              rt, (unsigned long long)target);
        ld_r(x86::rax, rt, sf);
        dirty_flags();                                  // the test below clobbers RFLAGS
        a.test(sf ? x86::rax : x86::eax, sf ? x86::rax : x86::eax);
        Label skip = a.new_label();
        if (nz) a.jz(skip); else a.jnz(skip);
        exit_const(target);
        a.bind(skip);
        return Step::kNext;
    }
    // TBZ / TBNZ
    if ((insn & 0x7E000000u) == 0x36000000u) {
        uint32_t pos = (bit(insn, 31) << 5) | bits(insn, 23, 19);
        bool nz = bit(insn, 24);
        int rt = (int)bits(insn, 4, 0);
        uint64_t target = pc + (uint64_t)(sext(bits(insn, 18, 5), 14) * 4);
        e.log(1, "    [JIT] %s X%d, #%u, 0x%llx\n", nz ? "TBNZ" : "TBZ", rt, pos,
              (unsigned long long)target);
        ld_r(x86::rax, rt, true);
        a.bt(x86::rax, Imm(pos));
        Label skip = a.new_label();
        if (nz) a.jnc(skip); else a.jc(skip);
        exit_const(target);
        a.bind(skip);
        return Step::kNext;
    }
    // B.cond
    if ((insn & 0xFF000010u) == 0x54000000u) {
        uint32_t cond = bits(insn, 3, 0);
        uint64_t target = pc + (uint64_t)(sext(bits(insn, 23, 5), 19) * 4);
        e.log(1, "    [JIT] B.cond(%u) 0x%llx\n", cond, (unsigned long long)target);
        // Fast path: the flags are still in RFLAGS — a single jcc instead of
        // a round-trip through memory.
        x86::CondCode cc;
        Label skip = a.new_label();
        if (take_fused_cond(cond, cc)) {
            a.j(inv_cc(cc), skip);
        } else {
            dirty_flags();
            emit_cond(cond);
            a.test(x86::r8b, x86::r8b);
            a.jz(skip);
        }
        exit_const(target);
        a.bind(skip);
        return Step::kNext;
    }
    // BR / BLR / RET
    if ((insn & 0xFFFFFC1Fu) == 0xD61F0000u ||
        (insn & 0xFFFFFC1Fu) == 0xD63F0000u ||
        (insn & 0xFFFFFC1Fu) == 0xD65F0000u) {
        int rn = (int)bits(insn, 9, 5);
        bool is_blr = (insn & 0xFFFFFC1Fu) == 0xD63F0000u;
        bool is_ret = (insn & 0xFFFFFC1Fu) == 0xD65F0000u;
        e.log(1, "    [JIT] %s X%d\n", is_ret ? "RET" : (is_blr ? "BLR" : "BR"), rn);
        ld_r(x86::rax, rn, true);
        if (is_blr) { a.mov(x86::rcx, Imm(pc + 4)); st_r(30, x86::rcx, true); }
        exit_reg(x86::rax);
        return Step::kEnd;
    }
    // SVC #imm — leave to the dispatcher, it has a small syscall handler
    if ((insn & 0xFFE0001Fu) == 0xD4000001u) {
        uint32_t imm16 = bits(insn, 20, 5);
        e.log(1, "    [JIT] SVC #%u\n", imm16);
        dirty_flags();                                  // leaving the block — materialize flags
        a.mov(x86::rax, Imm(pc + 4));
        a.mov(pc_mem(), x86::rax);
        a.mov(x86::dword_ptr(CTX, (int32_t)offsetof(GuestCpu, exit_reason)), Imm((uint32_t)kExitSyscall));
        a.mov(x86::dword_ptr(CTX, (int32_t)offsetof(GuestCpu, exit_info)), Imm(imm16));
        a.jmp(epilogue);
        return Step::kEnd;
    }
    // HINTs (NOP/YIELD/...) and barriers (DMB/DSB/ISB) — no-ops for us
    if ((insn & 0xFFFFF01Fu) == 0xD503201Fu || (insn & 0xFFFFF01Fu) == 0xD503301Fu) {
        e.log(1, "    [JIT] NOP/barrier\n");
        return Step::kNext;
    }
    return Step::kBad;
}

// ---------------------------------------------------------------------------
//  Loads and Stores
// ---------------------------------------------------------------------------
Step BlockBuilder::tr_ldst(uint32_t insn, uint64_t pc) {
    uint32_t V = bit(insn, 26);

    // ---- LDR (literal) ----
    if (bits(insn, 29, 27) == 0b011 && bits(insn, 25, 24) == 0b00) {
        if (V) return Step::kBad;                        // FP variant unsupported
        uint32_t opc = bits(insn, 31, 30);
        int rt = (int)bits(insn, 4, 0);
        uint64_t addr = pc + (uint64_t)(sext(bits(insn, 23, 5), 19) * 4);
        if (opc == 3) return Step::kNext;                // PRFM literal
        e.log(1, "    [JIT] LDR (literal) X%d, 0x%llx\n", rt, (unsigned long long)addr);
        base_clobber_rsi();                             // rsi = literal address
        a.mov(x86::rsi, Imm(addr));
        if (opc == 0)      { auto m = mem_ref(x86::rsi, 4, pc); a.mov(x86::eax, m);   st_r(rt, x86::rax, false); }
        else if (opc == 1) { auto m = mem_ref(x86::rsi, 8, pc); a.mov(x86::rax, m);   st_r(rt, x86::rax, true);  }
        else               { auto m = mem_ref(x86::rsi, 4, pc); a.movsxd(x86::rax, m); st_r(rt, x86::rax, true); }
        return Step::kNext;
    }

    // ---- LDP / STP ----
    if (bits(insn, 29, 27) == 0b101 && V == 0) {
        uint32_t kind = bits(insn, 25, 23);   // 000 NT, 001 post, 010 offset, 011 pre
        if (kind > 3) return Step::kBad;
        bool load = bit(insn, 22);
        uint32_t opc = bits(insn, 31, 30);
        bool is64 = false, sxtw = false;
        if (opc == 0)      { is64 = false; }
        else if (opc == 1) { if (!load) return Step::kBad; is64 = true; sxtw = true; }  // LDPSW
        else if (opc == 2) { is64 = true; }
        else return Step::kBad;

        int scale = (opc == 2) ? 8 : 4;
        int64_t off = sext(bits(insn, 21, 15), 7) * scale;
        int rt2 = (int)bits(insn, 14, 10), rn = (int)bits(insn, 9, 5), rt = (int)bits(insn, 4, 0);
        bool wback = (kind == 1 || kind == 3);
        bool post  = (kind == 1);

        e.log(1, "    [JIT] %s %s, %s, [%s, #%lld]%s\n", load ? "LDP" : "STP",
              rn_z(rt, is64), rn_z(rt2, is64), rn_sp(rn, true), (long long)off,
              wback ? (post ? " (post)" : " (pre)") : "");

        base_load(rn);
        // The address [BIAS + rsi + disp] is encoded right into the access
        // operand — no intermediate lea, no inline bounds checks.
        int32_t base_disp = post ? 0 : (int32_t)off;
        for (int k = 0; k < 2; k++) {
            int r = (k == 0) ? rt : rt2;
            x86::Mem m = mem_at(x86::rsi, base_disp + k * scale, (uint32_t)scale);
            const x86::Gp& pin = pin_of(r);
            if (load) {
                if (pin.is_valid()) {
                    base_invalidate(r);             // pin overwritten — base cache dead
                    // Load straight into the pin, no rax detour.
                    if (sxtw)         a.movsxd(pin, m);
                    else if (is64)    a.mov(pin, m);
                    else              a.mov(pin.r32(), m);
                } else {
                    if (sxtw)      { a.movsxd(x86::rax, m); st_r(r, x86::rax, true); }
                    else if (is64) { a.mov(x86::rax, m);    st_r(r, x86::rax, true); }
                    else           { a.mov(x86::eax, m);    st_r(r, x86::rax, false); }
                }
            } else {
                if (pin.is_valid()) {
                    if (is64) a.mov(m, pin);
                    else      a.mov(m, pin.r32());
                } else {
                    ld_r(x86::rax, r, is64);
                    if (is64) a.mov(m, x86::rax); else a.mov(m, x86::eax);
                }
            }
        }
        if (wback) { base_clobber_rsi(); a.lea(x86::rsi, x86::ptr(x86::rsi, (int32_t)off)); st_rsp(rn, x86::rsi, true); }
        return Step::kNext;
    }

    // ---- LDR/STR: single registers ----
    if (bits(insn, 29, 27) == 0b111 && V == 0) {
        uint32_t size = bits(insn, 31, 30);
        uint32_t opc  = bits(insn, 23, 22);
        uint32_t nbytes = 1u << size;
        int rt = (int)bits(insn, 4, 0), rn = (int)bits(insn, 9, 5);

        bool wback = false, post = false;
        int64_t off = 0;
        bool reg_offset = false;

        if (bits(insn, 25, 24) == 0b01) {            // unsigned immediate offset
            off = (int64_t)(bits(insn, 21, 10) << size);
        } else if (bits(insn, 25, 24) == 0b00) {
            if (bit(insn, 21) == 0) {                 // unscaled / pre / post
                uint32_t mode = bits(insn, 11, 10);
                off = sext(bits(insn, 20, 12), 9);
                if (mode == 0b01)      { wback = true; post = true; }
                else if (mode == 0b11) { wback = true; post = false; }
                else if (mode != 0b00) return Step::kBad;   // LDTR/STTR — unsupported
            } else if (bits(insn, 11, 10) == 0b10) {  // register offset
                reg_offset = true;
            } else return Step::kBad;
        } else return Step::kBad;

        // PRFM and unallocated combinations
        if (size == 3 && opc == 2) return Step::kNext;
        if (opc == 3 && size >= 2) return Step::kBad;

        bool is_store = (opc == 0);
        static const char* ops[4][4] = {
            {"STRB", "LDRB", "LDRSB", "LDRSB"},
            {"STRH", "LDRH", "LDRSH", "LDRSH"},
            {"STR",  "LDR",  "LDRSW", "?"},
            {"STR",  "LDR",  "PRFM",  "?"},
        };
        e.log(1, "    [JIT] %s %s, [%s%s, #%lld]\n", ops[size][opc],
              rn_z(rt, size == 3 || opc == 2), rn_sp(rn, true),
              reg_offset ? ", Xm" : "", (long long)off);

        base_load(rn);
        x86::Mem m(0);
        if (reg_offset) {
            // x86 addressing cannot take two index registers — compute the
            // address in rdi.
            int rm = (int)bits(insn, 20, 16);
            uint32_t option = bits(insn, 15, 13), S = bit(insn, 12);
            ld_r(x86::rdx, rm, true);
            switch (option) {
                case 2: a.mov(x86::edx, x86::edx); break;          // UXTW
                case 3: break;                                      // LSL / UXTX
                case 6: a.movsxd(x86::rdx, x86::edx); break;        // SXTW
                case 7: break;                                      // SXTX
                default: return Step::kBad;
            }
            a.lea(x86::rdi, x86::ptr(x86::rsi, x86::rdx, (S && size) ? size : 0));
            m = mem_ref(x86::rdi, nbytes, pc);
        } else {
            m = mem_at(x86::rsi, post ? 0 : (int32_t)off, nbytes);
        }
        const x86::Gp& pin = pin_of(rt);
        if (is_store) {
            if (pin.is_valid()) {
                switch (size) {
                    case 0: a.mov(m, pin.r8());  break;
                    case 1: a.mov(m, pin.r16()); break;
                    case 2: a.mov(m, pin.r32()); break;
                    default: a.mov(m, pin);      break;
                }
            } else {
                ld_r(x86::rax, rt, size == 3);
                switch (size) {
                    case 0: a.mov(m, x86::al);  break;
                    case 1: a.mov(m, x86::ax);  break;
                    case 2: a.mov(m, x86::eax); break;
                    default: a.mov(m, x86::rax); break;
                }
            }
        } else if (opc == 1) {                          // ordinary load
            if (pin.is_valid()) {
                base_invalidate(rt);                    // pin overwritten — base cache dead
                switch (size) {
                    case 0: a.movzx(pin.r32(), m); break;
                    case 1: a.movzx(pin.r32(), m); break;
                    case 2: a.mov(pin.r32(), m);   break;
                    default: a.mov(pin, m);        break;
                }
            } else {
                switch (size) {
                    case 0: a.movzx(x86::eax, m); st_r(rt, x86::rax, false); break;
                    case 1: a.movzx(x86::eax, m); st_r(rt, x86::rax, false); break;
                    case 2: a.mov(x86::eax, m);   st_r(rt, x86::rax, false); break;
                    default: a.mov(x86::rax, m);  st_r(rt, x86::rax, true);  break;
                }
            }
        } else if (opc == 2) {                          // sign-extended load into X
            if (pin.is_valid()) {
                base_invalidate(rt);                    // pin overwritten — base cache dead
                switch (size) {
                    case 0: a.movsx(pin, m);  break;
                    case 1: a.movsx(pin, m);  break;
                    default: a.movsxd(pin, m); break;
                }
            } else {
                switch (size) {
                    case 0: a.movsx(x86::rax, m);  break;
                    case 1: a.movsx(x86::rax, m);  break;
                    default: a.movsxd(x86::rax, m); break;
                }
                st_r(rt, x86::rax, true);
            }
        } else {                                        // sign-extended load into W
            if (pin.is_valid()) {
                base_invalidate(rt);                    // pin overwritten — base cache dead
                switch (size) {
                    case 0: a.movsx(pin.r32(), m); break;
                    case 1: a.movsx(pin.r32(), m); break;
                    default: return Step::kBad;
                }
            } else {
                switch (size) {
                    case 0: a.movsx(x86::eax, m); break;
                    case 1: a.movsx(x86::eax, m); break;
                    default: return Step::kBad;
                }
                st_r(rt, x86::rax, false);
            }
        }

        if (wback) { a.lea(x86::rsi, x86::ptr(x86::rsi, (int32_t)off)); st_rsp(rn, x86::rsi, true); }
        return Step::kNext;
    }

    return Step::kBad;
}

// ---------------------------------------------------------------------------
//  Data Processing -- Register
// ---------------------------------------------------------------------------
Step BlockBuilder::tr_dp_reg(uint32_t insn, uint64_t pc) {
    bool sf = bit(insn, 31);
    int rd = (int)bits(insn, 4, 0), rn = (int)bits(insn, 9, 5), rm = (int)bits(insn, 20, 16);
    auto A = [&](const x86::Gp& g) { return sf ? g.r64() : g.r32(); };

    // Applies an ARM shift to a register.
    auto do_shift = [&](const x86::Gp& r, uint32_t type, uint32_t amount) {
        if (!amount) return;
        dirty_flags();
        switch (type) {
            case 0: a.shl(A(r), Imm(amount)); break;   // LSL
            case 1: a.shr(A(r), Imm(amount)); break;   // LSR
            case 2: a.sar(A(r), Imm(amount)); break;   // ASR
            default: a.ror(A(r), Imm(amount)); break;  // ROR
        }
    };

    // ---- Logical (shifted register) ----
    if ((insn & 0x1F000000u) == 0x0A000000u) {
        uint32_t opc = bits(insn, 30, 29), shift = bits(insn, 23, 22), N = bit(insn, 21);
        uint32_t imm6 = bits(insn, 15, 10);
        if (!sf && imm6 >= 32) return Step::kBad;
        static const char* nm[4][2] = {{"AND","BIC"},{"ORR","ORN"},{"EOR","EON"},{"ANDS","BICS"}};
        e.log(1, "    [JIT] %s %s%d, %s%d, %s%d\n", nm[opc][N], sf ? "X" : "W", rd,
              sf ? "X" : "W", rn, sf ? "X" : "W", rm);

        ld_r(x86::rcx, rm, sf);
        do_shift(x86::rcx, shift, imm6);
        if (N) a.not_(A(x86::rcx));                     // not preserves RFLAGS
        ld_r(x86::rax, rn, sf);
        if (opc == 3) begin_flag_def(); else dirty_flags();
        switch (opc) {
            case 0: case 3: a.and_(A(x86::rax), A(x86::rcx)); break;
            case 1: a.or_(A(x86::rax), A(x86::rcx)); break;
            default: a.xor_(A(x86::rax), A(x86::rcx)); break;
        }
        if (opc == 3) mark_flags(FlagSrc::kFromLogic);
        st_r(rd, x86::rax, sf);
        return Step::kNext;
    }

    // ---- Add/subtract (shifted register) ----
    if ((insn & 0x1F200000u) == 0x0B000000u) {
        bool is_sub = bit(insn, 30), setf = bit(insn, 29);
        uint32_t shift = bits(insn, 23, 22), imm6 = bits(insn, 15, 10);
        if (shift == 3) return Step::kBad;
        if (!sf && imm6 >= 32) return Step::kBad;
        e.log(1, "    [JIT] %s%s %s%d, %s%d, %s%d\n", is_sub ? "SUB" : "ADD",
              setf ? "S" : "", sf ? "X" : "W", rd, sf ? "X" : "W", rn, sf ? "X" : "W", rm);

        if (setf) begin_flag_def();          // the S form overwrites guest flags
        // SUBS XZR, Xn, Xm == CMP: compare against the pin or memory.
        if (sf && setf && rd == 31) {
            ld_r(x86::rcx, rm, sf);
            do_shift(x86::rcx, shift, imm6);
            const x86::Gp& pin = (rn == 31) ? PIN_SP : pin_of(rn);
            if (pin.is_valid()) a.cmp(pin, A(x86::rcx));
            else               a.cmp(reg_mem(rn), A(x86::rcx));
            mark_flags(FlagSrc::kFromSub);
            return Step::kNext;
        }
        // 64-bit Rd == Rn: the result goes straight into the pin, or into
        // memory via add/sub qword[ctx], reg.
        if (sf && rd == rn) {
            ld_r(x86::rcx, rm, sf);
            do_shift(x86::rcx, shift, imm6);
            const x86::Gp& pin = (rn == 31) ? PIN_SP : pin_of(rn);
            if (!setf) dirty_flags();
            base_invalidate(rn);                      // rn overwritten directly
            if (pin.is_valid()) {
                if (is_sub) a.sub(pin, x86::rcx);
                else        a.add(pin, x86::rcx);
            } else {
                x86::Mem dstm = reg_mem(rn);
                if (is_sub) a.sub(dstm, x86::rcx);
                else        a.add(dstm, x86::rcx);
            }
            if (setf) mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd);
            return Step::kNext;
        }
        // 64-bit Rd == Rm and the operation is commutative (ADD): add Rn to [Rd].
        if (sf && rd == rm && !is_sub) {
            ld_r(x86::rcx, rn, sf);
            if (!setf) dirty_flags();
            const x86::Gp& pin = (rd == 31) ? PIN_SP : pin_of(rd);
            base_invalidate(rd);                      // rd overwritten directly
            if (pin.is_valid()) a.add(pin, x86::rcx);
            else               a.add(reg_mem(rd), x86::rcx);
            if (setf) mark_flags(FlagSrc::kFromAdd);
            return Step::kNext;
        }

        ld_r(x86::rcx, rm, sf);
        do_shift(x86::rcx, shift, imm6);
        ld_r(x86::rax, rn, sf);
        if (is_sub) a.sub(A(x86::rax), A(x86::rcx)); else a.add(A(x86::rax), A(x86::rcx));
        if (setf) { mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd); }
        st_r(rd, x86::rax, sf);
        return Step::kNext;
    }

    // ---- Add/subtract (extended register) ----
    if ((insn & 0x1F200000u) == 0x0B200000u) {
        if (bits(insn, 23, 22) != 0) return Step::kBad;
        bool is_sub = bit(insn, 30), setf = bit(insn, 29);
        uint32_t option = bits(insn, 15, 13), imm3 = bits(insn, 12, 10);
        if (imm3 > 4) return Step::kBad;
        e.log(1, "    [JIT] %s%s (ext) %s%d, %s%d, %s%d\n", is_sub ? "SUB" : "ADD",
              setf ? "S" : "", sf ? "X" : "W", rd, rn == 31 ? "SP" : "X", rn, "X", rm);

        if (setf) begin_flag_def(); else dirty_flags();
        ld_r(x86::rcx, rm, true);
        switch (option) {
            case 0: a.movzx(x86::ecx, x86::cl); break;              // UXTB
            case 1: a.movzx(x86::ecx, x86::cx); break;              // UXTH
            case 2: a.mov(x86::ecx, x86::ecx); break;               // UXTW
            case 3: break;                                          // UXTX / LSL
            case 4: a.movsx(x86::rcx, x86::cl); break;              // SXTB
            case 5: a.movsx(x86::rcx, x86::cx); break;              // SXTH
            case 6: a.movsxd(x86::rcx, x86::ecx); break;            // SXTW
            default: break;                                         // SXTX
        }
        if (imm3) a.shl(x86::rcx, Imm(imm3));
        ld_rsp(x86::rax, rn, sf);
        if (is_sub) a.sub(A(x86::rax), A(x86::rcx)); else a.add(A(x86::rax), A(x86::rcx));
        if (setf) { mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd);
                    st_r(rd, x86::rax, sf); }
        else      { st_rsp(rd, x86::rax, sf); }
        return Step::kNext;
    }

    // ---- Conditional compare (CCMP / CCMN) ----
    if ((insn & 0x1FE00000u) == 0x1A400000u && bit(insn, 10) == 0 && bit(insn, 4) == 0) {
        bool is_ccmp = bit(insn, 30);
        uint32_t cond = bits(insn, 15, 12), nzcv = bits(insn, 3, 0);
        bool imm_form = bit(insn, 11);
        e.log(1, "    [JIT] %s %s%d, cond=%u\n", is_ccmp ? "CCMP" : "CCMN",
              sf ? "X" : "W", rn, cond);

        // Sample the condition before anything else; past the branch both
        // paths materialize the flags themselves, so kill the pending state
        // here.
        x86::CondCode cc;
        if (take_fused_cond(cond, cc)) a.set(cc, x86::r8b);
        else { dirty_flags(); emit_cond(cond); }
        a.test(x86::r8b, x86::r8b);
        Label lelse = a.new_label(), ldone = a.new_label();
        a.jz(lelse);
        ld_r(x86::rax, rn, sf);
        if (imm_form) {
            uint64_t imm5 = bits(insn, 20, 16);
            if (is_ccmp) { alu_ri(Alu::kCmp, x86::rax, imm5, sf); flags_sub(); }
            else         { alu_ri(Alu::kAdd, x86::rax, imm5, sf); flags_add(); }
        } else {
            ld_r(x86::rcx, rm, sf);
            if (is_ccmp) { a.cmp(A(x86::rax), A(x86::rcx)); flags_sub(); }
            else         { a.add(A(x86::rax), A(x86::rcx)); flags_add(); }
        }
        a.jmp(ldone);
        a.bind(lelse);
        a.mov(fN(), Imm((nzcv >> 3) & 1)); a.mov(fZ(), Imm((nzcv >> 2) & 1));
        a.mov(fC(), Imm((nzcv >> 1) & 1)); a.mov(fV(), Imm(nzcv & 1));
        a.bind(ldone);
        flags_pending = false;                     // both paths are already in memory
        return Step::kNext;
    }

    // ---- Conditional select (CSEL / CSINC / CSINV / CSNEG) ----
    if ((insn & 0x1FE00000u) == 0x1A800000u) {
        uint32_t cond = bits(insn, 15, 12), op2 = bits(insn, 11, 10);
        bool op = bit(insn, 30);
        if (op2 > 1) return Step::kBad;
        static const char* nm[2][2] = {{"CSEL","CSINC"},{"CSINV","CSNEG"}};
        e.log(1, "    [JIT] %s %s%d, %s%d, %s%d, cond=%u\n", nm[op][op2],
              sf ? "X" : "W", rd, sf ? "X" : "W", rn, sf ? "X" : "W", rm, cond);

        // Sample the condition BEFORE computing the values: while RFLAGS are
        // alive, one setcc beats the multi-step emit_cond.
        x86::CondCode cc;
        bool fused = take_fused_cond(cond, cc);
        if (fused) a.set(cc, x86::r8b);
        else dirty_flags();                        // materialize before add/neg below
        ld_r(x86::rax, rm, sf);                       // "condition false" path
        if (!op && op2 == 1)      a.add(A(x86::rax), Imm(1));     // CSINC
        else if (op && op2 == 0)  a.not_(A(x86::rax));            // CSINV
        else if (op && op2 == 1)  a.neg(A(x86::rax));             // CSNEG
        ld_r(x86::rcx, rn, sf);                       // "condition true" path
        if (!fused) emit_cond(cond);
        a.test(x86::r8b, x86::r8b);
        a.cmovne(A(x86::rax), A(x86::rcx));
        st_r(rd, x86::rax, sf);
        return Step::kNext;
    }

    // ---- Add/subtract with carry (ADC / SBC) ----
    if ((insn & 0x1FE0FC00u) == 0x1A000000u) {
        bool is_sub = bit(insn, 30), setf = bit(insn, 29);
        e.log(1, "    [JIT] %s%s %s%d, %s%d, %s%d\n", is_sub ? "SBC" : "ADC",
              setf ? "S" : "", sf ? "X" : "W", rd, sf ? "X" : "W", rn, sf ? "X" : "W", rm);
        dirty_flags();                             // fC is read from memory — it must be up to date
        ld_r(x86::rcx, rm, sf);
        ld_r(x86::rax, rn, sf);
        a.mov(x86::dl, fC());
        if (is_sub) a.xor_(x86::dl, Imm(1));          // ARM: SBC = Rn + ~Rm + C
        a.shr(x86::dl, Imm(1));                       // bit 0 -> CF
        if (is_sub) a.sbb(A(x86::rax), A(x86::rcx)); else a.adc(A(x86::rax), A(x86::rcx));
        if (setf) mark_flags(is_sub ? FlagSrc::kFromSub : FlagSrc::kFromAdd);
        st_r(rd, x86::rax, sf);
        return Step::kNext;
    }

    // ---- Data-processing (2 source): UDIV/SDIV/LSLV/LSRV/ASRV/RORV ----
    // The mask includes bit 30 (opc=00) so that the 1-source group
    // (CLZ/RBIT: opc=10) is not caught — the two sets overlap otherwise.
    if ((insn & 0x5FE00000u) == 0x1AC00000u) {
        uint32_t opcode = bits(insn, 15, 10);
        if (opcode == 0x02 || opcode == 0x03) {       // UDIV / SDIV
            bool is_signed = (opcode == 0x03);
            e.log(1, "    [JIT] %s %s%d, %s%d, %s%d\n", is_signed ? "SDIV" : "UDIV",
                  sf ? "X" : "W", rd, sf ? "X" : "W", rn, sf ? "X" : "W", rm);
            Label zero = a.new_label(), done = a.new_label(), normal = a.new_label();
            ld_r(x86::rcx, rm, sf);
            ld_r(x86::rax, rn, sf);
            dirty_flags();
            a.test(A(x86::rcx), A(x86::rcx));
            a.jz(zero);                                // ARM: division by zero yields 0
            if (is_signed) {
                // x86 raises #DE on INT_MIN / -1, ARM returns INT_MIN.
                a.cmp(A(x86::rcx), Imm(-1));
                a.jne(normal);
                a.neg(A(x86::rax));
                a.jmp(done);
                a.bind(normal);
                if (sf) a.cqo(); else a.cdq();
                a.idiv(A(x86::rcx));
            } else {
                a.xor_(x86::edx, x86::edx);
                a.div(A(x86::rcx));
            }
            a.jmp(done);
            a.bind(zero);
            a.xor_(x86::eax, x86::eax);
            a.bind(done);
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        if (opcode >= 0x08 && opcode <= 0x0B) {        // LSLV / LSRV / ASRV / RORV
            static const char* nm[] = {"LSLV", "LSRV", "ASRV", "RORV"};
            e.log(1, "    [JIT] %s %s%d, %s%d, %s%d\n", nm[opcode - 8], sf ? "X" : "W",
                  rd, sf ? "X" : "W", rn, sf ? "X" : "W", rm);
            ld_r(x86::rcx, rm, sf);                   // x86 masks the count to 5/6 bits itself
            ld_r(x86::rax, rn, sf);
            dirty_flags();
            switch (opcode) {
                case 0x08: a.shl(A(x86::rax), x86::cl); break;
                case 0x09: a.shr(A(x86::rax), x86::cl); break;
                case 0x0A: a.sar(A(x86::rax), x86::cl); break;
                default:   a.ror(A(x86::rax), x86::cl); break;
            }
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        return Step::kBad;
    }

    // ---- Data-processing (1 source): RBIT/REV16/REV32/REV/CLZ/CLS ----
    if ((insn & 0x5FE00000u) == 0x5AC00000u) {
        uint32_t opcode = bits(insn, 15, 10);
        if (bits(insn, 20, 16) != 0) return Step::kBad;

        // One swap-network step: exchange neighboring s-bit groups
        // (Hacker's Delight).
        auto swap_step = [&](unsigned s, uint64_t m) {
            x86::Gp v = sf ? x86::rax : x86::eax;
            x86::Gp t = sf ? x86::rcx : x86::ecx;
            x86::Gp k = sf ? x86::rdx : x86::edx;
            if (sf) a.mov(k, Imm(m)); else a.mov(k, Imm((uint32_t)m));
            a.mov(t, v);
            a.and_(t, k);
            a.shr(t, Imm(s));
            a.and_(v, k);
            a.shl(v, Imm(s));
            a.or_(v, t);
        };

        if (opcode == 0x00) {                          // RBIT
            e.log(1, "    [JIT] RBIT %s%d, %s%d\n", sf ? "X" : "W", rd, sf ? "X" : "W", rn);
            ld_r(x86::rax, rn, sf);
            dirty_flags();
            swap_step(1, 0x5555555555555555ull);
            swap_step(2, 0x3333333333333333ull);
            swap_step(4, 0x0F0F0F0F0F0F0F0Full);
            swap_step(8, 0x00FF00FF00FF00FFull);
            swap_step(16, 0x0000FFFF0000FFFFull);
            if (sf) {                                  // swap the 32-bit halves
                a.mov(x86::rcx, x86::rax);
                a.shr(x86::rax, Imm(32));
                a.shl(x86::rcx, Imm(32));
                a.or_(x86::rax, x86::rcx);
            }
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        if (opcode == 0x01) {                          // REV16
            e.log(1, "    [JIT] REV16 %s%d, %s%d\n", sf ? "X" : "W", rd, sf ? "X" : "W", rn);
            ld_r(x86::rax, rn, sf);
            dirty_flags();
            swap_step(8, 0x00FF00FF00FF00FFull);
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        if (sf && opcode == 0x02) {                    // REV32 (64)
            e.log(1, "    [JIT] REV32 X%d, X%d\n", rd, rn);
            ld_r(x86::rax, rn, sf);
            dirty_flags();
            swap_step(8, 0x00FF00FF00FF00FFull);
            swap_step(16, 0x0000FFFF0000FFFFull);
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        if (opcode == 0x04 || opcode == 0x05) {        // CLZ / CLS
            bool is_cls = (opcode == 0x05);
            unsigned size = sf ? 64 : 32;
            e.log(1, "    [JIT] %s %s%d, %s%d\n", is_cls ? "CLS" : "CLZ",
                  sf ? "X" : "W", rd, sf ? "X" : "W", rn);
            Label done = a.new_label();
            ld_r(x86::rax, rn, sf);
            dirty_flags();
            if (is_cls) {                              // clz(x ^ (x >> (size-1)))
                a.mov(x86::rcx, x86::rax);
                if (sf) a.sar(x86::rcx, Imm(63)); else a.sar(x86::ecx, Imm(31));
                a.xor_(x86::rax, x86::rcx);
            }
            a.mov(x86::edx, Imm(size));
            a.test(sf ? x86::rax : x86::eax, sf ? x86::rax : x86::eax);
            a.jz(done);
            a.bsr(sf ? x86::rcx : x86::ecx, sf ? x86::rax : x86::eax);
            a.mov(x86::edx, Imm(size - 1));
            a.sub(x86::edx, x86::ecx);
            a.bind(done);
            st_r(rd, x86::rdx, sf);
            return Step::kNext;
        }
        if ((sf && opcode == 0x03) || (!sf && opcode == 0x02)) {   // REV
            e.log(1, "    [JIT] REV %s%d, %s%d\n", sf ? "X" : "W", rd, sf ? "X" : "W", rn);
            ld_r(x86::rax, rn, sf);
            a.bswap(sf ? x86::rax : x86::eax);
            st_r(rd, x86::rax, sf);
            return Step::kNext;
        }
        return Step::kBad;
    }

    // ---- Data-processing (3 source): MADD/MSUB/MUL/UMULH/SMULH/[SU]MADDL ----
    if ((insn & 0x1F000000u) == 0x1B000000u) {
        if (bits(insn, 30, 29) != 0) return Step::kBad;
        uint32_t op31 = bits(insn, 23, 21), o0 = bit(insn, 15);
        int ra = (int)bits(insn, 14, 10);

        if (op31 == 0) {                               // MADD / MSUB (and MUL with Ra=XZR)
            e.log(1, "    [JIT] %s %s%d, %s%d, %s%d, %s%d\n", o0 ? "MSUB" : "MADD",
                  sf ? "X" : "W", rd, sf ? "X" : "W", rn, sf ? "X" : "W", rm, sf ? "X" : "W", ra);
            ld_r(x86::rax, rn, sf);
            ld_r(x86::rcx, rm, sf);
            dirty_flags();
            a.imul(A(x86::rax), A(x86::rcx));
            ld_r(x86::rdx, ra, sf);
            if (o0) { a.sub(A(x86::rdx), A(x86::rax)); st_r(rd, x86::rdx, sf); }
            else    { a.add(A(x86::rax), A(x86::rdx)); st_r(rd, x86::rax, sf); }
            return Step::kNext;
        }
        if (op31 == 0b001 || op31 == 0b101) {          // SMADDL/SMSUBL, UMADDL/UMSUBL
            bool is_signed = (op31 == 0b001);
            e.log(1, "    [JIT] %cM%sL X%d, W%d, W%d, X%d\n", is_signed ? 'S' : 'U',
                  o0 ? "SUB" : "ADD", rd, rn, rm, ra);
            ld_r(x86::rax, rn, false);
            ld_r(x86::rcx, rm, false);
            if (is_signed) { a.movsxd(x86::rax, x86::eax); a.movsxd(x86::rcx, x86::ecx); }
            dirty_flags();
            a.imul(x86::rax, x86::rcx);
            ld_r(x86::rdx, ra, true);
            if (o0) { a.sub(x86::rdx, x86::rax); st_r(rd, x86::rdx, true); }
            else    { a.add(x86::rax, x86::rdx); st_r(rd, x86::rax, true); }
            return Step::kNext;
        }
        if ((op31 == 0b010 || op31 == 0b110) && o0 == 0) {   // SMULH / UMULH
            bool is_signed = (op31 == 0b010);
            e.log(1, "    [JIT] %sMULH X%d, X%d, X%d\n", is_signed ? "S" : "U", rd, rn, rm);
            ld_r(x86::rax, rn, true);
            ld_r(x86::rcx, rm, true);
            dirty_flags();
            if (is_signed) a.imul(x86::rcx); else a.mul(x86::rcx);
            st_r(rd, x86::rdx, true);                  // the high 64 bits land in RDX
            return Step::kNext;
        }
        return Step::kBad;
    }

    return Step::kBad;
}
