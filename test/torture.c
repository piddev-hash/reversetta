#include <stdio.h>
#include <stdint.h>

// Torture test for the integer subset of ARM64.
// The golden reference is the x64 build of this same file: the diff of the
// two outputs must be empty.

static uint64_t t_add(uint64_t a, uint64_t b) { return a + b; }
static uint64_t t_sub(uint64_t a, uint64_t b) { return a - b; }
static uint64_t t_mul(uint64_t a, uint64_t b) { return a * b; }
static uint64_t t_and(uint64_t a, uint64_t b) { return a & b; }
static uint64_t t_or (uint64_t a, uint64_t b) { return a | b; }
static uint64_t t_xor(uint64_t a, uint64_t b) { return a ^ b; }
static uint64_t t_div(uint64_t a, uint64_t b) { return b ? a / b : 0; }
static int64_t  t_sdiv(int64_t a, int64_t b) { return b ? a / b : 0; }
static uint64_t t_umulh(uint64_t a, uint64_t b) { return (uint64_t)(((__uint128_t)a * b) >> 64); }
static int64_t  t_smulh(int64_t a, int64_t b) { return (int64_t)(((__int128_t)a * b) >> 64); }
static uint64_t t_madd(uint64_t a, uint64_t b, uint64_t c) { return a * b + c; }
static uint64_t t_msub(uint64_t a, uint64_t b, uint64_t c) { return c - a * b; }

static uint64_t t_shifts(uint64_t v, uint32_t s) {
    uint64_t r = 0;
    r += v << (s & 31);
    r += v >> (s & 31);
    r += (uint64_t)((int64_t)v >> (s & 31));
    r += (v << 32) >> (s & 63);
    r += v >> (s & 63);
    r += (uint64_t)((int64_t)v >> (s & 63));
    return r;
}
static uint64_t t_ror(uint64_t v, uint32_t s) { return (v >> (s & 63)) | (v << ((64 - s) & 63)); }

static uint32_t t_wops(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    r += a + b; r += a - b; r += a * b;
    r += a & b; r += a | b; r += a ^ b;
    r += a << (b & 31); r += a >> (b & 31);
    r += (uint32_t)((int32_t)a >> (b & 31));
    r += b ? a / b : 0;
    return r;
}

static uint64_t t_clzrev(uint64_t v) {
    return __builtin_clzll(v) * 1000000ull + __builtin_clz((uint32_t)v) * 1000ull
         + __builtin_ctzll(v) * 7ull + __builtin_bswap64(v) + __builtin_bswap32((uint32_t)v);
}

static uint64_t t_bitfields(uint64_t v) {
    uint64_t r = 0;
    r += (v >> 7) & 0x1FULL;
    r += (uint64_t)(((int64_t)v << 5) >> 9);
    r += (v & 0xFF00FF00FULL) | 0x0FULL;
    return r;
}

static uint64_t t_csel(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t r = a > b ? a : b;
    r += a <= b ? 1 : 2;
    r = (a == b) ? (r * 3) : (r + c);
    if (a < b) r ^= c;
    return r;
}

static uint64_t t_calls(uint64_t n) {
    if (n < 2) return 1;
    return t_calls(n - 1) + t_calls(n - 2);
}

typedef uint64_t (*fn_t)(uint64_t);
static uint64_t t_indirect(fn_t f, uint64_t v) { return f(v) + f(v + 1); }

static uint64_t t_mem(uint64_t n) {
    uint64_t buf[64];
    for (uint64_t i = 0; i < n; i++) buf[i] = i * i;
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += buf[i];
    return s;
}

static uint64_t t_mix(void) {
    volatile uint64_t q = 0x1122334455667788ull;
    volatile uint32_t w = 0xdeadbeefu;
    volatile uint16_t h = 0xcafeu;
    volatile uint8_t  b = 0x5a;
    return q + w + h + b;
}

static uint64_t t_loop_carry(void) {
    uint64_t hi = 0, lo = 0;
    for (int i = 0; i < 50; i++) {
        uint64_t t = lo + (uint64_t)i;
        if (t < lo) hi++;             // carry via comparison
        lo = t;
    }
    return hi * 1000 + (lo % 977);
}

int main(void) {
    printf("%llu\n", t_add(0x1111111111111111ull, 0x2222222222222222ull));
    printf("%llu\n", t_sub(0ull, 1ull));
    printf("%llu\n", t_mul(0x123456789ull, 0x9abcdefull));
    printf("%llu\n", t_and(0xF0F0F0F0F0F0F0F0ull, 0x0FF00FF00FF00FF0ull));
    printf("%llu\n", t_or (0xF0F0F0F0F0F0F0F0ull, 0x0F0F0F0F0F0F0F0Full));
    printf("%llu\n", t_xor(0xAAAAAAAAAAAAAAAAull, 0xFFFFFFFFFFFFFFFFull));
    printf("%llu\n", t_div(1000000007ull * 998244353ull, 998244353ull));
    printf("%lld\n", t_sdiv(-1000000007ll * 3, 3));
    printf("%llu\n", t_div(12345, 0));
    printf("%llu\n", t_umulh(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull));
    printf("%lld\n", t_smulh(INT64_MIN, INT64_MIN));
    printf("%llu\n", t_madd(0x9999999ull, 0x5555555ull, 0x12345ull));
    printf("%llu\n", t_msub(0x9999999ull, 0x5555555ull, 0x12345ull));
    printf("%llu\n", t_shifts(0x0123456789ABCDEFull, 13));
    printf("%llu\n", t_shifts(0x0123456789ABCDEFull, 1));
    printf("%llu\n", t_ror(0x0123456789ABCDEFull, 17));
    printf("%llu\n", t_ror(0x0123456789ABCDEFull, 1));
    printf("%llu\n", (uint64_t)t_wops(0xdeadbeefu, 0x13579BDFu));
    printf("%llu\n", (uint64_t)t_wops(0x13579BDFu, 0xdeadbeefu));
    printf("%llu\n", t_clzrev(1ull));
    printf("%llu\n", t_clzrev(0xF000000100000003ull));
    printf("%llu\n", t_bitfields(0x0123456789ABCDEFull));
    printf("%llu\n", t_csel(10, 20, 7));
    printf("%llu\n", t_csel(30, 20, 7));
    printf("%llu\n", t_csel(5, 5, 100));
    printf("%llu\n", t_calls(20));
    printf("%llu\n", t_indirect(t_calls, 10));
    printf("%llu\n", t_mem(64));
    printf("%llu\n", t_mix());
    printf("%llu\n", t_loop_carry());
    return 0;
}
