/*
 * AVX2 feed paths — two routes dispatched by whether the parser has a
 * transition callback:
 *
 *   1. Hybrid (callback set): vectorize byte-classification only; the FSM
 *      step is byte-wise scalar so we deliver every transition with its
 *      absolute offset to the callback. Bottlenecked by the per-byte state
 *      dependency.
 *
 *   2. Silent (cb == NULL, used for throughput): per chunk we never look
 *      at intermediate states. We collapse the chunk to one of four
 *      cumulative functions purely from two bitmasks (nl positions, sp
 *      positions). Two SIMD compares + two movemasks per chunk. No scan,
 *      no closure, no dependency chain.
 *
 * Closure of {f_nl, f_sp, f_ns} under composition (ids 0..3 below):
 *   f_nl    = [NL,NL,NL,NL]   "→NL"            id 0
 *   f_sp    = [ID,ID,ID,VL]   "→ID unless VL"  id 1
 *   f_ns    = [VL,VL,VL,VL]   "→VL"            id 2
 *   f_to_id = [ID,ID,ID,ID]   "→ID"            id 3   (= f_sp ∘ f_nl)
 *
 * Why a chunk reduces to exactly one of these four:
 *   - no nl in chunk:
 *       all spaces      -> spaces only ever map ST/NL/ID->ID and VL->VL,
 *                          which is exactly f_sp.
 *       has any ns      -> the first ns from any state lands in VL; rest
 *                          stay in VL. So all states map to VL = f_ns.
 *   - has nl in chunk, with last nl at position L:
 *       L == 31         -> last byte is nl, leaves us in NL regardless of
 *                          start state = f_nl.
 *       trailing all sp -> after last nl we're in NL, then spaces drive
 *                          us to ID, where they stay. f_to_id.
 *       trailing has ns -> after last nl + first ns we're in VL, stay
 *                          there. f_ns.
 */

#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

#include "yaal-parser-internal.h"

#define YAAL_AVX2_CHUNK 32

/* ------------------------------------------------------------------------ */
/* Shared classification (used by the hybrid path).                         */
/* ------------------------------------------------------------------------ */

static inline __m256i classify_chunk_avx2(__m256i chunk)
{
    const __m256i nl = _mm256_set1_epi8('\n');
    const __m256i sp = _mm256_set1_epi8(' ');
    const __m256i v_ns = _mm256_set1_epi8(YAAL_CLASS_NS);
    const __m256i v_sp = _mm256_set1_epi8(YAAL_CLASS_SP);
    const __m256i v_nl = _mm256_set1_epi8(YAAL_CLASS_NL);

    __m256i is_sp = _mm256_cmpeq_epi8(chunk, sp);
    __m256i is_nl = _mm256_cmpeq_epi8(chunk, nl);

    __m256i out = _mm256_blendv_epi8(v_ns, v_sp, is_sp);
    out = _mm256_blendv_epi8(out, v_nl, is_nl);
    return out;
}

/* ------------------------------------------------------------------------ */
/* Hybrid feed (callback path).                                             */
/* ------------------------------------------------------------------------ */

static void feed_avx2_emit(struct yaal_parser *p, const uint8_t *buf, size_t len)
{
    size_t base = p->base_offset;
    size_t i = 0;

    uintptr_t addr = (uintptr_t)buf;
    size_t misalign = (size_t)(-addr) & (YAAL_AVX2_CHUNK - 1);
    size_t head_end = misalign < len ? misalign : len;

    for (; i < head_end; i++)
        yaal_parser_step(p, yaal_classify_byte(buf[i]), base + i);

    size_t body_end = i + ((len - i) & ~(size_t)(YAAL_AVX2_CHUNK - 1));

    alignas(32) uint8_t classes[YAAL_AVX2_CHUNK];

    for (; i < body_end; i += YAAL_AVX2_CHUNK) {
        __m256i chunk = _mm256_load_si256((const __m256i *)(buf + i));
        __m256i cls = classify_chunk_avx2(chunk);
        _mm256_store_si256((__m256i *)classes, cls);

        for (size_t k = 0; k < YAAL_AVX2_CHUNK; k++)
            yaal_parser_step(p, (enum yaal_class)classes[k], base + i + k);
    }

    for (; i < len; i++)
        yaal_parser_step(p, yaal_classify_byte(buf[i]), base + i);

    p->base_offset = base + len;
}

/* ------------------------------------------------------------------------ */
/* Silent feed (no callback) — bitmask-only.                                */
/* ------------------------------------------------------------------------ */

/*
 * apply_function_table[F*4 + state] -> output state, packed uint8_t so the
 * whole table fits in 16 bytes (one half of a cache line) and the load is
 * a single byte rather than a 4-byte enum.
 * Function ids: 0=f_nl, 1=f_sp, 2=f_ns, 3=f_to_id.
 */
static const uint8_t apply_function_table[16] = {
    YAAL_STATE_NL, YAAL_STATE_NL, YAAL_STATE_NL, YAAL_STATE_NL, /* f_nl */
    YAAL_STATE_ID, YAAL_STATE_ID, YAAL_STATE_ID, YAAL_STATE_VL, /* f_sp */
    YAAL_STATE_VL, YAAL_STATE_VL, YAAL_STATE_VL, YAAL_STATE_VL, /* f_ns */
    YAAL_STATE_ID, YAAL_STATE_ID, YAAL_STATE_ID, YAAL_STATE_ID, /* f_to_id */
};

#define APPLY_F(F, state) ((enum yaal_state)apply_function_table[((F) << 2) | (state)])

/*
 * compose_table[prev*4 + cur] = cur ∘ prev (apply prev first, then cur).
 * Used to fold per-chunk Fs into a single F before applying once to the
 * running state.
 */
static const uint8_t compose_table[16] = {
    /* prev=0 (f_nl):     cur=0..3 -> cur ∘ f_nl    */ 0,
    3,
    2,
    3,
    /* prev=1 (f_sp):     cur=0..3 -> cur ∘ f_sp    */ 0,
    1,
    2,
    3,
    /* prev=2 (f_ns):     cur=0..3 -> cur ∘ f_ns    */ 0,
    2,
    2,
    3,
    /* prev=3 (f_to_id):  cur=0..3 -> cur ∘ f_to_id */ 0,
    3,
    2,
    3,
};

static inline uint8_t fcompose(uint8_t a, uint8_t b) /* returns b ∘ a — a applied first */
{
    return compose_table[(a << 2) | b];
}

/*
 * Reduce a 32-byte chunk to its cumulative function id (0..3).
 *
 * Key observation: F is fully determined by the LAST non-space byte:
 *   - no non-space bytes (chunk is all spaces)              -> F = 1 (f_sp)
 *   - last non-space byte is ns                             -> F = 2 (f_ns)
 *   - last non-space byte is nl, at position 31 (last byte) -> F = 0 (f_nl)
 *   - last non-space byte is nl, at position < 31           -> F = 3 (f_to_id)
 *
 * That collapses everything to:
 *   - 2 vpcmpeqb + 2 vpmovmskb
 *   - lzcnt on (nl_mask | ns_mask | 1)
 *   - one bit-extract from nl_mask
 *   - branch-free mux
 */
__attribute__((always_inline)) static inline uint8_t chunk_function_id_bitmask(__m256i chunk)
{
    __m256i is_nl = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n'));
    __m256i is_sp = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(' '));

    uint32_t nl_mask = (uint32_t)_mm256_movemask_epi8(is_nl);
    uint32_t sp_mask = (uint32_t)_mm256_movemask_epi8(is_sp);

    /* "non-space bytes" = nl ∪ ns = ~sp directly. */
    uint32_t combined = ~sp_mask;

    /*
	 * Position of last non-space byte (0..31). lzcnt is well-defined for
	 * the | 1u sentinel; the resulting last_pos is only consumed via
	 * mask_combined below, which gates it to 0 when combined == 0.
	 */
    uint32_t last_pos = 31u - (uint32_t)_lzcnt_u32(combined | 1u);
    uint32_t is_nl_at_last = (nl_mask >> last_pos) & 1u; /* 0 or 1 */

    /* Booleans → 0/-1 selector masks. */
    uint32_t b_no_combined = (uint32_t)(combined == 0);
    uint32_t mask_no_comb = 0u - b_no_combined;
    uint32_t mask_comb = ~mask_no_comb;

    /* Combined != 0 arm: 2 by default; if last is nl, override to 0 or 3. */
    uint32_t b_at31 = combined >> 31;              /* 1 if last_pos == 31 */
    uint32_t F_nl_arm = 3u & (0u - (b_at31 ^ 1u)); /* 3 if !at31, else 0 */
    uint32_t mask_isnl = 0u - is_nl_at_last;       /* 0 or -1 */
    uint32_t F_combined = (F_nl_arm & mask_isnl) | (2u & ~mask_isnl);

    /* Final mux: 1 if all-space, otherwise F_combined. */
    uint32_t F = (1u & mask_no_comb) | (F_combined & mask_comb);
    return (uint8_t)F;
}

/*
 * 4-way unrolled silent feed.
 *
 * Per 128-byte iteration: 4 independent chunk-F computations (no inter-
 * dependency), tree-folded into a single F0123, then one state apply.
 * The state-carried dependency across iterations is therefore one apply
 * (≈4 cycles) per 128 bytes — well above memory bandwidth.
 */
static void feed_avx2_silent(struct yaal_parser *p, const uint8_t *buf, size_t len)
{
    enum yaal_state state = p->state;
    size_t i = 0;

    /* Head: scalar march until 32-byte aligned. */
    uintptr_t addr = (uintptr_t)buf;
    size_t misalign = (size_t)(-addr) & (YAAL_AVX2_CHUNK - 1);
    size_t head_end = misalign < len ? misalign : len;

    for (; i < head_end; i++)
        state = (enum yaal_state)yaal_transition_table[state][yaal_classify_byte(buf[i])];

    size_t body_end = i + ((len - i) & ~(size_t)(YAAL_AVX2_CHUNK - 1));

    /*
	 * 2-way unrolled body. One state apply per 64 bytes via tree-fold;
	 * compiler keeps everything in registers without spilling.
	 */
    size_t unroll2_end = i + ((body_end - i) & ~(size_t)(2 * YAAL_AVX2_CHUNK - 1));

    for (; i < unroll2_end; i += 2 * YAAL_AVX2_CHUNK) {
        if (i + 256 < unroll2_end)
            _mm_prefetch((const char *)(buf + i + 256), _MM_HINT_T0);
        __m256i c0 = _mm256_load_si256((const __m256i *)(buf + i));
        __m256i c1 = _mm256_load_si256((const __m256i *)(buf + i + 32));
        uint8_t F0 = chunk_function_id_bitmask(c0);
        uint8_t F1 = chunk_function_id_bitmask(c1);
        uint8_t F = fcompose(F0, F1);
        state = APPLY_F(F, state);
    }

    /* Per-chunk tail. */
    for (; i < body_end; i += YAAL_AVX2_CHUNK) {
        __m256i chunk = _mm256_load_si256((const __m256i *)(buf + i));
        uint8_t F = chunk_function_id_bitmask(chunk);
        state = APPLY_F(F, state);
    }

    /* Scalar tail. */
    for (; i < len; i++)
        state = (enum yaal_state)yaal_transition_table[state][yaal_classify_byte(buf[i])];

    p->state = state;
    p->base_offset += len;
}

/* ------------------------------------------------------------------------ */
/* Public dispatch.                                                         */
/* ------------------------------------------------------------------------ */

void yaal_parser_feed_avx2(struct yaal_parser *p, const uint8_t *buf, size_t len)
{
    if (p->cb)
        feed_avx2_emit(p, buf, len);
    else
        feed_avx2_silent(p, buf, len);
}
