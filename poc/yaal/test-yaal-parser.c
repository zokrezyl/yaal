/*
 * Differential test: scalar vs AVX2 feed paths must produce identical
 * transition streams for any input.
 *
 * Each parser is created with a "collector" callback that appends every
 * (offset, from, to) into a buffer. We then byte-compare the two buffers.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yaal-parser.h"

#define EVENTS_CAP (1u << 20)
#define BUF_PAD 64

struct collector {
    struct yaal_transition *evs;
    size_t count;
    size_t cap;
    int overflow;
};

static void collect_cb(void *vctx, size_t offset, enum yaal_state from, enum yaal_state to)
{
    struct collector *c = vctx;
    if (c->count >= c->cap) {
        c->overflow = 1;
        return;
    }
    c->evs[c->count].offset = offset;
    c->evs[c->count].from = from;
    c->evs[c->count].to = to;
    c->count++;
}

static const char *state_name(enum yaal_state s)
{
    switch (s) {
    case YAAL_STATE_ST:
        return "ST";
    case YAAL_STATE_NL:
        return "NL";
    case YAAL_STATE_ID:
        return "ID";
    case YAAL_STATE_VL:
        return "VL";
    }
    return "??";
}

static int run_one(const uint8_t *buf, size_t len, enum yaal_impl impl, struct collector *c)
{
    c->count = 0;
    c->overflow = 0;
    struct yaal_parser *p = yaal_parser_create(impl, collect_cb, c);
    if (!p)
        return -1;
    yaal_parser_feed(p, buf, len);
    yaal_parser_destroy(p);
    return c->overflow ? -1 : 0;
}

static int diff(const struct collector *a, const struct collector *b, size_t len, const char *label)
{
    if (a->overflow || b->overflow) {
        fprintf(stderr, "[%s] event buffer overflow\n", label);
        return 1;
    }
    if (a->count != b->count) {
        fprintf(stderr, "[%s] count differs: scalar=%zu avx2=%zu (len=%zu)\n", label, a->count,
                b->count, len);
        return 1;
    }
    for (size_t i = 0; i < a->count; i++) {
        const struct yaal_transition *x = &a->evs[i];
        const struct yaal_transition *y = &b->evs[i];
        if (x->offset != y->offset || x->from != y->from || x->to != y->to) {
            fprintf(stderr,
                    "[%s] event %zu differs:\n"
                    "  scalar: off=%zu %s->%s\n"
                    "  avx2:   off=%zu %s->%s\n",
                    label, i, x->offset, state_name(x->from), state_name(x->to), y->offset,
                    state_name(y->from), state_name(y->to));
            return 1;
        }
    }
    return 0;
}

struct buffers {
    uint8_t *raw;
    size_t raw_len;
    struct collector ca, cb;
};

static enum yaal_state final_state_silent(const uint8_t *buf, size_t len, enum yaal_impl impl)
{
    struct yaal_parser *p = yaal_parser_create(impl, NULL, NULL);
    yaal_parser_feed(p, buf, len);
    enum yaal_state s = yaal_parser_state(p);
    yaal_parser_destroy(p);
    return s;
}

static int diff_silent(const uint8_t *buf, size_t len, const char *label,
                       const struct collector *scalar_cb)
{
    enum yaal_state from_scalar_cb =
        scalar_cb->count == 0 ? YAAL_STATE_ST : scalar_cb->evs[scalar_cb->count - 1].to;
    enum yaal_state silent_avx2 = final_state_silent(buf, len, YAAL_IMPL_AVX2);
    if (from_scalar_cb != silent_avx2) {
        fprintf(stderr,
                "[%s silent] final state mismatch: "
                "scalar(cb)=%s avx2(silent)=%s (len=%zu)\n",
                label, state_name(from_scalar_cb), state_name(silent_avx2), len);
        return 1;
    }
    return 0;
}

static int run_diff(struct buffers *bx, const uint8_t *buf, size_t len, const char *label)
{
    if (run_one(buf, len, YAAL_IMPL_SCALAR, &bx->ca) ||
        run_one(buf, len, YAAL_IMPL_AVX2, &bx->cb)) {
        fprintf(stderr, "[%s] parse failed\n", label);
        return 1;
    }
    if (diff(&bx->ca, &bx->cb, len, label))
        return 1;
    /* Silent (NULL-cb) AVX2 path must agree on the final state. */
    return diff_silent(buf, len, label, &bx->ca);
}

static int run_at_alignment(struct buffers *bx, const uint8_t *payload, size_t plen,
                            size_t align_off, const char *label)
{
    uintptr_t base = (uintptr_t)bx->raw;
    uintptr_t aligned = (base + 31) & ~(uintptr_t)31;
    uint8_t *target = (uint8_t *)(aligned + align_off);

    if ((uintptr_t)(target + plen) > (uintptr_t)(bx->raw + bx->raw_len)) {
        fprintf(stderr, "[%s] payload would overflow test buffer\n", label);
        return 1;
    }
    memcpy(target, payload, plen);

    char sub[128];
    snprintf(sub, sizeof(sub), "%s align=%zu len=%zu", label, align_off, plen);
    return run_diff(bx, target, plen, sub);
}

static int test_crafted(struct buffers *bx)
{
    struct case_def {
        const char *label;
        const char *text;
    };
    static const struct case_def cases[] = {
        {"empty", ""},
        {"just-nl", "\n"},
        {"just-sp", "   "},
        {"just-ns", "abc"},
        {"indent-then-value", "    hello"},
        {"value-with-space", "hello world more"},
        {"two-lines", "a\nb\n"},
        {"indent-blank", "   \n"},
        {"long-mixed", "foo: bar\n"
                       "  baz: qux\n"
                       "    deeper: value with spaces and more\n"
                       "\n"
                       "another\n"
                       "     last\n"},
        {"spaces-in-value", "x   spaced   value\n  indent and value with    runs\n"},
    };

    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        size_t plen = strlen(cases[i].text);
        for (size_t off = 0; off < 32; off++) {
            if (run_at_alignment(bx, (const uint8_t *)cases[i].text, plen, off, cases[i].label))
                rc = 1;
        }
    }
    return rc;
}

static uint8_t random_byte(unsigned *seed)
{
    unsigned r = (unsigned)rand_r(seed) & 0xff;
    if (r < 16)
        return '\n';
    if (r < 96)
        return ' ';
    return (uint8_t)('a' + (r & 0x0f));
}

static int test_fuzz(struct buffers *bx, size_t iters)
{
    unsigned seed = 0xC0FFEEu;
    int rc = 0;

    uintptr_t base = (uintptr_t)bx->raw;
    uintptr_t aligned = (base + 31) & ~(uintptr_t)31;
    size_t headroom = bx->raw_len - (size_t)(aligned - base);

    static const size_t lens[] = {
        0,  1,  2,  7,  15, 16,  17,  31,  32,  33,  47,  48,  49,   63,
        64, 65, 95, 96, 97, 127, 128, 200, 257, 511, 512, 513, 1024, 4096,
    };

    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t plen = lens[li];
        if (plen + 32 > headroom)
            continue;
        for (size_t off = 0; off < 32; off++) {
            uint8_t *target = (uint8_t *)(aligned + off);
            for (size_t i = 0; i < plen; i++)
                target[i] = random_byte(&seed);
            char sub[64];
            snprintf(sub, sizeof(sub), "fuzz off=%zu len=%zu", off, plen);
            if (run_diff(bx, target, plen, sub))
                rc = 1;
        }
    }

    for (size_t it = 0; it < iters; it++) {
        size_t plen = 1 + ((unsigned)rand_r(&seed) % 8192);
        if (plen + 32 > headroom)
            continue;
        size_t off = (unsigned)rand_r(&seed) % 32;
        uint8_t *target = (uint8_t *)(aligned + off);
        for (size_t i = 0; i < plen; i++)
            target[i] = random_byte(&seed);
        char sub[64];
        snprintf(sub, sizeof(sub), "fuzz-rand it=%zu off=%zu len=%zu", it, off, plen);
        if (run_diff(bx, target, plen, sub))
            return 1;
    }
    return rc;
}

int main(void)
{
    struct buffers bx;
    bx.raw_len = 16 * 1024 + BUF_PAD;
    bx.raw = aligned_alloc(64, bx.raw_len);
    bx.ca.cap = EVENTS_CAP;
    bx.cb.cap = EVENTS_CAP;
    bx.ca.evs = malloc(EVENTS_CAP * sizeof(*bx.ca.evs));
    bx.cb.evs = malloc(EVENTS_CAP * sizeof(*bx.cb.evs));
    if (!bx.raw || !bx.ca.evs || !bx.cb.evs) {
        fprintf(stderr, "alloc failed\n");
        return 2;
    }

    int rc = 0;
    if (test_crafted(&bx))
        rc = 1;
    if (test_fuzz(&bx, 256))
        rc = 1;

    free(bx.raw);
    free(bx.ca.evs);
    free(bx.cb.evs);

    if (rc == 0)
        printf("OK: scalar and AVX2 produced identical event streams\n");
    else
        printf("FAIL\n");
    return rc;
}
