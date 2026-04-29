/*
 * Internal definitions shared between the scalar and AVX2 implementation
 * units. Not part of the public API.
 */

#ifndef YAAL_PARSER_INTERNAL_H
#define YAAL_PARSER_INTERNAL_H

#include "yaal-parser.h"

struct yaal_parser {
    enum yaal_state state;
    size_t base_offset; /* total bytes fed so far */
    yaal_on_transition_fn cb;
    void *ctx;
    enum yaal_impl impl;
};

/*
 * The single FSM step. Given the current parser, the input class for one
 * byte, and that byte's absolute offset in the stream, advance the state
 * and (if the state changed) invoke the callback.
 *
 * Inlined into both the scalar and AVX2 feed loops so every byte goes
 * through exactly the same δ. There is no other place the FSM step is
 * implemented.
 */
static inline void yaal_parser_step(struct yaal_parser *p, enum yaal_class c, size_t offset)
{
    enum yaal_state next = (enum yaal_state)yaal_transition_table[p->state][c];

    if (next != p->state) {
        if (p->cb)
            p->cb(p->ctx, offset, p->state, next);
        p->state = next;
    }
}

static inline enum yaal_class yaal_classify_byte(uint8_t b)
{
    if (b == '\n')
        return YAAL_CLASS_NL;
    if (b == ' ')
        return YAAL_CLASS_SP;
    return YAAL_CLASS_NS;
}

/* Internal feed implementations — selected via parser->impl. */
void yaal_parser_feed_scalar(struct yaal_parser *p, const uint8_t *buf, size_t len);
void yaal_parser_feed_avx2(struct yaal_parser *p, const uint8_t *buf, size_t len);

#endif /* YAAL_PARSER_INTERNAL_H */
