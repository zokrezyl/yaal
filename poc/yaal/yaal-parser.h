/*
 * yaal — simplified-YAML SIMD parser POC
 *
 * Per-byte FSM (4 states, 3 input classes):
 *
 *           nl     sp     ns
 *   ST  ->  NL     ID     VL
 *   NL  ->  NL     ID     VL
 *   ID  ->  NL     ID     VL
 *   VL  ->  NL     VL     VL
 *
 * The parser walks input bytes through this FSM. Every actual state CHANGE
 * (next != state) invokes a user-supplied callback. Self-loops (NL+nl, ID+sp,
 * VL+sp, VL+ns) trigger no callback.
 *
 * The callback is the single extension point. To collect transitions to a
 * buffer, supply a callback that writes to a buffer. To benchmark raw FSM
 * throughput, pass NULL (no callback) and the parser only walks the FSM. To
 * verify against a known transition stream, supply a callback that compares.
 *
 * One implementation of the FSM step exists. Two byte-classification paths
 * exist: scalar and AVX2. Both feed into the same step.
 */

#ifndef YAAL_PARSER_H
#define YAAL_PARSER_H

#include <stddef.h>
#include <stdint.h>

enum yaal_state {
    YAAL_STATE_ST = 0, /* start */
    YAAL_STATE_NL = 1, /* newline (last consumed byte was '\n') */
    YAAL_STATE_ID = 2, /* indent run on current line */
    YAAL_STATE_VL = 3, /* value run on current line */
};

enum yaal_class {
    YAAL_CLASS_NL = 0, /* '\n' */
    YAAL_CLASS_SP = 1, /* ' ' */
    YAAL_CLASS_NS = 2, /* anything else */
};

enum yaal_impl {
    YAAL_IMPL_SCALAR = 0,
    YAAL_IMPL_AVX2 = 1,
};

struct yaal_transition {
    size_t offset;
    enum yaal_state from;
    enum yaal_state to;
};

/*
 * Called on every state CHANGE. NULL means silent (no callback invoked).
 * Offsets are absolute over the whole stream of bytes fed to the parser
 * (multiple feed calls accumulate).
 */
typedef void (*yaal_on_transition_fn)(void *ctx, size_t offset, enum yaal_state from,
                                      enum yaal_state to);

extern const uint8_t yaal_transition_table[4][3];

struct yaal_parser;

struct yaal_parser *yaal_parser_create(enum yaal_impl impl, yaal_on_transition_fn cb, void *ctx);
void yaal_parser_destroy(struct yaal_parser *p);
void yaal_parser_feed(struct yaal_parser *p, const uint8_t *buf, size_t len);
enum yaal_state yaal_parser_state(const struct yaal_parser *p);
size_t yaal_parser_offset(const struct yaal_parser *p);

#endif /* YAAL_PARSER_H */
