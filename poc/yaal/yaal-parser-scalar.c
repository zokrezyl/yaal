/*
 * Scalar feed path: byte-by-byte. Hosts the public lifecycle API
 * (create/destroy/feed/state/offset) and the shared transition table.
 */

#include <stdlib.h>

#include "yaal-parser-internal.h"

const uint8_t yaal_transition_table[4][3] = {
	[YAAL_STATE_ST] = {
		[YAAL_CLASS_NL] = YAAL_STATE_NL,
		[YAAL_CLASS_SP] = YAAL_STATE_ID,
		[YAAL_CLASS_NS] = YAAL_STATE_VL,
	},
	[YAAL_STATE_NL] = {
		[YAAL_CLASS_NL] = YAAL_STATE_NL,
		[YAAL_CLASS_SP] = YAAL_STATE_ID,
		[YAAL_CLASS_NS] = YAAL_STATE_VL,
	},
	[YAAL_STATE_ID] = {
		[YAAL_CLASS_NL] = YAAL_STATE_NL,
		[YAAL_CLASS_SP] = YAAL_STATE_ID,
		[YAAL_CLASS_NS] = YAAL_STATE_VL,
	},
	[YAAL_STATE_VL] = {
		[YAAL_CLASS_NL] = YAAL_STATE_NL,
		[YAAL_CLASS_SP] = YAAL_STATE_VL,
		[YAAL_CLASS_NS] = YAAL_STATE_VL,
	},
};

struct yaal_parser *
yaal_parser_create(enum yaal_impl impl,
		   yaal_on_transition_fn cb, void *ctx)
{
	struct yaal_parser *p = calloc(1, sizeof(*p));
	if (!p)
		return NULL;
	p->state = YAAL_STATE_ST;
	p->base_offset = 0;
	p->cb = cb;
	p->ctx = ctx;
	p->impl = impl;
	return p;
}

void
yaal_parser_destroy(struct yaal_parser *p)
{
	free(p);
}

void
yaal_parser_feed(struct yaal_parser *p, const uint8_t *buf, size_t len)
{
	switch (p->impl) {
	case YAAL_IMPL_SCALAR:
		yaal_parser_feed_scalar(p, buf, len);
		break;
	case YAAL_IMPL_AVX2:
		yaal_parser_feed_avx2(p, buf, len);
		break;
	}
}

enum yaal_state
yaal_parser_state(const struct yaal_parser *p)
{
	return p->state;
}

size_t
yaal_parser_offset(const struct yaal_parser *p)
{
	return p->base_offset;
}

void
yaal_parser_feed_scalar(struct yaal_parser *p, const uint8_t *buf, size_t len)
{
	size_t base = p->base_offset;

	for (size_t i = 0; i < len; i++)
		yaal_parser_step(p, yaal_classify_byte(buf[i]), base + i);

	p->base_offset = base + len;
}
