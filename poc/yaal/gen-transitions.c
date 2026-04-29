/*
 * Program A: generate a random sequence of valid yaal FSM transitions.
 *
 * Walks the state machine starting at ST and emits exactly N transitions
 * to a binary output file. Records are written as `struct yaal_transition`
 * (offset, from, to). Filler length between transitions is chosen at random
 * within each state's self-loop budget (ST has no self-loop, so the very
 * first transition is always at offset 0 — and any time we re-enter ST we
 * leave it on the next byte; in practice ST is only the initial state).
 *
 * Usage:
 *   gen-transitions -n <count> -o <file> [-s <seed>] [-f <max_filler>]
 *
 * Output format: raw binary, count * sizeof(struct yaal_transition).
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "yaal-parser.h"

/* splitmix64 — small, fast, deterministic PRNG. */
static inline uint64_t
sm64(uint64_t *s)
{
	uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

/*
 * Allowed (from, to) targets, indexed by current state. We list only real
 * transitions (from != to) — that's the parser's contract.
 */
static const enum yaal_state allowed_from_st[] = {
	YAAL_STATE_NL, YAAL_STATE_ID, YAAL_STATE_VL,
};
static const enum yaal_state allowed_from_nl[] = {
	YAAL_STATE_ID, YAAL_STATE_VL,
};
static const enum yaal_state allowed_from_id[] = {
	YAAL_STATE_NL, YAAL_STATE_VL,
};
static const enum yaal_state allowed_from_vl[] = {
	YAAL_STATE_NL,
};

static const struct {
	const enum yaal_state *targets;
	uint8_t count;
	uint8_t has_self_loop;	/* ST has no self-loop, all others do */
} state_info[4] = {
	[YAAL_STATE_ST] = { allowed_from_st, 3, 0 },
	[YAAL_STATE_NL] = { allowed_from_nl, 2, 1 },
	[YAAL_STATE_ID] = { allowed_from_id, 2, 1 },
	[YAAL_STATE_VL] = { allowed_from_vl, 1, 1 },
};

static void
usage(const char *p)
{
	fprintf(stderr,
		"usage: %s -n <count> -o <file> [-s <seed>] [-f <max_filler>]\n"
		"  -n  number of transitions to generate\n"
		"  -o  output file (binary)\n"
		"  -s  PRNG seed (default: time-based)\n"
		"  -f  max filler bytes between transitions (default: 16)\n",
		p);
}

#define OUT_BUF_RECS	(1u << 16)	/* 64K records ~= 1 MiB */

int
main(int argc, char **argv)
{
	uint64_t count = 0;
	uint64_t seed = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
	uint32_t max_filler = 16;
	const char *out_path = NULL;

	int opt;
	while ((opt = getopt(argc, argv, "n:o:s:f:h")) != -1) {
		switch (opt) {
		case 'n': count = strtoull(optarg, NULL, 10); break;
		case 'o': out_path = optarg; break;
		case 's': seed = strtoull(optarg, NULL, 10); break;
		case 'f': max_filler = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'h':
		default: usage(argv[0]); return opt == 'h' ? 0 : 2;
		}
	}

	if (count == 0 || !out_path) {
		usage(argv[0]);
		return 2;
	}

	int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		perror("open");
		return 1;
	}

	struct yaal_transition *buf =
		malloc(OUT_BUF_RECS * sizeof(*buf));
	if (!buf) {
		fprintf(stderr, "alloc failed\n");
		close(fd);
		return 1;
	}

	uint64_t prng = seed ? seed : 0x9E3779B97F4A7C15ull;
	enum yaal_state state = YAAL_STATE_ST;
	uint64_t offset = 0;
	uint64_t emitted = 0;
	size_t buf_idx = 0;

	while (emitted < count) {
		/*
		 * Filler: bytes that self-loop the current state. ST cannot
		 * self-loop, so filler is always 0 there.
		 */
		uint32_t filler = 0;
		if (state_info[state].has_self_loop && max_filler > 0)
			filler = (uint32_t)(sm64(&prng) % (max_filler + 1));

		/* Position of the byte that triggers this transition. */
		offset += filler;

		/* Pick a random allowed destination. */
		uint32_t pick = (uint32_t)(sm64(&prng) % state_info[state].count);
		enum yaal_state next = state_info[state].targets[pick];

		buf[buf_idx].offset = (size_t)offset;
		buf[buf_idx].from = state;
		buf[buf_idx].to = next;
		buf_idx++;

		if (buf_idx == OUT_BUF_RECS) {
			ssize_t w = write(fd, buf, buf_idx * sizeof(*buf));
			if (w < 0 || (size_t)w != buf_idx * sizeof(*buf)) {
				perror("write");
				free(buf);
				close(fd);
				return 1;
			}
			buf_idx = 0;
		}

		state = next;
		offset += 1;	/* the transition byte itself */
		emitted++;
	}

	if (buf_idx > 0) {
		ssize_t w = write(fd, buf, buf_idx * sizeof(*buf));
		if (w < 0 || (size_t)w != buf_idx * sizeof(*buf)) {
			perror("write");
			free(buf);
			close(fd);
			return 1;
		}
	}

	free(buf);
	close(fd);

	fprintf(stderr,
		"gen-transitions: wrote %" PRIu64 " records (%" PRIu64 " bytes)\n"
		"  seed=%" PRIu64 " max_filler=%" PRIu32 " final_state=%d\n"
		"  implied input length = %" PRIu64 " bytes\n",
		emitted, emitted * (uint64_t)sizeof(struct yaal_transition),
		seed, max_filler, (int)state, offset);
	return 0;
}
