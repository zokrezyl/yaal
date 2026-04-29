/*
 * Program B: turn a transitions file (from gen-transitions) into an input
 * byte file that, when parsed, will produce exactly those transitions.
 *
 * For each transition record (offset, from, to):
 *   - Fill positions [cursor, offset) with self-loop filler for `from`.
 *   - Place the trigger byte for the from->to change at `offset`.
 *   - cursor := offset + 1; state := to.
 *
 * Trigger bytes by destination state:
 *   ->NL : '\n'
 *   ->ID : ' '
 *   ->VL : any non-space, non-newline byte (we pick from 'a'..'z')
 *
 * Self-loop filler bytes by current state:
 *   NL : '\n' (only newline self-loops NL)
 *   ID : ' '  (only space  self-loops ID)
 *   VL : any non-newline    (pick a non-newline byte; mix 'a'..'z' and ' ')
 *   ST : impossible — every byte leaves ST. Records that imply ST filler
 *        are rejected as invalid input.
 *
 * Reads records in fixed-size chunks; writes output in fixed-size chunks.
 * Streaming, so it scales to multi-GB inputs.
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
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "yaal-parser.h"

static inline uint64_t sm64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

#define OUT_BUF_BYTES (1u << 20) /* 1 MiB */
#define IN_BUF_RECS (1u << 16)   /* 64K records */

static uint8_t trigger_byte(enum yaal_state to, uint64_t *prng)
{
    switch (to) {
    case YAAL_STATE_NL:
        return (uint8_t)'\n';
    case YAAL_STATE_ID:
        return (uint8_t)' ';
    case YAAL_STATE_VL: {
        /* Any byte that is neither '\n' nor ' '. */
        uint8_t b = (uint8_t)('a' + (sm64(prng) % 26));
        return b;
    }
    case YAAL_STATE_ST:
        break;
    }
    /* should not happen */
    return (uint8_t)'?';
}

/*
 * Write `n` self-loop filler bytes for the given state to the output buffer.
 * Returns 0 on success, -1 if the state has no self-loop (caller error).
 */
static int emit_filler(uint8_t *out, size_t n, enum yaal_state state, uint64_t *prng)
{
    if (n == 0)
        return 0;
    switch (state) {
    case YAAL_STATE_NL:
        memset(out, '\n', n);
        return 0;
    case YAAL_STATE_ID:
        memset(out, ' ', n);
        return 0;
    case YAAL_STATE_VL: {
        /*
		 * VL accepts any non-newline. Generate a chunk of randomish
		 * bytes by drawing 8 bytes from the PRNG at a time and
		 * masking — anything that comes out as '\n' is replaced.
		 */
        size_t i = 0;
        while (i < n) {
            uint64_t r = sm64(prng);
            for (int b = 0; b < 8 && i < n; b++, r >>= 8) {
                uint8_t v = (uint8_t)(r & 0xff);
                if (v == '\n')
                    v = ' '; /* still self-loops VL */
                out[i++] = v;
            }
        }
        return 0;
    }
    case YAAL_STATE_ST:
        return -1;
    }
    return -1;
}

static void usage(const char *p)
{
    fprintf(stderr,
            "usage: %s -i <transitions> -o <input> [-s <seed>] [-t <trail>]\n"
            "  -i  input transitions file (from gen-transitions)\n"
            "  -o  output input bytes file\n"
            "  -s  PRNG seed (default: time-based)\n"
            "  -t  trailing filler bytes after last transition (default: 0)\n",
            p);
}

int main(int argc, char **argv)
{
    const char *trans_path = NULL;
    const char *out_path = NULL;
    uint64_t seed = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
    uint64_t trail = 0;

    int opt;
    while ((opt = getopt(argc, argv, "i:o:s:t:h")) != -1) {
        switch (opt) {
        case 'i':
            trans_path = optarg;
            break;
        case 'o':
            out_path = optarg;
            break;
        case 's':
            seed = strtoull(optarg, NULL, 10);
            break;
        case 't':
            trail = strtoull(optarg, NULL, 10);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return opt == 'h' ? 0 : 2;
        }
    }
    if (!trans_path || !out_path) {
        usage(argv[0]);
        return 2;
    }

    int in_fd = open(trans_path, O_RDONLY);
    if (in_fd < 0) {
        perror("open trans");
        return 1;
    }
    int out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        perror("open input");
        close(in_fd);
        return 1;
    }

    struct yaal_transition *recs = malloc(IN_BUF_RECS * sizeof(*recs));
    uint8_t *out = malloc(OUT_BUF_BYTES);
    if (!recs || !out) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    uint64_t prng = seed ? seed : 0xA5A5A5A5A5A5A5A5ull;
    enum yaal_state state = YAAL_STATE_ST;
    uint64_t cursor = 0; /* next byte to emit */
    uint64_t total = 0;
    size_t out_off = 0;
    uint64_t total_records = 0;
    int rc = 0;

#define FLUSH_OUT()                                                                                \
    do {                                                                                           \
        if (out_off > 0) {                                                                         \
            ssize_t _w = write(out_fd, out, out_off);                                              \
            if (_w < 0 || (size_t)_w != out_off) {                                                 \
                perror("write input");                                                             \
                rc = 1;                                                                            \
                goto done;                                                                         \
            }                                                                                      \
            total += out_off;                                                                      \
            out_off = 0;                                                                           \
        }                                                                                          \
    } while (0)

#define EMIT_BYTES(src, len)                                                                       \
    do {                                                                                           \
        size_t _remaining = (len);                                                                 \
        const uint8_t *_p = (const uint8_t *)(src);                                                \
        while (_remaining > 0) {                                                                   \
            size_t _space = OUT_BUF_BYTES - out_off;                                               \
            size_t _take = _remaining < _space ? _remaining : _space;                              \
            memcpy(out + out_off, _p, _take);                                                      \
            out_off += _take;                                                                      \
            _p += _take;                                                                           \
            _remaining -= _take;                                                                   \
            if (out_off == OUT_BUF_BYTES)                                                          \
                FLUSH_OUT();                                                                       \
        }                                                                                          \
    } while (0)

    for (;;) {
        ssize_t r = read(in_fd, recs, IN_BUF_RECS * sizeof(*recs));
        if (r < 0) {
            perror("read trans");
            rc = 1;
            goto done;
        }
        if (r == 0)
            break;
        if ((size_t)r % sizeof(*recs) != 0) {
            fprintf(stderr, "trans file size not record-aligned\n");
            rc = 1;
            goto done;
        }
        size_t n = (size_t)r / sizeof(*recs);
        total_records += n;

        for (size_t i = 0; i < n; i++) {
            struct yaal_transition rec = recs[i];

            if ((size_t)rec.offset < cursor) {
                fprintf(stderr,
                        "trans not monotonic at rec %" PRIu64 ": offset %zu < cursor %" PRIu64 "\n",
                        total_records - n + i, rec.offset, cursor);
                rc = 1;
                goto done;
            }
            if (rec.from != state) {
                fprintf(stderr, "trans state desync at rec %" PRIu64 ": from=%d expected=%d\n",
                        total_records - n + i, (int)rec.from, (int)state);
                rc = 1;
                goto done;
            }

            size_t filler = rec.offset - cursor;
            if (filler > 0) {
                if (state == YAAL_STATE_ST) {
                    fprintf(stderr,
                            "trans implies ST filler "
                            "(rec %" PRIu64 "): impossible\n",
                            total_records - n + i);
                    rc = 1;
                    goto done;
                }
                /*
				 * Stream filler in chunks bounded by output
				 * buffer space — emit_filler fills exactly
				 * what we ask for.
				 */
                while (filler > 0) {
                    size_t space = OUT_BUF_BYTES - out_off;
                    size_t take = filler < space ? filler : space;
                    if (emit_filler(out + out_off, take, state, &prng) < 0) {
                        fprintf(stderr, "emit_filler failed\n");
                        rc = 1;
                        goto done;
                    }
                    out_off += take;
                    filler -= take;
                    if (out_off == OUT_BUF_BYTES)
                        FLUSH_OUT();
                }
            }

            /* Trigger byte at rec.offset. */
            if (out_off == OUT_BUF_BYTES)
                FLUSH_OUT();
            out[out_off++] = trigger_byte(rec.to, &prng);
            cursor = rec.offset + 1;
            state = rec.to;
        }
    }

    /* Optional trailing filler in the final state. */
    if (trail > 0) {
        if (state == YAAL_STATE_ST) {
            fprintf(stderr, "warning: trail>0 but final state is ST; skipping\n");
        } else {
            while (trail > 0) {
                size_t space = OUT_BUF_BYTES - out_off;
                size_t take = trail < space ? trail : space;
                if (emit_filler(out + out_off, take, state, &prng) < 0) {
                    fprintf(stderr, "emit_filler failed (trail)\n");
                    rc = 1;
                    goto done;
                }
                out_off += take;
                trail -= take;
                if (out_off == OUT_BUF_BYTES)
                    FLUSH_OUT();
            }
        }
    }

    FLUSH_OUT();

    fprintf(stderr,
            "gen-input: read %" PRIu64 " transitions, wrote %" PRIu64
            " input bytes\n  seed=%" PRIu64 " final_state=%d\n",
            total_records, total, seed, (int)state);

done:
    free(recs);
    free(out);
    close(in_fd);
    close(out_fd);
    return rc;
}
