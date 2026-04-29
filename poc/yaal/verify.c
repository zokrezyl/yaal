/*
 * Round-trip correctness check.
 *
 * Inputs:
 *   - input bytes file (gen-input output)
 *   - transitions file (gen-transitions output, the ground truth)
 *
 * Behaviour: parse the input file with a callback that streams the next
 * transition record from the ground-truth file and compares it with what
 * the parser just emitted. Any mismatch is fatal.
 *
 * The whole comparison is streaming on both sides — works for multi-GB
 * inputs without loading anything into memory.
 *
 * Usage:
 *   verify -i <input> -t <transitions> [-c (scalar|avx2)] [-b <chunk>]
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "yaal-parser.h"

#define TRANS_BUF_RECS (1u << 16)

struct verify_ctx {
    int trans_fd;
    struct yaal_transition *buf;
    size_t buf_count;
    size_t buf_pos;
    uint64_t total_compared;
    int eof;
    int mismatch;
};

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

static int refill(struct verify_ctx *v)
{
    if (v->eof)
        return 0;
    ssize_t r = read(v->trans_fd, v->buf, TRANS_BUF_RECS * sizeof(*v->buf));
    if (r < 0) {
        perror("read trans");
        v->mismatch = 1;
        return 0;
    }
    if (r == 0) {
        v->eof = 1;
        v->buf_count = 0;
        v->buf_pos = 0;
        return 0;
    }
    if ((size_t)r % sizeof(*v->buf) != 0) {
        fprintf(stderr, "trans file size not record-aligned\n");
        v->mismatch = 1;
        return 0;
    }
    v->buf_count = (size_t)r / sizeof(*v->buf);
    v->buf_pos = 0;
    return 1;
}

static void verify_cb(void *vctx, size_t offset, enum yaal_state from, enum yaal_state to)
{
    struct verify_ctx *v = vctx;
    if (v->mismatch)
        return;

    if (v->buf_pos == v->buf_count)
        (void)refill(v);

    if (v->buf_pos == v->buf_count) {
        fprintf(stderr,
                "verify: parser emitted extra event @off=%zu %s->%s "
                "(beyond ground truth count %" PRIu64 ")\n",
                offset, state_name(from), state_name(to), v->total_compared);
        v->mismatch = 1;
        return;
    }

    struct yaal_transition expected = v->buf[v->buf_pos++];
    if (expected.offset != offset || expected.from != from || expected.to != to) {
        fprintf(stderr,
                "verify: mismatch at event #%" PRIu64 ":\n"
                "  expected: off=%zu %s->%s\n"
                "  parser:   off=%zu %s->%s\n",
                v->total_compared, expected.offset, state_name(expected.from),
                state_name(expected.to), offset, state_name(from), state_name(to));
        v->mismatch = 1;
        return;
    }
    v->total_compared++;
}

static void usage(const char *p)
{
    fprintf(stderr,
            "usage: %s -i <input> -t <transitions> [-c scalar|avx2] [-b chunk_kib]\n"
            "  -i  input bytes file (from gen-input)\n"
            "  -t  transitions file (from gen-transitions)\n"
            "  -c  parser implementation (default: avx2)\n"
            "  -b  feed chunk size in KiB (default: 1024)\n",
            p);
}

int main(int argc, char **argv)
{
    const char *input_path = NULL;
    const char *trans_path = NULL;
    enum yaal_impl impl = YAAL_IMPL_AVX2;
    size_t chunk_kib = 1024;

    int opt;
    while ((opt = getopt(argc, argv, "i:t:c:b:h")) != -1) {
        switch (opt) {
        case 'i':
            input_path = optarg;
            break;
        case 't':
            trans_path = optarg;
            break;
        case 'c':
            if (strcmp(optarg, "scalar") == 0)
                impl = YAAL_IMPL_SCALAR;
            else if (strcmp(optarg, "avx2") == 0)
                impl = YAAL_IMPL_AVX2;
            else {
                usage(argv[0]);
                return 2;
            }
            break;
        case 'b':
            chunk_kib = strtoul(optarg, NULL, 10);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return opt == 'h' ? 0 : 2;
        }
    }
    if (!input_path || !trans_path) {
        usage(argv[0]);
        return 2;
    }

    int in_fd = open(input_path, O_RDONLY);
    if (in_fd < 0) {
        perror("open input");
        return 1;
    }
    struct stat st;
    if (fstat(in_fd, &st) < 0) {
        perror("fstat input");
        close(in_fd);
        return 1;
    }

    int trans_fd = open(trans_path, O_RDONLY);
    if (trans_fd < 0) {
        perror("open trans");
        close(in_fd);
        return 1;
    }

    struct verify_ctx vctx = {0};
    vctx.trans_fd = trans_fd;
    vctx.buf = malloc(TRANS_BUF_RECS * sizeof(*vctx.buf));
    if (!vctx.buf) {
        fprintf(stderr, "alloc failed\n");
        close(in_fd);
        close(trans_fd);
        return 1;
    }

    struct yaal_parser *p = yaal_parser_create(impl, verify_cb, &vctx);
    if (!p) {
        fprintf(stderr, "parser create failed\n");
        free(vctx.buf);
        close(in_fd);
        close(trans_fd);
        return 1;
    }

    size_t chunk_bytes = chunk_kib * 1024;
    uint8_t *chunk = aligned_alloc(64, chunk_bytes);
    if (!chunk) {
        fprintf(stderr, "alloc failed\n");
        yaal_parser_destroy(p);
        free(vctx.buf);
        close(in_fd);
        close(trans_fd);
        return 1;
    }

    uint64_t total = 0;
    int rc = 0;
    for (;;) {
        ssize_t r = read(in_fd, chunk, chunk_bytes);
        if (r < 0) {
            perror("read input");
            rc = 1;
            break;
        }
        if (r == 0)
            break;
        yaal_parser_feed(p, chunk, (size_t)r);
        total += (uint64_t)r;
        if (vctx.mismatch) {
            rc = 1;
            break;
        }
    }

    if (rc == 0) {
        /* Drain any remaining ground-truth records to confirm parity. */
        if (vctx.buf_pos < vctx.buf_count) {
            fprintf(stderr,
                    "verify: parser produced %" PRIu64 " events, but ground truth has more\n",
                    vctx.total_compared);
            rc = 1;
        } else {
            ssize_t r = read(trans_fd, vctx.buf, sizeof(*vctx.buf));
            if (r > 0) {
                fprintf(stderr, "verify: ground truth has more events than parser\n");
                rc = 1;
            } else if (r < 0) {
                perror("read trans (drain)");
                rc = 1;
            }
        }
    }

    if (rc == 0)
        printf("OK: %" PRIu64 " bytes parsed, %" PRIu64 " transitions matched (impl=%s)\n", total,
               vctx.total_compared, impl == YAAL_IMPL_AVX2 ? "avx2" : "scalar");
    else
        printf("FAIL\n");

    free(chunk);
    yaal_parser_destroy(p);
    free(vctx.buf);
    close(in_fd);
    close(trans_fd);
    return rc;
}
