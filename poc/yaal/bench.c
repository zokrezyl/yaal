/*
 * Single-run, three-pass bench:
 *
 *   Pass 1: raw 32-byte AVX2 reads over the whole buffer — warmup.
 *           Result discarded but consumed (sink) so the compiler can't DCE
 *           the loop; this brings the buffer (or as much as fits) into the
 *           caches and faults in any pages mmap may have left cold.
 *
 *   Pass 2: same raw read again, this time timed. Establishes the raw
 *           memory-read throughput on this host for this buffer size.
 *
 *   Pass 3: yaal_parser_feed in silent mode (NULL callback) over the same
 *           buffer, timed. Compares directly with pass 2.
 *
 * Anti-DCE: raw reduction lands in a volatile global; parser final state
 * lands in a volatile global. main() prints both values at the end —
 * because they are observed at the program boundary, the compiler cannot
 * elide the work that produced them.
 *
 * Usage:
 *   bench -i <input> [-c scalar|avx2]
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <fcntl.h>
#include <getopt.h>
#include <immintrin.h>
#include <inttypes.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "yaal-parser.h"

/* Globals are volatile so writes are observable at program boundary. */
volatile uint64_t bench_sink_raw;
volatile uint8_t  bench_sink_parser;

static inline double
now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * Raw-read pass: XOR every aligned 32-byte chunk into an AVX2 accumulator,
 * reduce to a u64, write to volatile sink. Returns elapsed seconds.
 *
 * Buffer is required to be 32-byte aligned (mmap returns page-aligned, so
 * this is fine). Tail bytes are folded in scalar.
 */
static double
raw_read_pass(const uint8_t *buf, size_t len)
{
	__m256i acc = _mm256_setzero_si256();
	size_t i = 0;
	size_t end = len & ~(size_t)31;

	double t0 = now_sec();

	for (; i < end; i += 32) {
		__m256i x = _mm256_load_si256((const __m256i *)(buf + i));
		acc = _mm256_xor_si256(acc, x);
	}

	uint8_t tail = 0;
	for (; i < len; i++)
		tail ^= buf[i];

	double dt = now_sec() - t0;

	/* Reduce 32 bytes -> 1 u64 and stash in volatile global. */
	alignas(32) uint8_t out[32];
	_mm256_store_si256((__m256i *)out, acc);
	uint64_t s = (uint64_t)tail;
	for (int k = 0; k < 32; k++)
		s ^= ((uint64_t)out[k]) << ((k & 7) * 8);
	bench_sink_raw ^= s;
	return dt;
}

static double
parser_silent_pass(const uint8_t *buf, size_t len, enum yaal_impl impl,
		   enum yaal_state *out_state)
{
	struct yaal_parser *p = yaal_parser_create(impl, NULL, NULL);
	if (!p) {
		fprintf(stderr, "parser create failed\n");
		exit(1);
	}

	double t0 = now_sec();
	yaal_parser_feed(p, buf, len);
	double dt = now_sec() - t0;

	enum yaal_state s = yaal_parser_state(p);
	yaal_parser_destroy(p);

	bench_sink_parser ^= (uint8_t)s;
	if (out_state)
		*out_state = s;
	return dt;
}

static void
usage(const char *p)
{
	fprintf(stderr,
		"usage: %s -i <input> [-c scalar|avx2]\n"
		"  -i  input bytes file (from gen-input)\n"
		"  -c  parser implementation (default: avx2)\n",
		p);
}

int
main(int argc, char **argv)
{
	const char *input_path = NULL;
	enum yaal_impl impl = YAAL_IMPL_AVX2;

	int opt;
	while ((opt = getopt(argc, argv, "i:c:h")) != -1) {
		switch (opt) {
		case 'i': input_path = optarg; break;
		case 'c':
			if (strcmp(optarg, "scalar") == 0)
				impl = YAAL_IMPL_SCALAR;
			else if (strcmp(optarg, "avx2") == 0)
				impl = YAAL_IMPL_AVX2;
			else { usage(argv[0]); return 2; }
			break;
		case 'h':
		default: usage(argv[0]); return opt == 'h' ? 0 : 2;
		}
	}
	if (!input_path) {
		usage(argv[0]);
		return 2;
	}

	int fd = open(input_path, O_RDONLY);
	if (fd < 0) { perror("open"); return 1; }

	struct stat st;
	if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }

	size_t len = (size_t)st.st_size;
	if (len == 0) {
		fprintf(stderr, "empty input file\n");
		close(fd);
		return 1;
	}

	void *map = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
	madvise(map, len, MADV_SEQUENTIAL);

	const uint8_t *buf = map;

	/* --- Pass 1: raw read warmup ----------------------------------- */
	double t_raw_warmup = raw_read_pass(buf, len);

	/* --- Pass 2: raw read measure ---------------------------------- */
	double t_raw = raw_read_pass(buf, len);

	/* --- Pass 3: parser silent ------------------------------------- */
	enum yaal_state final_state = YAAL_STATE_ST;
	double t_parse = parser_silent_pass(buf, len, impl, &final_state);

	double GiB = 1.0 / (1024.0 * 1024.0 * 1024.0);
	double size_GiB = (double)len * GiB;

	double raw_bw = size_GiB / t_raw;
	double parse_bw = size_GiB / t_parse;
	double ratio = parse_bw / raw_bw * 100.0;

	printf("input:                 %.3f GiB (%zu bytes)\n", size_GiB, len);
	printf("impl:                  %s\n",
	       impl == YAAL_IMPL_AVX2 ? "avx2" : "scalar");
	printf("\n");
	printf("pass 1 raw   (warmup): %.3f ms\n", t_raw_warmup * 1000.0);
	printf("pass 2 raw   (measure):%.3f ms   %.3f GiB/s\n",
	       t_raw   * 1000.0, raw_bw);
	printf("pass 3 parse (silent): %.3f ms   %.3f GiB/s    (%.1f%% of raw)\n",
	       t_parse * 1000.0, parse_bw, ratio);
	printf("\n");
	printf("final_state:           %d\n", (int)final_state);
	printf("sink_raw:              0x%016" PRIx64 "\n", bench_sink_raw);
	printf("sink_parser:           0x%02x\n", bench_sink_parser);

	munmap(map, len);
	close(fd);
	return 0;
}
