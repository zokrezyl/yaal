#
# yaal SIMD parser POC
#
# Build output goes to <repo-root>/build-linux/poc/yaal/.
#
# Common targets:
#   make            -> builds and runs the differential test
#   make build      -> builds all binaries
#   make run        -> runs the differential test
#   make roundtrip  -> A -> B -> verify (correctness via ground-truth)
#   make bench      -> A -> B -> bench (throughput)
#   make clean      -> removes build dir
#
# Roundtrip / bench knobs:
#   N         transitions to generate (default: 1000000 for roundtrip,
#             50000000 for bench — produces ~400 MiB of input on average)
#   SEED      PRNG seed (default: 1)
#   IMPL      scalar | avx2 (default: avx2)
#

REPO_ROOT := $(abspath ../..)
BUILD_DIR := $(REPO_ROOT)/build-linux/poc/yaal

CC      ?= cc
CFLAGS  ?= -O3 -g -Wall -Wextra -Wpedantic -std=c11
CFLAGS  += -mavx2 -mbmi -mbmi2 -mlzcnt -fno-plt

PARSER_OBJS := $(BUILD_DIR)/yaal-parser-scalar.o $(BUILD_DIR)/yaal-parser-avx2.o

BINS := \
	$(BUILD_DIR)/test-yaal-parser \
	$(BUILD_DIR)/gen-transitions \
	$(BUILD_DIR)/gen-input \
	$(BUILD_DIR)/verify \
	$(BUILD_DIR)/bench

DEPS := $(addsuffix .d, $(basename $(wildcard $(BUILD_DIR)/*.o)))

.PHONY: all build run roundtrip bench clean
all: run

build: $(BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test-yaal-parser: $(BUILD_DIR)/test-yaal-parser.o $(PARSER_OBJS)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/gen-transitions: $(BUILD_DIR)/gen-transitions.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/gen-input: $(BUILD_DIR)/gen-input.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/verify: $(BUILD_DIR)/verify.o $(PARSER_OBJS)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/bench: $(BUILD_DIR)/bench.o $(PARSER_OBJS)
	$(CC) $(CFLAGS) $^ -o $@

run: $(BUILD_DIR)/test-yaal-parser
	$<

# --- Roundtrip / bench harness ---------------------------------------------

ROUND_N    ?= 1000000
BENCH_N    ?= 50000000
SEED       ?= 1
IMPL       ?= avx2
TRANS_FILE := $(BUILD_DIR)/transitions.bin
INPUT_FILE := $(BUILD_DIR)/input.bin

roundtrip: $(BUILD_DIR)/gen-transitions $(BUILD_DIR)/gen-input $(BUILD_DIR)/verify
	$(BUILD_DIR)/gen-transitions -n $(ROUND_N) -s $(SEED) -o $(TRANS_FILE)
	$(BUILD_DIR)/gen-input -i $(TRANS_FILE) -s $(SEED) -o $(INPUT_FILE)
	$(BUILD_DIR)/verify -i $(INPUT_FILE) -t $(TRANS_FILE) -c $(IMPL)

bench: $(BUILD_DIR)/gen-transitions $(BUILD_DIR)/gen-input $(BUILD_DIR)/bench
	$(BUILD_DIR)/gen-transitions -n $(BENCH_N) -s $(SEED) -o $(TRANS_FILE)
	$(BUILD_DIR)/gen-input -i $(TRANS_FILE) -s $(SEED) -o $(INPUT_FILE)
	$(BUILD_DIR)/bench -i $(INPUT_FILE) -c $(IMPL)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
