# sml-autodiff build
#
#   make            build the test binary with MLton (default)
#   make test       build + run tests under MLton
#   make test-poly  run tests under Poly/ML (use-and-run; no link step)
#   make all-tests  run the suite under both compilers
#   make example    run the gradient-descent / Newton demo, write assets/demo.txt
#   make clean      remove build artifacts
#
# Layout A, dependency-free: the library sources live directly under
# lib/github.com/sjqtentacles/sml-autodiff/ and rely on nothing beyond the
# Standard ML Basis Library.  The test suite and the example both reference
# that one .mlb; there are no vendored dependencies.

MLTON      ?= mlton
POLY       ?= poly
BIN        := bin
LIBDIR     := lib/github.com/sjqtentacles/sml-autodiff
CORE       := $(LIBDIR)/autodiff.sig $(LIBDIR)/autodiff.sml $(LIBDIR)/autodiff.mlb
TEST_MLB   := test/sources.mlb
SRCS       := $(CORE) $(wildcard test/*.sml) $(TEST_MLB)

.PHONY: all test poly test-poly all-tests example clean

all: $(BIN)/test-mlton

$(BIN)/test-mlton: $(SRCS) | $(BIN)
	$(MLTON) -output $@ $(TEST_MLB)

test: $(BIN)/test-mlton
	$(BIN)/test-mlton

# Poly/ML has no native .mlb support; the suite runs at top level and exits on
# its own.  The library is basis-only, so we just load the sig, the
# implementation, the harness/helpers, then the per-mode suites and driver in
# dependency order.
poly test-poly:
	printf 'use "$(LIBDIR)/autodiff.sig";\nuse "$(LIBDIR)/autodiff.sml";\nuse "test/harness.sml";\nuse "test/support.sml";\nuse "test/test_forward.sml";\nuse "test/test_reverse.sml";\nuse "test/test_hessian.sml";\nuse "test/entry.sml";\nuse "test/main.sml";\n' | $(POLY) -q --error-exit

all-tests: test test-poly

example: $(BIN)/demo
	mkdir -p assets
	./$(BIN)/demo

$(BIN)/demo: $(CORE) examples/demo.sml examples/sources.mlb | $(BIN)
	$(MLTON) -output $@ examples/sources.mlb

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f $(BIN)/test-mlton $(BIN)/demo
