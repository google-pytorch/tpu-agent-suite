.PHONY: help build build-experimental verify-experimental dist archives verify validate-kb lint coverage coverage-html coverage-diff test test-all test-framework test-package test-knowledge-base test-scripts test-e2e clean

PYTHON ?= python3
COVERAGE_RC := .coveragerc
DIFF_BASE ?= origin/main
KB_BASE ?=
DIFF_COVERAGE_MIN ?= 80
# Second manifest in package/: tpu-agent-suite-experimental, a complete package for
# testing that releases everything tpu-agent-suite does plus the vLLM
# optimization skills. Install one of the two packages in a workspace, not both.
EXPERIMENTAL_MANIFEST := package.experimental.json
EXPERIMENTAL_NAME := tpu-agent-suite-experimental

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Primary targets:"
	@echo "  build           Build distributions for all target harnesses (dist/)"
	@echo "  build-experimental  Build $(EXPERIMENTAL_NAME) from $(EXPERIMENTAL_MANIFEST) (dist/)"
	@echo "  archives        Create standalone .tar.gz release archives (dist/archives/)"
	@echo "  verify          Gate: manifest, graph, schemas, enforced content checks, knowledge base"
	@echo "  verify-experimental Gate: the same package checks for $(EXPERIMENTAL_MANIFEST)"
	@echo "  validate-kb     Knowledge base index/log integrity (KB_BASE=origin/main also requires a log entry for changes since that ref)"
	@echo "  lint            Report content problems not covered by verify"
	@echo "  test            Run framework, package and tpu-knowledge-base test suites"
	@echo "  test-all        Also run the bundled companion *_test.py suites"
	@echo "  coverage        Measure framework coverage and enforce the floor"
	@echo "  clean           Remove build artifacts and caches"
	@echo ""
	@echo "Granular test targets:"
	@echo "  test-framework      Run framework unit tests only"
	@echo "  test-package        Run package translation, parity, and lint tests only"
	@echo "  test-knowledge-base Run the tpu-knowledge-base index resolve and search tests only"
	@echo "  test-scripts        Run companion *_test.py suites (local only; needs extra deps)"
	@echo "  test-e2e            Run live harness CLI tests (needs claude/agy/cursor/codex on PATH)"
	@echo ""
	@echo "Coverage targets (need: pip install coverage diff-cover):"
	@echo "  coverage-html   Coverage plus a browsable HTML report"
	@echo "  coverage-diff   Coverage of lines changed vs $(DIFF_BASE)"

verify: validate-kb
	$(PYTHON) framework/compiler.py verify package

verify-experimental:
	$(PYTHON) framework/compiler.py verify package --manifest $(EXPERIMENTAL_MANIFEST)

validate-kb:
	$(PYTHON) package/data/knowledgebase/scripts/validate_kb.py $(if $(KB_BASE),--base $(KB_BASE),)

build:
	$(PYTHON) framework/compiler.py translate package --harness claude-code --output dist/claude-code/tpu-agent-suite
	$(PYTHON) framework/compiler.py translate package --harness codex --output dist/codex/tpu-agent-suite
	$(PYTHON) framework/compiler.py translate package --harness antigravity --output dist/antigravity/tpu-agent-suite
	$(PYTHON) framework/compiler.py translate package --harness cursor --output dist/cursor/tpu-agent-suite

build-experimental:
	$(PYTHON) framework/compiler.py translate package --manifest $(EXPERIMENTAL_MANIFEST) --harness claude-code --output dist/claude-code/$(EXPERIMENTAL_NAME)
	$(PYTHON) framework/compiler.py translate package --manifest $(EXPERIMENTAL_MANIFEST) --harness codex --output dist/codex/$(EXPERIMENTAL_NAME)
	$(PYTHON) framework/compiler.py translate package --manifest $(EXPERIMENTAL_MANIFEST) --harness antigravity --output dist/antigravity/$(EXPERIMENTAL_NAME)
	$(PYTHON) framework/compiler.py translate package --manifest $(EXPERIMENTAL_MANIFEST) --harness cursor --output dist/cursor/$(EXPERIMENTAL_NAME)

dist: build

archives: build
	@rm -rf dist/archives
	$(PYTHON) framework/compiler.py archive --dist-dir dist --output dist/archives


# Exits non-zero when findings exist, like any linter. These checks are
# advisory: keep `lint` out of the *required* status checks rather than
# muting it, so findings stay visible without blocking a merge.
lint:
	$(PYTHON) framework/compiler.py lint package

# Coverage is deliberately NOT part of `make test` since coverage is not necessarily
# testing functionality.

coverage:
	@rm -f .coverage
	-@$(PYTHON) -m coverage run --rcfile=$(COVERAGE_RC) -m unittest discover -s framework/tests -t .
	-@$(PYTHON) -m coverage run --rcfile=$(COVERAGE_RC) --append -m unittest discover -s package/tests
	@$(PYTHON) -m coverage report --rcfile=$(COVERAGE_RC)

coverage-html: coverage
	@$(PYTHON) -m coverage html --rcfile=$(COVERAGE_RC)
	@echo "open htmlcov/index.html"

# Coverage of the lines this branch changed, ignoring untouched legacy code.
# Needs full git history: a shallow clone cannot diff against main.
coverage-diff:
	@rm -f .coverage coverage.xml
	-@$(PYTHON) -m coverage run --rcfile=$(COVERAGE_RC) -m unittest discover -s framework/tests -t .
	-@$(PYTHON) -m coverage run --rcfile=$(COVERAGE_RC) --append -m unittest discover -s package/tests
	@$(PYTHON) -m coverage xml --rcfile=$(COVERAGE_RC)
	@$(PYTHON) -m diff_cover.diff_cover_tool coverage.xml \
		--compare-branch=$(DIFF_BASE) --fail-under=$(DIFF_COVERAGE_MIN)

test-framework:
	$(PYTHON) -m unittest discover -s framework/tests -t . -v

test-package:
	$(PYTHON) -m unittest discover -s package/tests -v

# The tpu-knowledge-base companion suites, which need only the standard library.
# `test-package` does not find them: they are named *_test.py, in a directory
# that is not an importable package.
test-knowledge-base:
	$(PYTHON) -m unittest discover -s package/tests/tpu-knowledge-base -p "*_test.py" -v

# Scoped to the suites that need only the standard library. Harbor is not a CI
# dependency, so discovering the whole directory errors on collection rather
# than skipping. Install harbor and run the directory directly to get those as well.
test-eval:
	$(PYTHON) -m unittest discover -s eval/tests -p "test_antigravity_session_supervisor.py" -v
	$(PYTHON) -m unittest discover -s eval/tests -p "test_attention_analyze.py" -v
	$(PYTHON) -m unittest discover -s eval/tests -p "test_claude_session_supervisor.py" -v
	$(PYTHON) -m unittest discover -s eval/tests -p "test_invoked_subagent.py" -v
	$(PYTHON) -m unittest discover -s eval/tests -p "test_cpu_batch_verifier.py" -v


# Live harness CLI tests. Each test skips itself when its binary is absent,
# so this is safe to run with only some harnesses installed -- but a run
# that skips everything still exits 0, so check the skip count before
# trusting a green result.
test-e2e:
	RUN_HARNESS_CLI_E2E=1 $(PYTHON) -m unittest framework.tests.test_hooks_e2e -v

test-scripts:
	@fail=0; for f in $$(find package -name '*_test.py' | sort); do \
		printf '=== %s\n' "$$f"; \
		( cd "$$(dirname $$f)" && $(PYTHON) "$$(basename $$f)" ) || fail=1; \
	done; exit $$fail

test: test-framework test-package test-knowledge-base

test-all: test test-scripts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf dist/ htmlcov/
	rm -f .coverage coverage.xml
	# inference_test.py writes deployments.json into its cwd
	find . -name deployments.json -path '*/scripts/*' -delete
	rm -f deployments.json
