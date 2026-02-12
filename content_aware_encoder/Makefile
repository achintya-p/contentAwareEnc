.PHONY: run dry clean help

PYTHON ?= python

# ── Quick commands ────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

dry: ## Run pipeline end-to-end in dry_run mode
	$(PYTHON) -m content_aware_encoder.scripts.run_pipeline \
		--config content_aware_encoder/configs/default.yaml \
		--run_name demo \
		--dry_run --verbose

run: ## Run pipeline with real encoding (requires ffmpeg + sample.mp4)
	$(PYTHON) -m content_aware_encoder.scripts.run_pipeline \
		--config content_aware_encoder/configs/default.yaml \
		--run_name real_run

clean: ## Remove all results
	rm -rf results/
