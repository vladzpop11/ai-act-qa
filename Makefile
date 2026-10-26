.PHONY: help setup test data questions split retrieval retrieval-experiments retrieval-test

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

setup:  ## Install dependencies from uv.lock
	uv sync --locked

test:  ## Ruff and pytest
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

data:  ## Check (or download) the pinned AI Act and parse it into citable units
	uv run python -m ai_act_qa.ingest
	uv run python -m ai_act_qa.parse

questions:  ## Check the evaluation questions (passage ids, split)
	uv run python -m ai_act_qa.eval_set check

split:  ## Split the questions into development and test, once (refuses if done)
	uv run python -m ai_act_qa.eval_set split

retrieval:  ## Score the configured retrieval on the development questions (reports/retrieval.json)
	uv run python -m ai_act_qa.retrieval_eval

retrieval-experiments:  ## Compare retrieval settings on the development questions
	uv run python -m ai_act_qa.retrieval_eval --experiments

retrieval-test:  ## Score retrieval on the test questions: once, at the end of the project
	uv run python -m ai_act_qa.retrieval_eval --test
