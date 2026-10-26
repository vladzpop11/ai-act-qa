.PHONY: help setup test data questions split retrieval retrieval-experiments retrieval-test review review-apply retrieval-gate mac-latency answers-report labels labels-apply judge-agreement

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

review:  ## Build the question review page (eval/review.html) and open it
	uv run python -m ai_act_qa.review page
	open eval/review.html

review-apply:  ## Record downloaded verdicts: make review-apply FILE=~/Downloads/review.json
	uv run python -m ai_act_qa.review apply "$(FILE)"

retrieval-gate:  ## CI: fail if retrieval on the development questions drops below the gate
	uv run python -m ai_act_qa.retrieval_eval --gate

mac-latency:  ## Seconds per answer of each Mac candidate on this Mac (Docker llama.cpp, CPU)
	uv run python -m ai_act_qa.answer_eval mac-latency

answers-report:  ## Metrics per model and the served model, from the Kaggle results (reports/answers_dev.json)
	uv run python -m ai_act_qa.answer_eval report

labels:  ## Sample answers for checking the judge and build the labelling page (eval/judge_labels.html)
	@test -f eval/judge_sample.json || uv run python -m ai_act_qa.judge_agreement sample
	uv run python -m ai_act_qa.judge_agreement page
	open eval/judge_labels.html

labels-apply:  ## Record downloaded labels: make labels-apply FILE=~/Downloads/judge_labels.json
	uv run python -m ai_act_qa.judge_agreement apply "$(FILE)"

judge-agreement:  ## Agreement between the labels and the judge -> reports/judge_agreement.json
	uv run python -m ai_act_qa.judge_agreement report
