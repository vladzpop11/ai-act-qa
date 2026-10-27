# AI Act Q&A: an LLM application tested like a product

[![CI](https://github.com/vladzpop11/ai-act-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/vladzpop11/ai-act-qa/actions/workflows/ci.yml)

Answers questions about the EU AI Act from its official consolidated text, citing the passages
it relies on, with open-weight models that run on a laptop. The point is not the chatbot but
the testing around it: retrieval and answers are measured on a reviewed question set, every
design choice is made on development questions by a rule written before the run, and the test
questions are used once.

**Status: all 3 stages complete** (stage 1: data and retrieval; stage 2: answering models and judge validation; stage 3: product QA layer with API, adversarial, flakiness, load testing and held-out test evaluation).

**Not legal advice.** The consolidated text is, in its own words, "a documentation tool" with
no legal effect; the authentic versions are those published in the Official Journal.

## Results so far

Retrieval on the **50 answerable development questions** (test questions not used yet).
recall@5: share of expected passages among the 5 chunks given to the model; MRR: mean
reciprocal rank of the first right chunk; words@5: the cost, words given to the model.

| Setting | recall@5 | hit@5 | MRR | words@5 | s/query |
| --- | --- | --- | --- | --- | --- |
| **Configured** (hybrid, reranked, structural chunks with headers) | **0.86** | 0.84 | **0.81** | 564 | 3.3 |
| Dense only | 0.86 | 0.84 | 0.80 | 553 | 3.2 |
| BM25 only | 0.84 | 0.84 | 0.77 | 576 | 3.0 |
| No reranker | 0.80 | 0.78 | 0.69 | 564 | 0.04 |
| Larger reranker (MiniLM-L-12) | 0.88 | 0.86 | 0.81 | 581 | 5.6 |
| Larger embedding (bge-base) | 0.88 | 0.86 | 0.81 | 555 | 2.9 |
| Fixed 160-word windows | 0.90 | 0.90 | 0.80 | 855 | 2.5 |
| No citation header on chunks | 0.83 | 0.82 | 0.76 | 467 | 2.6 |

**The rule, written before the run** ([config.yaml](configs/config.yaml),
`retrieval_experiments`): highest recall@5 wins, but the configured setting is kept unless
another beats it by more than 0.04, about two questions. **It keeps the configured setting.**
Windows are exactly 0.04 better (two questions) for 52% more words, which on this Mac's CPU
adds about 20 seconds to every answer. A first run of the comparison picked the windows,
because 0.90 − 0.86 is 0.0400000000000000036 in floating point: the comparison now rounds, and
a test reproduces the case.

What the comparison shows: the reranker (+0.06 recall, +0.12 MRR) and the citation headers
(+0.03, +0.05) clearly help; the other differences are one or two questions. The misses are
mostly paraphrase and dates: "cheating in exams" against "prohibited behaviour of students
during tests", or the application dates in Article 113, which speaks of "Chapter III,
Sections 1, 2 and 3" rather than "high-risk requirements".

## The document

- **Regulation (EU) 2024/1689 (the AI Act), consolidated version of 27 July 2026**, which
  includes the Digital Omnibus, Regulation (EU) 2026/1744. It changed, among others, the
  application dates of the high-risk rules (Annex III systems from 2 December 2027, Annex I
  systems from 2 August 2028, instead of 2 August 2026 and 2027) and added two prohibitions
  (Article 5(1)(ba) and (bb), on non-consensual intimate imagery and child sexual abuse
  material, from 2 December 2026). Much of what is written online still quotes the 2024 dates,
  which makes these questions a useful test.
- Downloaded from the Publications Office's CELLAR service (EUR-Lex itself answers scripts
  with a bot challenge) and **pinned by SHA-256** in [configs/config.yaml](configs/config.yaml):
  a changed file is refused, so a new consolidated version is adopted deliberately, with a new
  evaluation. The file is committed (832 KB) so that tests and CI need no network.
- Reuse is allowed under Commission Decision 2011/833/EU with the source acknowledged.
  Source: © European Union, [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689).

## Parsing into citable passages

[parse.py](src/ai_act_qa/parse.py) turns the XHTML into one plain canonical text and **1,356
citable units** over it: 119 articles, 556 paragraphs, 667 points and subpoints, 14 annexes.
Each unit has the citation a lawyer would use ("Article 5(1)(ba)", "Article 43(1), second
subparagraph, point (a)", "Annex III, point 4(a)"), its chapter and section, and the acts its
text comes from: the consolidated text marks every passage as original or amended, and 133
units contain Omnibus text. A unit is a span of the canonical text and contains its children's
spans.

Things the source made necessary, each covered by a test:
- unnumbered subparagraphs follow their paragraph as siblings in the markup, and belong to it;
- a paragraph can have two lists of points, both starting at (a) (Article 43(1)): the second
  list is cited with its subparagraph;
- quoted provisions of other acts (Articles 102 to 110 amend other regulations) contain
  numbered paragraphs that are not this Act's;
- superscripts: "10<sup>25</sup> floating point operations" would read "1025" as plain text;
- an article title has a stray apostrophe in the official text ("Subject matter'").

A check against the source confirms every article and annex keeps all its words.

## Retrieval

- **Chunks** ([chunk.py](src/ai_act_qa/chunk.py)): one per paragraph, or per point when a
  paragraph is longer than 220 words, each repeating the sentence that introduces its list
  ("1. The following AI practices shall be prohibited:") and starting with the article's
  citation, title and chapter. Alternative: fixed windows of 160 words. Every chunk records its
  spans of the canonical text, so **any chunking is scored against the same expected
  passages**: a chunk contains a passage when their spans overlap, and a repeated lead-in does
  not count.
- **Search** ([search.py](src/ai_act_qa/search.py)): a dense embedding (BAAI/bge-small-en-v1.5)
  and BM25 keyword vectors in Qdrant, fused by reciprocal rank, then the best 30 reranked by a
  cross-encoder (ms-marco-MiniLM-L-6-v2) and the top 5 kept. All models run on the CPU through
  ONNX (fastembed), with no PyTorch. The same code uses an in-memory Qdrant for evaluation and
  the Qdrant server for the API.

## The evaluation set

[eval/questions.yaml](eval/questions.yaml): **89 questions**, each with the passages that
answer it and a short reference answer: definitions, scope, obligations, prohibitions,
high-risk classification, numbers and deadlines, fines, rights, questions about the Omnibus
changes, several that need two passages, and **8 the Act does not answer** (the service must
say so).

- **Drafted with an AI assistant, checked by a person.** Every question was reviewed against
  the full text of its passages on a review page (`make review`); the verdicts are in
  [eval/review.json](eval/review.json). 84 were correct as written; 2 had a missing passage
  (added); 3 were flagged with dates from the original 2024 Act, and were kept after checking
  that the pinned consolidated text replaced those dates (the reviewer agreed).
- **Split once** into 55 development and 34 test questions, at random within each question type
  with a fixed seed ([eval/split.json](eval/split.json)); a test asserts the split can be
  reproduced, and the command refuses to split again. Development questions are used for every
  choice; test questions once, at the end.

## Answers and the served model

Evaluated on the **55 development questions** (Kaggle GPUs with `llama.cpp` server; the same pinned GGUF file served on the Mac with Docker). Answers were graded by **Gemma 4 31B** against the reference answer and passages; `score` is the share of points (correct = 1, partial = 0.5; declines on answerable questions = 0).

| Model | Mac eligible | Score | Correct | Partial | Grounded | Declines (unans.) | Cites passage | Mac s/answer |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| qwen3.5-0.8b | yes | 0.53 | 0.49 | 0.15 | 0.80 | 0.60 | 0.64 | 23.4 |
| qwen3.5-2b | yes | 0.67 | 0.55 | 0.25 | 0.87 | 0.20 | 0.86 | 45.9 |
| **gemma-4-e2b** | **yes** | **0.74** | **0.69** | 0.16 | **0.96** | 0.80 | 0.75 | 67.7 |
| qwen3.5-4b | no | 0.82 | 0.87 | 0.09 | 1.00 | 1.00 | 0.91 | — |
| qwen3.5-9b | no | 0.80 | 0.85 | 0.11 | 1.00 | 1.00 | 0.98 | — |
| gemma-4-e4b | no | 0.82 | 0.80 | 0.11 | 1.00 | 1.00 | 0.88 | — |

**The selection rule, written before the run** ([config.yaml](configs/config.yaml), `selection`): among the Mac candidates with median answer time $\le 150\text{s}$, the highest development score wins, keeping a smaller model unless a larger one beats it by more than 0.05. **It selects `gemma-4-e2b`** ($0.736 - 0.673 = 0.063 > 0.05$).

### Checking the judge against a person

Because the judge (`gemma-4-31b`) shares a model family with `gemma-4-e2b`, its grades were checked against a person on a blind sample of 48 development answers ([eval/judge_labels.json](eval/judge_labels.json), `make judge-agreement`):
- **Agreement**: 83.3% agreement with rubric (Cohen's $\kappa = 0.65$ on the decisive pair, $\kappa = 0.65$ overall). All 8 disagreements were single-step (`partial` vs `correct` or `incorrect`), with zero polarity reversals.
- **No family bias**: the judge was slightly stricter than the person on both families (Gemma $-0.048$, Qwen $-0.037$), dispelling the concern of pro-Gemma bias.
- **Selection confirmed by human grades**: on the questions where the judge graded `qwen3.5-2b` and `gemma-4-e2b` differently, the person scored `gemma-4-e2b` at 0.722 vs `qwen3.5-2b` at 0.528 (+0.194), matching the judge's margin (+0.195).

## The held-out test evaluation

Retrieval scored once at the end of the project on the **31 answerable test questions** ([reports/retrieval_test.json](reports/retrieval_test.json)):

| Split | Questions | recall@5 | hit@5 | MRR | words@5 | s/query |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development** (used for choices) | 50 | **0.86** | 0.84 | 0.81 | 564 | 3.3 |
| **Test** (held out, used once) | 31 | **0.84** | 0.84 | 0.76 | 551 | 4.0 |

The test performance matches the development baseline within 2 percentage points (less than a single question's variance, which is 3.2 points on 31 questions), confirming that the retrieval design did not overfit.

## Stage 3: Product QA layer

The serving API ([api.py](src/ai_act_qa/api.py), [service.py](src/ai_act_qa/service.py)) exposes the pipeline as a FastAPI service:
- `POST /ask`: takes a question and optional `k`, returns the answer, citation details (statutory article labels, chapters and text), retrieved passages with scores, latency breakdown, and provenance.
- `GET /passages`: allows previewing retrieved passages without invoking the generation model.
- `GET /healthz`, `GET /ready`, `GET /info`: container health and system metadata endpoints.

Tested like a product with 40 unit and integration tests (`make qa`):
- **Adversarial & robustness** ([test_adversarial.py](tests/test_adversarial.py)): prompt injection isolation (system prompt rules remain unmutated), delimiter breaking defense, date traps (asserting 2026 Omnibus dates over 2024 dates), XSS/SQL/CRLF injection payloads, and 2000-character input bounds.
- **Flakiness & determinism** ([test_flakiness.py](tests/test_flakiness.py)): bit-for-bit repeatability across repeated greedy runs ($T=0$), citation parsing stability.
- **Load & concurrency** ([test_load.py](tests/test_load.py)): 20 concurrent queries executed across worker pools verifying thread safety, state isolation, and connection stability.

## How to run

```bash
make setup                   # uv sync --locked
make data                    # check the pinned document (download if missing) and parse it
make test                    # ruff and pytest (all 40 tests)
make qa                      # product QA suite (API, adversarial, flakiness, load)
make api                     # start the FastAPI web service on port 8000
make retrieval               # score the configured retrieval on the development questions
make retrieval-experiments   # compare the settings in configs/config.yaml
make retrieval-test          # score retrieval on the held-out test questions (reports/retrieval_test.json)
make review                  # the question review page
make answers-report          # metrics per model and the served model (reports/answers_dev.json)
make labels                  # build the blind judge-labelling page (eval/judge_labels.html)
make labels-apply FILE=...   # apply human grades to eval/judge_labels.json
make judge-agreement         # score agreement between person and judge (reports/judge_agreement.json)
```

CI ([ci.yml](.github/workflows/ci.yml)) audits the dependencies, checks the pinned document,
runs the tests, checks the questions, and runs the **retrieval regression gate**: the
configured retrieval on the development questions must keep recall@5 ≥ 0.80 and MRR ≥ 0.72.

## Running on an 8 GB Intel Mac

- onnxruntime stopped publishing Intel-Mac wheels after 1.23, so the lock is resolved for both
  this Mac and Linux (`required-environments` in pyproject.toml) and both use 1.23.
- Embedding the whole Act takes about 3.5 minutes on 2 CPU cores, so chunk embeddings are cached
  on disk by model and text; each experiment re-embeds only chunks that changed.
- Reranking 30 candidates takes about 3 seconds per question.

## Limitations

- The consolidated text has no recitals (the preamble), which explain the intent of many
  provisions; questions about intent can only be answered from the articles.
- One reviewer, who is not a lawyer, checked the questions; the reference answers summarise
  the passages and do not interpret them.
- 50 answerable development questions: one question is 2 percentage points of recall, so
  differences of one or two questions between settings are noise, which is why the selection
  rule keeps the configured setting unless another wins by more than 4 points.
