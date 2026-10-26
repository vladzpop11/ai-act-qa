# AI Act Q&A: an LLM application tested like a product

[![CI](https://github.com/vladzpop11/ai-act-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/vladzpop11/ai-act-qa/actions/workflows/ci.yml)

Answers questions about the EU AI Act from its official consolidated text, citing the passages
it relies on, with open-weight models that run on a laptop. The point is not the chatbot but
the testing around it: retrieval and answers are measured on a reviewed question set, every
design choice is made on development questions by a rule written before the run, and the test
questions are used once.

**Status: stage 1 of 3** (data, retrieval and the evaluation set). Stage 2 adds the answering
model, served with llama.cpp on the Mac and evaluated on Kaggle GPUs with the same model file;
stage 3 the product QA layer (API, adversarial tests, flakiness, load and accessibility tests).

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

## How to run

```bash
make setup        # uv sync --locked
make data         # check the pinned document (download if missing) and parse it
make test         # ruff and pytest
make retrieval    # score the configured retrieval on the development questions
make retrieval-experiments   # compare the settings in configs/config.yaml
make review       # the question review page
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
