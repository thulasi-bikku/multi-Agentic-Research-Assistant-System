# MARAS Re-Evaluation Pipeline

This replaces the retrieval evaluation in the MARAS paper with an honest,
reproducible pipeline that runs on the real BEIR benchmarks, and adds the
missing pieces the three reviewers asked for. It does **not** patch the old
numbers — it recomputes everything from scratch on standard protocol.

## Important — run this on your own machine, not in a sandbox

This code downloads real datasets (TREC-COVID, SciFact, NFCorpus, MS MARCO
subset) from the BEIR benchmark host and HuggingFace. That requires open
internet access. I could not execute the full pipeline in this session
because my sandbox only allows a whitelist of package-registry domains
(pypi, npm, github) and can't reach the dataset hosts. I did verify:
- every module imports and compiles cleanly
- the ranking/eval/statistics logic is correct on a synthetic corpus
  (see `tests/test_synthetic.py`, which you can run right now with no
  network access to sanity-check the math)

Run it yourself with:
```bash
pip install -r requirements.txt
python run_full_pipeline.py --datasets trec-covid scifact nfcorpus msmarco --out results/
```
Expect this to take a while (embedding ~300k documents) and to need a GPU
for the semantic/dense stages to be fast enough to be practical.

## What changed, mapped to each reviewer point

### Reviewer 3 (the blocking one)

| # | Comment | Fix | Where |
|---|---|---|---|
| 1 | NDCG@10=0.912 implausible vs BEIR SOTA; protocol undisclosed | Full-collection retrieval (not pool re-ranking), standard BEIR qrels, retrieval depth k=1000 for candidates / metrics cut at 10 & 100, all stated explicitly in `evaluate.py` header and printed in every run's log | `evaluate.py`, `config.py` |
| 2 | Results pooled across 5 datasets, weighting unstated | Per-dataset table is now the primary output; macro-avg (unweighted mean of dataset means) and micro-avg (query-count-weighted) are both computed and *labeled as such* | `evaluate.py::aggregate()` |
| 3 | MRR contradiction (1.000 vs 0.941) + leakage risk | MRR computed exactly once, from per-query reciprocal ranks, no rounding to "1.000" anywhere; `evaluate.py::check_leakage()` flags any query whose top-ranked doc's qrel was created from an identical string match to the query (the actual bug that produces perfect MRR) | `evaluate.py::check_leakage()` |
| 4 | Only self-ablations, no real baselines | Adds (a) literature BM25 numbers for each BEIR set from the BEIR paper, (b) a real dense retriever (`msmarco-distilbert-base-v3` or user-supplied) run locally, alongside the hybrid model, all in the same table | `baselines.py` |
| 5 | Wrong unit of analysis (10 seeds instead of per-query) | Statistical testing is redone as a **paired test across queries**: per-query NDCG@10/Recall for hybrid vs. each baseline, paired t-test + Wilcoxon signed-rank, Holm–Bonferroni correction across the multiple baseline comparisons, plus paired Cohen's d | `significance.py` |
| 6 | Table 5 sensitivity analysis has no numbers, tuning set unstated | Grid search over (α, β, γ) reports actual F1/NDCG@10 numbers per config, computed on a validation query split that is explicitly held out and asserted disjoint from the test split | `sensitivity.py` |
| 7 | Table 7 unsubstantiated checkmarks vs. commercial tools | Old table generation removed. Replaced with a script that only fills a cell if backed by a citation/observation you supply; unverified cells are marked `not verified`, not left blank/checked. A head-to-head shared-query harness is scaffolded for genuine comparison if you have API access to those tools | `honest_comparison.py` |
| 8 | Writing Agent unevaluated | Automated grounding metrics (ROUGE-L overlap + embedding similarity between generated feedback and the retrieved evidence block) computed for every output, plus a required control condition (same LLM, no retrieval context) so you can report a delta, not just a raw score | `writing_eval.py` |
| 9 | Human eval needs raters, IRR, control | `human_eval_protocol.py` generates a rating template with independent-rater columns for all 5 Likert criteria (not just the ArXiv relevance labels), computes Cohen's/Fleiss' κ properly, and requires a control-condition column (LLM-only summaries) before results can be aggregated | `human_eval_protocol.py` |
| 10 | "Multi-agent" overstates a sequential pipeline | Not a code fix — text fix. Suggested rewording is at the bottom of this file. | — |

### Reviewer 1
- Single-LLM ablation baseline added (`baselines.py::single_llm_baseline`) — full pipeline vs one LLM call with no agent decomposition, on the same queries.
- AutoGen-style coordination discussed explicitly vs MARAS's static shared-state pipeline in the ablation report header.

### Reviewer 2
- P/R/F1 now reported for the Writing Agent's structured-extraction sub-task (methodology/dataset/metric/limitation extraction) using the existing gold-labeled setup from Table 1, extended in `writing_eval.py::extraction_prf()`.

## Suggested abstract/intro rewording for point 10 (Reviewer 3)

Replace "multi-agent" claims of autonomy with something like:
> "a modular six-stage pipeline with specialized components and a single
> optional feedback edge, coordinated through a shared-state repository"

and reserve "multi-agent" for the related-work comparison, not the system's
own framing, unless dynamic task allocation is actually implemented.

## Files

```
config.py              run parameters (weights, cutoffs, dataset list)
data_loader.py          BEIR + custom ArXiv-CS loader
retrieval/bm25.py        BM25 lexical scorer
retrieval/semantic.py    sentence-transformer semantic scorer
retrieval/temporal.py    exponential recency scorer
retrieval/hybrid.py      combines the three under Score = a*BM25+b*Sem+g*Rec
baselines.py             BM25-only / semantic-only / TF-IDF-only / dense retriever / single-LLM / published numbers
evaluate.py              pytrec_eval wrapper, per-dataset + macro/micro tables, leakage check
significance.py          paired per-query stats, Holm-Bonferroni, effect sizes
sensitivity.py           real-number grid search over alpha/beta/gamma on held-out val split
ablation.py              runs each ablation config, real numbers table
writing_eval.py          automated writing-agent metrics + control condition + extraction P/R/F1
human_eval_protocol.py   rating template, kappa/ICC, requires control condition
honest_comparison.py     verified-only comparison table generator
run_full_pipeline.py     orchestrates everything, writes CSV + Markdown tables to results/
tests/test_synthetic.py  runs the whole pipeline on a small synthetic corpus — no network needed
```
