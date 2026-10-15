# Evaluation protocol

Run everything with:

```bash
python scripts/run_evaluation.py              # retrieval, NLI4CT, answers, figures
python scripts/run_evaluation.py --mode llm   # score answers produced by the local model
python scripts/train_models.py                # survival and subtype reports
python scripts/export_demo.py                 # browser engine retrieval report
```

Outputs land in `reports/` as JSON and in `reports/figures/` as PNG.

## Retrieval (`oncora/eval/retrieval_eval.py`)

| Set | Size | Construction | Use |
|---|---|---|---|
| PubMed known item, train | 1,200 | OBJECTIVE sentence of an abstract with half of its words removed (seeded); target is that abstract | Reranker training |
| PubMed known item, test | 500 | Same construction on disjoint abstracts | Evaluation |
| Card topical | 384 | Random four to eight word subsets of each card's title, tags and body | Reranker training |
| Trial registry | 400 | Results, intervention and adverse event text thinned to a quarter of its words | Reranker training |
| Clinical development | 50 | Authored clinician questions judged against knowledge cards | Evaluation; informed the lay lexicon |
| Clinical blind | 20 | Written after the lexicon and features were frozen, scored once | Headline evaluation |

Metrics are computed at document level after removing duplicate documents from the ranked passage list: recall at 1, 5 and 10, MRR at 10 and nDCG at 10.

## Answers (`oncora/eval/answer_eval.py`)

For each of the 70 clinical questions the full pipeline runs and the report records whether a relevant card was retrieved and cited, faithfulness, citation coverage, numbers absent from cited sources, confidence and latency. Five safety probes test out of scope refusal, crisis signposting, identifier redaction and prompt injection.

## NLI4CT (`oncora/eval/nli4ct_eval.py`)

* Evidence ranking: for each statement and trial section, lines are ranked by the scorer used in verification and compared with a random ordering by mean average precision and precision at 3.
* Evidence selection: set based precision, recall and F1 with a threshold tuned on train and development; on this dataset the optimum is to select every line, which is reported rather than hidden.
* Claim verification: gradient boosting on verifier signals (coverage, numeric agreement, negation, quantifiers, section) trained on 1,900 statements and tested on the 500 gold test statements.

## Survival (`oncora/models/survival.py`)

* Five fold stratified cross validation on METABRIC for three learners and the NPI baseline: Harrell and Uno C index, time dependent AUC at 36, 60 and 120 months, integrated Brier score between 12 and 180 months.
* External validation on GBSG2 at 36 and 60 months.
* Calibration by risk quintile against Kaplan Meier estimates with Greenwood confidence intervals.
* Risk group separation by log rank test in both cohorts; permutation importance of the forest.

## Subtype (`oncora/models/subtype.py`)

Five fold stratified cross validation with accuracy, balanced accuracy, macro F1, Cohen kappa, per class metrics and confusion matrices for expression and pathology models and the St Gallen rule baseline.
