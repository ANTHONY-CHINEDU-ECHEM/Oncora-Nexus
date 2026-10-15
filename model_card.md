# Model card

## Intended use

Decision support for qualified clinicians preparing breast cancer cases for a multidisciplinary team meeting: summarising evidence, presenting guideline aligned options, estimating prognosis, and screening trials. Not intended for patients, for autonomous decisions, or for use outside breast oncology.

## Components

### Hybrid retriever and reranker
* Inputs: free text question and optional deidentified case summary.
* Training: logistic regression on 184,581 query and candidate pairs from 1,950 synthetic queries (PubMed known item, card topical and trial registry queries). Clinical question sets are never used for training.
* Performance on 20 blind clinical questions: recall at 5 of 0.95, MRR at 10 of 0.777 (BM25 alone: 0.75 and 0.712).
* Failure modes: paraphrases with no shared vocabulary (for example durations expressed as weeks rather than fractions); questions about trials published after 2017 that are absent from the literature corpus and from the digests.

### Answer generation and verification
* Default model: Llama 3.1 8B Instruct through Ollama, temperature 0.1, 8,192 token context. Any Ollama model can be substituted.
* Fallback: extractive composer selecting verbatim evidence sentences.
* Verification: lexical support of at least 0.5 for supported claims, numeric fidelity for every multi digit number, citation validity.
* Measured: 93 percent of 70 test answers cite a relevant guideline digest; no answer contains a number absent from its sources (extractive mode).
* Limits: on NLI4CT the lexical and numeric signals reach 0.57 accuracy for entailment versus contradiction against 0.50 by majority, so verification flags unsupported content but cannot prove correctness.

### Relapse free survival model
* Type: Cox proportional hazards with ridge penalty (alpha 0.5), standardised inputs.
* Inputs: age, tumour size, grade, positive nodes (log transformed), ER, PR, menopausal status, endocrine therapy.
* Development: METABRIC, 1,812 patients, 735 events, median follow up 102 months.
* Internal performance: C index 0.650 ± 0.025 (five fold), integrated Brier score 0.184.
* External performance (GBSG2, 686 patients, 299 events): C index 0.669, time dependent AUC 0.727 at 3 years and 0.712 at 5 years, integrated Brier score 0.159. NPI on the same cohort: 0.659.
* Calibration: close to ideal internally; on GBSG2 observed five year relapse free survival is 12 to 17 points below prediction across quintiles. Local recalibration is required before use.
* Not causal: coefficients for treatments reflect who received them, not their effect.

### Subtype models
* Expression: multinomial logistic regression on 50 PAM50 genes, accuracy 0.77 and macro F1 0.74 in five fold cross validation against METABRIC PAM50 plus Claudin low labels (nearest centroid: 0.71 and 0.65).
* Pathology surrogate: multinomial logistic regression on ER, PR, HER2, grade, age and NPI, accuracy 0.46 and macro F1 0.42 (St Gallen rules: 0.50 and 0.37).
* Caveat: reference labels were derived from expression, so the expression result measures portability rather than biological truth.

### Trial criteria parser and matcher
* Rule based extraction of eleven constraint types from free text eligibility criteria; 7,068 constraints from 999 trials.
* Lines describing laboratory thresholds, consent, washout periods or investigator judgement are counted for manual review rather than parsed.
* Every verdict links to the source line. Likely eligible means no structured criterion is violated and none is unknown; it is a prompt for review, not a determination.

### Pathway rules
* 42 declarative rules executed identically in Python and TypeScript, each linked to knowledge cards, labelled standard, consider or caution.

## Ethical considerations

* Automation bias: answers show verification status per sentence and confidence per answer, and every recommendation carries its evidence card, to keep the clinician in the loop.
* Equity: training cohorts under represent non European ancestry and male breast cancer; estimates for under represented groups are less certain.
* Privacy: free text is redacted; the audit log stores only hashes and outcome metrics.
