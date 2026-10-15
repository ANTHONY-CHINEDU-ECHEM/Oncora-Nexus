# API reference

Base URL: `http://localhost:8000`. Interactive documentation at `/docs`.

## Patient profile

All clinical endpoints accept the same deidentified profile. Every field is optional.

```json
{
  "age": 63, "sex": "female", "menopausal_status": "postmenopausal", "ecog": 0,
  "setting": "early", "tumour_size_mm": 32, "grade": 3, "positive_nodes": 2,
  "er_status": "positive", "pr_status": "positive", "her2_status": "negative",
  "brca": "unknown", "recurrence_score": null, "pdl1_cps": null,
  "pik3ca_mutation": null, "esr1_mutation": null, "residual_disease": null,
  "prior_therapies": ["letrozole"], "brain_metastases": false, "visceral_crisis": false,
  "lvef": null, "pregnant": false
}
```

Allowed values: `setting` early, locally_advanced or metastatic; `er_status` positive, low, negative or unknown; `her2_status` positive, low, negative or unknown; `brca` brca1, brca2, negative or unknown.

## Endpoints

### POST /api/ask
Request: `{"question": "...", "case_context": "...", "sources": ["guideline", "pubmed", "trial"], "mode": "auto", "k": 8}`

Response fields: `answer` (structured text with [S#] citations), `mode` (llm, extractive or blocked), `evidence` (ranked passages with features), `verification` (per sentence status, faithfulness, unsupported numbers), `confidence`, `query` (intents and entities), `redactions`, `model`, `prompt_version`, `latency_ms`.

### POST /api/ask/stream
Server sent events: `evidence` (passages), repeated `token`, then `final` with the verified response.

### POST /api/search
`{"query": "...", "sources": null, "k": 10}` returns ranked passages with their eleven feature values.

### POST /api/prognosis
`{"patient": {...}, "model": "core"}` returns survival at 36, 60 and 120 months, monthly curve with baseline, risk group, relative hazard, feature contributions, NPI, imputed fields and caveats.

### POST /api/subtype
`{"patient": {...}, "expression": {"ESR1": 10.2, "ERBB2": 9.1}}` returns pathology surrogate probabilities and, when expression is given, the PAM50 expression model prediction with missing genes listed.

### POST /api/pathway
`{"patient": {...}}` returns fired rules with category, strength, text and linked knowledge cards.

### POST /api/trials/match
`{"patient": {...}, "query": "palbociclib", "top_k": 20, "include_excluded": false}` returns tier counts and ranked trials with every criterion verdict, explanation and source line.

### POST /api/case/brief
`{"patient": {...}, "question": null, "mode": "auto"}` returns pathway, prognosis (early disease), subtype, top trials and an evidence answer for the case.

### GET endpoints
`/api/health`, `/api/meta`, `/api/cards`, `/api/documents/{doc_id}`, `/api/trials/{nct_id}`, `/api/cohorts`, `/api/evaluation`, `/api/audit?limit=50`.
