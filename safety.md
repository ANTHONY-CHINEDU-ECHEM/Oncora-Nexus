# Safety and governance

## Threats and controls

| Risk | Control | Where |
|---|---|---|
| Invented facts or numbers | Answers composed only from retrieved passages; sentence level verification with numeric fidelity; unsupported sentences flagged in the interface | `rag/prompts.py`, `rag/verify.py`, `Evidence.tsx` |
| Overconfidence | Confidence level per answer; evidence level per passage; caveats section required by the prompt | `rag/pipeline.py` |
| Patient identifiers leaking into logs or models | Regex redaction of NHS numbers, dates, names with titles, emails, phone numbers, postcodes and hospital identifiers before any processing; audit log stores only a hash of the redacted input | `rag/guard.py`, `api/audit.py` |
| Prompt injection through retrieved documents | Instruction like text is neutralised before entering the context window | `rag/guard.py` |
| Off label use by non clinicians | Scope guard declines non oncology questions; disclaimer on every page; prompt refuses individual dosing | `rag/guard.py`, `App.tsx` |
| Distress | Crisis language returns UK support services (999, Samaritans 116 123, NHS 111) and stops processing | `rag/guard.py` |
| Automation bias in treatment options | Options labelled standard, consider or caution, each linked to evidence; framed as options for MDT discussion | `models/therapy_rules.json` |
| Model drift and silent changes | Versioned prompts, model registry with data fingerprints, reproducible reports | `rag/prompts.py`, `data/models/registry.json` |

## Audit

Every call to ask, prognosis, trial matching and case brief writes a row with timestamp, request identifier, endpoint, SHA 256 of the redacted input, generation mode, model, prompt version, latency and outcome metrics. `GET /api/audit` returns recent rows and summary statistics.

## Regulatory note

Software that informs treatment decisions may be a medical device under UK MDR 2002 and EU MDR 2017/745. This project is a research prototype and has not been through clinical safety assessment (DCB0129 and DCB0160 in the NHS), clinical validation or regulatory review.
