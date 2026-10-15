# Architecture

## Principles

1. **The language model is a writer, never a source.** Every factual statement must come from a retrieved passage and carry its citation; the verifier checks this after generation.
2. **Local by default.** Retrieval, models and generation run on hospital hardware. The default language model is served by Ollama; nothing is sent to a hosted API.
3. **Graceful degradation.** If the model is unavailable the system returns an extractive evidence brief; if the server is unavailable the interface runs a browser engine with the same contract.
4. **One source of truth per behaviour.** The tokenizer lexicon, query rules, pathway rules, Cox coefficients and parsed trial criteria are exported from Python and consumed unchanged by the browser engine.
5. **Everything measurable is measured.** Each component has an evaluation in `oncora/eval`, and the interface surfaces the results.

## Components

| Layer | Module | Responsibility |
|---|---|---|
| Acquisition | `oncora/data/acquire.py` | Declarative source manifest, resumable verified downloads, streamed PAM50 extraction from a 690 MB matrix |
| Literature | `oncora/data/pubmed_rct.py` | Parse PubMed 200k RCT, breast oncology relevance scoring, topic and study design tagging |
| Trials | `oncora/data/trials.py` | Parse NLI4CT and live ClinicalTrials.gov records into `TrialRecord`, infer population setting |
| Cohorts | `oncora/data/cohorts.py` | Harmonise METABRIC, GBSG2 and TCGA onto one clinical schema, Kaplan Meier and log rank |
| Text | `oncora/rag/text.py` | Clinical normalisation, tokenisation, stemming, number extraction |
| Chunking | `oncora/rag/documents.py` | Section aware passages with evidence levels and line provenance |
| Index | `oncora/rag/bm25.py`, `dense.py`, `index.py` | Precomputed BM25 weights in a sparse matrix, LSA or sentence transformer embeddings, persistence |
| Retrieval | `oncora/rag/retriever.py`, `query.py` | Query understanding, candidate generation, fusion, eleven feature reranker, diversity cap |
| Generation | `oncora/rag/llm.py`, `prompts.py`, `composer.py` | Ollama client with streaming, versioned prompts, extractive fallback |
| Verification | `oncora/rag/verify.py` | Sentence status, numeric fidelity, citation validity, faithfulness score |
| Safety | `oncora/rag/guard.py` | PHI redaction, scope and crisis detection, prompt injection scrubbing |
| Models | `oncora/models/*` | Survival, subtype, trial matcher, pathway rules, cohort analytics |
| Service | `oncora/api/*` | FastAPI routes, schemas, service container, append only audit log |
| Interface | `frontend/src` | React views, shared engine contract, browser engine in `src/engine` |

## Request flow for a question

1. The question and optional case summary are redacted (NHS numbers, dates, names, contacts, postcodes, hospital identifiers).
2. The guard declines out of scope questions and answers crisis language with support services.
3. Query understanding extracts intents (efficacy, toxicity, eligibility, prognosis, testing, supportive, recommendation), drugs, subtype, setting and population.
4. BM25 and the dense encoder each nominate 60 candidates; reciprocal rank fusion merges them.
5. The reranker scores candidates on eleven features; a diversity step caps each document at two passages and returns eight.
6. Retrieved text is scrubbed of instruction like content before entering the prompt.
7. The local model writes a structured answer (bottom line, key evidence, caveats) citing passages as [S1], [S2].
8. The verifier scores each sentence, flags invented numbers and invalid citations, and the confidence level is computed.
9. The response, evidence, verification and query analysis are returned; a hashed, identifier free record goes to the audit log.

## Two engines, one contract

`frontend/src/engine/types.ts` defines the `Engine` interface. `HttpEngine` calls the FastAPI service; `BrowserEngine` loads `frontend/public/demo` (corpus with precomputed tokens, lexicon, lexical reranker, Cox and subtype coefficients, pathway rules, parsed trial criteria, cohort analytics and reports) and executes the same steps in TypeScript. The browser engine omits the dense encoder to keep the bundle to 5.7 MB and uses a reranker trained on lexical features only; its retrieval quality is measured and reported separately.

## Scaling

The bundled corpus holds 13,660 passages and answers a query in about 5 ms of retrieval time. BM25 scoring is a sparse column slice and row sum, so it scales linearly with postings; at a million passages a sharded inverted index or an approximate nearest neighbour library (FAISS, hnswlib) for the dense side would replace the in memory matrices without changing the retriever interface.
