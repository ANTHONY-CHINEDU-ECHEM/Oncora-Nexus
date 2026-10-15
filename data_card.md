# Data card

| Source | Origin | Licence and terms | Citation |
|---|---|---|---|
| PubMed 200k RCT | github.com/Franck-Dernoncourt/pubmed-rct | Abstracts from PubMed under NLM terms; dataset released for research | Dernoncourt F, Lee JY. PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. IJCNLP 2017 |
| NLI4CT | github.com/ai-systems/nli4ct | Released for SemEval 2023; trial text from ClinicalTrials.gov (US public domain) | Jullien M, Valentino M, Frost H, et al. SemEval 2023 Task 7: Multi Evidence Natural Language Inference for Clinical Trial Data. SemEval 2023 |
| METABRIC clinical, mutations and expression | cBioPortal datahub, brca_metabric | Open Database Licence | Curtis C, et al. Nature 2012; Pereira B, et al. Nature Communications 2016 |
| TCGA BRCA PanCancer Atlas | cBioPortal datahub, brca_tcga_pan_can_atlas_2018 | Open Database Licence; TCGA data use policies | Liu J, et al. An Integrated TCGA Pan Cancer Clinical Data Resource. Cell 2018 |
| GBSG2 | Bundled with scikit survival | Public research dataset | Schumacher M, et al. Journal of Clinical Oncology 1994 |
| Knowledge cards | Authored for this project | CC BY 4.0 | References listed on each card |

## Processing

* **Literature.** All 195,654 abstracts are scored for breast oncology relevance. An abstract qualifies when breast terms or breast specific agents dominate, a cancer term is present, and lactation or infant feeding terms are absent. 2,988 abstracts pass. Sentence role labels are retained; each abstract becomes two or three passages of about 180 words.
* **Trials.** 999 reports, of which 956 mention breast cancer explicitly (the rest are mixed solid tumour studies that enrolled breast cancer patients). Blank lines are kept so that line positions match the NLI4CT gold evidence indices. The population setting is inferred from the inclusion criteria only, because exclusion lists mention metastatic disease even in adjuvant trials.
* **Cohorts.** METABRIC patient and sample tables are merged; 1,812 patients have all eight core variables and a relapse free survival endpoint. GBSG2 receptor values in fmol per mg are dichotomised at 10. TCGA stages are mapped to integers and PAM50 calls to the METABRIC label set.
* **Expression.** The 690 MB METABRIC matrix is streamed and reduced to 51 rows (the 50 PAM50 genes plus the legacy ORC6L symbol), then ORC6L is merged into ORC6.

## Known biases

* METABRIC patients were diagnosed between 1977 and 2005 in the UK and Canada, predominantly of European ancestry, before routine HER2 targeted therapy.
* GBSG2 includes only node positive patients and has a maximum follow up of about seven years.
* The PubMed RCT corpus ends in 2017 and favours trials whose abstracts use structured headings.
* The NLI4CT trials are completed studies with posted results and over represent metastatic disease (342 of 999).

## Identifiers

No dataset contains direct identifiers. The interface accepts only a structured, deidentified profile; any free text is redacted before logging or model calls.
