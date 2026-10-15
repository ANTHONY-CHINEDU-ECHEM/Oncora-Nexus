// Shared response types. They mirror the FastAPI service exactly, so the
// interface renders identically whether data comes from the server or from
// the in browser demo engine.

export type Source = "guideline" | "pubmed" | "trial";

export interface PatientProfile {
  age?: number | null;
  sex: "female" | "male";
  menopausal_status: "premenopausal" | "postmenopausal" | "unknown";
  ecog?: number | null;
  setting: "early" | "locally_advanced" | "metastatic";
  tumour_size_mm?: number | null;
  grade?: number | null;
  positive_nodes?: number | null;
  er_status: "positive" | "low" | "negative" | "unknown";
  pr_status: "positive" | "negative" | "unknown";
  her2_status: "positive" | "low" | "negative" | "unknown";
  ki67_percent?: number | null;
  brca: "brca1" | "brca2" | "negative" | "unknown";
  pdl1_cps?: number | null;
  pik3ca_mutation?: boolean | null;
  esr1_mutation?: boolean | null;
  residual_disease?: boolean | null;
  prior_therapies: string[];
  brain_metastases: boolean;
  visceral_crisis: boolean;
  lvef?: number | null;
  recurrence_score?: number | null;
  pregnant: boolean;
  hormone_therapy_planned?: boolean | null;
}

export interface EvidenceHit {
  chunk_id: string;
  doc_id: string;
  source: Source;
  title: string;
  url: string;
  section: string;
  evidence_label: string;
  evidence_rank: number;
  score: number;
  rank: number;
  text: string;
  features: Record<string, number>;
  meta: Record<string, unknown>;
  line_start?: number | null;
}

export interface SentenceCheck {
  text: string;
  citations: number[];
  support: number;
  numbers_ok: boolean;
  status: "supported" | "partial" | "unsupported" | "uncited" | "meta";
  best_source: number | null;
}

export interface Verification {
  sentences: SentenceCheck[];
  faithfulness: number;
  citation_coverage: number;
  invalid_citations: number[];
  unsupported_numbers: string[];
}

export interface ParsedQuery {
  raw: string;
  tokens: string[];
  intents: string[];
  drugs: string[];
  subtypes: string[];
  settings: string[];
  populations: string[];
  primary_intent: string;
}

export interface AskResponse {
  request_id: string;
  question: string;
  mode: "llm" | "extractive" | "blocked";
  notice?: string | null;
  answer: string;
  query?: ParsedQuery;
  evidence: EvidenceHit[];
  verification: Verification | null;
  confidence: { level: "high" | "moderate" | "low" | "none"; score: number; strong_passages?: number; high_level_evidence?: boolean; faithfulness?: number };
  redactions: Record<string, number>;
  model?: string;
  prompt_version?: string;
  latency_ms: number;
}

export interface Contribution {
  feature: string;
  label: string;
  value: number | null;
  log_hazard: number;
  imputed: boolean;
}

export interface PrognosisResponse {
  linear_predictor: number;
  relative_hazard: number;
  risk_group: "low" | "intermediate" | "high";
  survival_at: Record<string, number>;
  curve: { time: number[]; survival: number[]; baseline: number[] };
  contributions: Contribution[];
  model: string;
  endpoint: string;
  imputed_fields: string[];
  npi: { value: number; group: string } | null;
  training_n: number;
  caveats: string[];
}

export interface SubtypeResponse {
  descriptions: Record<string, string>;
  surrogate?: { predicted: string; probabilities: Record<string, number>; description: string; model_note?: string };
  expression?: { predicted: string; probabilities: Record<string, number>; missing_genes: string[]; description: string };
}

export interface PathwayRule {
  id: string;
  category: string;
  strength: "standard" | "consider" | "caution";
  title: string;
  text: string;
  cards: string[];
  card_titles?: Record<string, string>;
}

export interface CriterionResult {
  line: number;
  text: string;
  kind: "inclusion" | "exclusion";
  ctype: string;
  requirement: Record<string, unknown>;
  verdict: "met" | "not_met" | "unknown";
  explanation: string;
}

export interface TrialMatch {
  nct_id: string;
  title: string;
  url: string;
  setting: string;
  agents: string[];
  arms: string[];
  tier: "likely eligible" | "needs review" | "excluded";
  score: number;
  criteria_met: number;
  criteria_not_met: number;
  criteria_unknown: number;
  manual_review_items: number;
  criteria: CriterionResult[];
}

export interface TrialMatchResponse {
  summary: string;
  tiers: Record<string, number>;
  trials: TrialMatch[];
  latency_ms?: number;
  data_note?: string;
}

export interface KMCurve {
  time: number[];
  survival: number[];
  lower: number[];
  upper: number[];
  at_risk: number[];
  n: number;
  events: number;
}

export interface Card {
  id: string;
  category: string;
  title: string;
  body: string;
  references: string[];
  tags: string[];
}

export interface EngineInfo {
  kind: "server" | "browser";
  label: string;
  llm: string | null;
  passages: number;
  documents: number;
}

export interface Engine {
  info(): Promise<EngineInfo>;
  ask(question: string, opts?: { caseContext?: string; sources?: Source[]; mode?: string }): Promise<AskResponse>;
  prognosis(patient: PatientProfile): Promise<PrognosisResponse>;
  subtype(patient: PatientProfile): Promise<SubtypeResponse>;
  pathway(patient: PatientProfile): Promise<PathwayRule[]>;
  trials(patient: PatientProfile, query?: string, includeExcluded?: boolean): Promise<TrialMatchResponse>;
  cohorts(): Promise<any>;
  evaluation(): Promise<any>;
  cards(): Promise<Card[]>;
  document(docId: string): Promise<{ doc_id: string; source: string; passages: { chunk_id: string; title: string; section: string; text: string; url: string }[] }>;
}
