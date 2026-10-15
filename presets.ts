import type { PatientProfile } from "../engine/types";

export const EMPTY_PATIENT: PatientProfile = {
  age: null,
  sex: "female",
  menopausal_status: "unknown",
  ecog: null,
  setting: "early",
  tumour_size_mm: null,
  grade: null,
  positive_nodes: null,
  er_status: "unknown",
  pr_status: "unknown",
  her2_status: "unknown",
  ki67_percent: null,
  brca: "unknown",
  pdl1_cps: null,
  pik3ca_mutation: null,
  esr1_mutation: null,
  residual_disease: null,
  prior_therapies: [],
  brain_metastases: false,
  visceral_crisis: false,
  lvef: null,
  recurrence_score: null,
  pregnant: false,
  hormone_therapy_planned: null,
};

export interface Preset {
  id: string;
  name: string;
  blurb: string;
  patient: PatientProfile;
}

// Fictional, deidentified teaching cases that exercise different pathways.
export const PRESETS: Preset[] = [
  {
    id: "luminal-high-risk",
    name: "Luminal, node positive, postmenopausal",
    blurb: "63 years, 32 mm grade 3, two nodes, ER and PR positive, HER2 negative",
    patient: { ...EMPTY_PATIENT, age: 63, menopausal_status: "postmenopausal", ecog: 0, tumour_size_mm: 32, grade: 3, positive_nodes: 2, er_status: "positive", pr_status: "positive", her2_status: "negative", ki67_percent: 28 },
  },
  {
    id: "tnbc-young",
    name: "Triple negative, premenopausal",
    blurb: "38 years, 28 mm grade 3, node negative, BRCA status pending",
    patient: { ...EMPTY_PATIENT, age: 38, menopausal_status: "premenopausal", ecog: 0, tumour_size_mm: 28, grade: 3, positive_nodes: 0, er_status: "negative", pr_status: "negative", her2_status: "negative" },
  },
  {
    id: "her2-residual",
    name: "HER2 positive with residual disease",
    blurb: "52 years, residual invasive cancer after neoadjuvant trastuzumab and pertuzumab",
    patient: { ...EMPTY_PATIENT, age: 52, menopausal_status: "postmenopausal", ecog: 1, tumour_size_mm: 35, grade: 3, positive_nodes: 1, er_status: "negative", pr_status: "negative", her2_status: "positive", residual_disease: true, lvef: 58, prior_therapies: ["trastuzumab", "pertuzumab", "taxane"] },
  },
  {
    id: "mbc-post-cdk",
    name: "Metastatic luminal after CDK4/6 inhibitor",
    blurb: "58 years, progression on letrozole and palbociclib, PIK3CA mutated, HER2 low",
    patient: { ...EMPTY_PATIENT, age: 58, setting: "metastatic", menopausal_status: "postmenopausal", ecog: 1, er_status: "positive", pr_status: "positive", her2_status: "low", pik3ca_mutation: true, prior_therapies: ["letrozole", "palbociclib"] },
  },
  {
    id: "older-low-risk",
    name: "Older woman, small low grade tumour",
    blurb: "74 years, 14 mm grade 1, node negative, strongly hormone receptor positive",
    patient: { ...EMPTY_PATIENT, age: 74, menopausal_status: "postmenopausal", ecog: 1, tumour_size_mm: 14, grade: 1, positive_nodes: 0, er_status: "positive", pr_status: "positive", her2_status: "negative", recurrence_score: 12 },
  },
];

export const THERAPY_OPTIONS = [
  ["tamoxifen", "Tamoxifen"],
  ["letrozole", "Letrozole"],
  ["anastrozole", "Anastrozole"],
  ["exemestane", "Exemestane"],
  ["fulvestrant", "Fulvestrant"],
  ["palbociclib", "Palbociclib"],
  ["ribociclib", "Ribociclib"],
  ["abemaciclib", "Abemaciclib"],
  ["trastuzumab", "Trastuzumab"],
  ["pertuzumab", "Pertuzumab"],
  ["t_dm1", "T-DM1"],
  ["taxane", "Taxane"],
  ["anthracycline", "Anthracycline"],
  ["capecitabine", "Capecitabine"],
  ["platinum", "Platinum"],
  ["immunotherapy", "Immunotherapy"],
  ["radiotherapy", "Radiotherapy"],
] as const;
