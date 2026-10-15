// Ports of oncora/models/patient.py, survival.cox_predict, subtype.predict_surrogate
// and the trial matcher's criterion evaluation.

import type { CriterionResult, PathwayRule, PatientProfile, PrognosisResponse, TrialMatch, TrialMatchResponse } from "./types";
import { tokenize } from "./text";

export function derived(p: PatientProfile): Record<string, any> {
  const d: Record<string, any> = { ...p };
  d.hr_positive = p.er_status === "positive" || p.er_status === "low" || p.pr_status === "positive";
  d.er_low = p.er_status === "low";
  d.her2_positive = p.her2_status === "positive";
  d.her2_low = p.her2_status === "low";
  d.triple_negative = p.er_status === "negative" && p.pr_status !== "positive" && (p.her2_status === "negative" || p.her2_status === "low");
  const nodes = p.positive_nodes;
  d.node_positive = nodes != null && nodes >= 1;
  d.high_risk_nodes = nodes != null && (nodes >= 4 || (nodes >= 1 && (p.grade === 3 || (p.tumour_size_mm ?? 0) >= 50)));
  d.brca_carrier = p.brca === "brca1" || p.brca === "brca2";
  const pt = new Set(p.prior_therapies.map((t) => t.toLowerCase()));
  d.prior_cdk46 = ["cdk4_6_inhibitor", "palbociclib", "ribociclib", "abemaciclib"].some((t) => pt.has(t));
  d.prior_her2_therapy = ["trastuzumab", "pertuzumab", "t_dm1", "trastuzumab_emtansine"].some((t) => pt.has(t));
  d.prior_therapies = Array.from(pt).sort();
  for (const k of Object.keys(d)) if (d[k] === undefined) d[k] = null;
  return d;
}

export function npi(p: PatientProfile): { value: number; group: string } | null {
  if (p.tumour_size_mm == null || p.grade == null || p.positive_nodes == null) return null;
  const nodeStage = p.positive_nodes === 0 ? 1 : p.positive_nodes <= 3 ? 2 : 3;
  const v = Math.round((0.2 * (p.tumour_size_mm / 10) + p.grade + nodeStage) * 100) / 100;
  return { value: v, group: v <= 3.4 ? "good" : v <= 5.4 ? "moderate" : "poor" };
}

export function summary(p: PatientProfile): string {
  const parts: string[] = [];
  if (p.age) parts.push(`${Math.round(p.age)} year old ${p.sex}`);
  if (p.menopausal_status !== "unknown") parts.push(p.menopausal_status);
  parts.push(`${p.setting.replace("_", " ")} breast cancer`);
  if (p.tumour_size_mm != null) parts.push(`${p.tumour_size_mm} mm`);
  if (p.grade) parts.push(`grade ${p.grade}`);
  if (p.positive_nodes != null) parts.push(`${p.positive_nodes} positive node${p.positive_nodes === 1 ? "" : "s"}`);
  parts.push(`ER ${p.er_status}, PR ${p.pr_status}, HER2 ${p.her2_status}`);
  if (p.brca === "brca1" || p.brca === "brca2") parts.push(`germline ${p.brca.toUpperCase()} carrier`);
  if (p.residual_disease) parts.push("residual disease after neoadjuvant therapy");
  if (p.prior_therapies.length) parts.push(`prior ${p.prior_therapies.join(", ")}`);
  return parts.join(", ");
}

// Pathway rules
function check(c: any, d: Record<string, any>): boolean {
  const { op, field } = c;
  if (op === "any") return c.value.some((x: any) => check(x, d));
  if (op === "all") return c.value.every((x: any) => check(x, d));
  const v = d[field];
  const t = c.value;
  if (op === "true") return v === true;
  if (op === "false") return v === false || v == null;
  if (op === "missing") return v == null;
  if (v == null) return op === "in" && Array.isArray(t) && t.includes(null);
  switch (op) {
    case "eq":
      return v === t;
    case "in":
      return t.includes(v);
    case "gt":
      return v > t;
    case "gte":
      return v >= t;
    case "lt":
      return v < t;
    case "lte":
      return v <= t;
    case "includes_any":
      return (v as string[]).some((x) => t.includes(x));
  }
  throw new Error(`Unknown operator ${op}`);
}

export function evaluatePathway(p: PatientProfile, rules: any, cards: Record<string, { title: string }>): PathwayRule[] {
  const d = derived(p);
  const order: Record<string, number> = { standard: 0, consider: 1, caution: 2 };
  return rules.rules
    .filter((r: any) => r.when.every((c: any) => check(c, d)))
    .map((r: any) => ({
      id: r.id,
      category: r.category,
      strength: r.strength,
      title: r.title,
      text: r.text,
      cards: r.cards,
      card_titles: Object.fromEntries(r.cards.filter((c: string) => cards[c]).map((c: string) => [c, cards[c].title])),
    }))
    .sort((a: PathwayRule, b: PathwayRule) => order[a.strength] - order[b.strength] || a.id.localeCompare(b.id));
}

// Survival
export function survivalFeatures(p: PatientProfile): Record<string, number | null> {
  let hormone = p.hormone_therapy_planned;
  if (hormone == null && p.er_status !== "unknown") hormone = p.er_status === "positive" || p.er_status === "low";
  return {
    age: p.age ?? null,
    tumour_size_mm: p.tumour_size_mm ?? null,
    grade: p.grade ?? null,
    positive_nodes: p.positive_nodes ?? null,
    er_positive: p.er_status === "unknown" ? null : p.er_status === "positive" || p.er_status === "low" ? 1 : 0,
    pr_positive: p.pr_status === "unknown" ? null : p.pr_status === "positive" ? 1 : 0,
    postmenopausal: p.menopausal_status === "unknown" ? null : p.menopausal_status === "postmenopausal" ? 1 : 0,
    hormone_therapy: hormone == null ? null : hormone ? 1 : 0,
  };
}

export function coxPredict(spec: any, p: PatientProfile): PrognosisResponse {
  const feats = survivalFeatures(p);
  const z: number[] = [];
  const contributions = spec.features.map((f: string, i: number) => {
    let val = feats[f];
    const imputed = val == null;
    if (val == null) val = spec.cohort_reference[f];
    const tr = spec.transforms[f];
    let x = Number(val);
    if (tr.kind === "scale") x *= tr.arg;
    else if (tr.kind === "cap_scale") x = Math.min(x, tr.arg[0]) * tr.arg[1];
    else if (tr.kind === "log1p") x = Math.log1p(Math.max(x, 0));
    const zi = (x - spec.mean[i]) / spec.scale[i];
    z.push(zi);
    return { feature: f, label: spec.feature_labels[f], value: val, log_hazard: Math.round(spec.coef[i] * zi * 1e4) / 1e4, imputed };
  });
  const lp = spec.coef.reduce((a: number, c: number, i: number) => a + c * z[i], 0);
  const s0: number[] = spec.baseline_survival.survival;
  const curve = s0.map((s) => Math.round(Math.pow(s, Math.exp(lp)) * 1e4) / 1e4);
  const at = (h: number) => curve[Math.min(h, curve.length - 1)];
  const group = lp < spec.risk_group_cuts[0] ? "low" : lp < spec.risk_group_cuts[1] ? "intermediate" : "high";
  const imputed = contributions.filter((c: any) => c.imputed).map((c: any) => c.feature);
  const caveats = [
    "Estimates reflect METABRIC patients diagnosed between 1977 and 2005 under the treatments they received; contemporary therapy, including HER2 targeted agents, will usually improve outcomes.",
    "External validation on GBSG2 showed good discrimination but optimistic absolute estimates (observed survival lower than predicted), so absolute values should be reconciled with PREDICT Breast.",
    "This is a prognostic model. It does not estimate what would happen under a different treatment choice.",
  ];
  if (imputed.length) caveats.unshift(`Missing inputs were set to the cohort median: ${imputed.join(", ")}.`);
  return {
    linear_predictor: Math.round(lp * 1e4) / 1e4,
    relative_hazard: Math.round(Math.exp(lp) * 1e4) / 1e4,
    risk_group: group,
    survival_at: { "36m": at(36), "60m": at(60), "120m": at(120) },
    curve: { time: spec.baseline_survival.time, survival: curve, baseline: s0.map((s) => Math.round(s * 1e4) / 1e4) },
    contributions: contributions.sort((a: any, b: any) => Math.abs(b.log_hazard) - Math.abs(a.log_hazard)),
    model: "cox_ph_ridge_core",
    endpoint: spec.endpoint,
    imputed_fields: imputed,
    npi: npi(p),
    training_n: spec.training_n,
    caveats,
  };
}

export function surrogateSubtype(spec: any, p: PatientProfile) {
  const n = npi(p);
  const feats: Record<string, number | null> = {
    er_positive: p.er_status === "unknown" ? null : p.er_status === "positive" || p.er_status === "low" ? 1 : 0,
    pr_positive: p.pr_status === "unknown" ? null : p.pr_status === "positive" ? 1 : 0,
    her2_positive: p.her2_status === "unknown" ? null : p.her2_status === "positive" ? 1 : 0,
    grade: p.grade ?? null,
    age: p.age ?? null,
    npi: n ? n.value : null,
  };
  const z = spec.features.map((f: string, i: number) => ((feats[f] ?? spec.medians[f]) - spec.mean[i]) / spec.scale[i]);
  const logits = spec.coef.map((row: number[], k: number) => row.reduce((a, c, i) => a + c * z[i], spec.intercept[k]));
  const m = Math.max(...logits);
  const ex = logits.map((l: number) => Math.exp(l - m));
  const sum = ex.reduce((a: number, b: number) => a + b, 0);
  const probs: Record<string, number> = Object.fromEntries(spec.classes.map((c: string, k: number) => [c, Math.round((ex[k] / sum) * 1e4) / 1e4]));
  const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
  return {
    predicted: top,
    probabilities: probs,
    description: spec.descriptions[top] ?? "",
    model_note: "Pathology surrogate: cross validated macro F1 about 0.42 against PAM50. Treat as a prior, not a call.",
  };
}

// Trials
const EXPANDS: Record<string, string[]> = {
  chemotherapy: ["chemotherapy", "taxane", "anthracycline", "capecitabine", "platinum"],
  endocrine_therapy: ["endocrine_therapy", "aromatase_inhibitor", "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant"],
  aromatase_inhibitor: ["aromatase_inhibitor", "letrozole", "anastrozole", "exemestane"],
  cdk4_6_inhibitor: ["cdk4_6_inhibitor", "palbociclib", "ribociclib", "abemaciclib"],
  trastuzumab: ["trastuzumab", "t_dm1", "trastuzumab_emtansine"],
};

function patientTherapies(p: PatientProfile): Set<string> {
  const base = new Set(p.prior_therapies.map((t) => t.toLowerCase()));
  const out = new Set(base);
  for (const [g, members] of Object.entries(EXPANDS)) if (members.some((m) => base.has(m))) out.add(g);
  if (["tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant", "aromatase_inhibitor"].some((t) => base.has(t))) out.add("endocrine_therapy");
  return out;
}

function evaluateCriterion(c: any, p: PatientProfile, d: Record<string, any>): CriterionResult {
  const excl = c.kind === "exclusion";
  const r = c.requirement;
  let verdict: CriterionResult["verdict"] = "unknown";
  let explanation = "";
  const set = (holds: boolean | null, why: string) => {
    explanation = why;
    if (holds === null) verdict = "unknown";
    else verdict = (excl ? !holds : holds) ? "met" : "not_met";
  };
  switch (c.ctype) {
    case "age":
      if (p.age == null) set(null, "Age not recorded");
      else set(p.age >= (r.min ?? 0) && p.age <= (r.max ?? 200), `Age ${p.age} against range ${r.min ?? "any"} to ${r.max ?? "any"}`);
      break;
    case "ecog":
      set(p.ecog == null ? null : p.ecog <= r.max, p.ecog == null ? "ECOG not recorded" : `ECOG ${p.ecog} against maximum ${r.max}`);
      break;
    case "her2":
      if (p.her2_status === "unknown") set(null, "HER2 status not recorded");
      else set(p.her2_status === r.status || (r.status === "negative" && p.her2_status === "low"), `HER2 ${p.her2_status}; criterion refers to HER2 ${r.status}`);
      break;
    case "hormone":
      if (p.er_status === "unknown") set(null, "Receptor status not recorded");
      else if (r.status === "triple_negative") set(d.triple_negative, "Triple negative status from ER, PR and HER2");
      else set(d.hr_positive === (r.status === "positive"), `Hormone receptor ${d.hr_positive ? "positive" : "negative"}; criterion refers to ${r.status}`);
      break;
    case "setting": {
      const have = p.setting === "metastatic" ? "metastatic" : "early";
      set(have === r.setting, `Disease setting ${p.setting.replace("_", " ")}; criterion refers to ${r.setting} disease`);
      break;
    }
    case "menopause":
      if (p.menopausal_status === "unknown") set(null, "Menopausal status not recorded");
      else set(p.menopausal_status === r.status, `Patient is ${p.menopausal_status}`);
      break;
    case "brca":
      if (p.brca === "unknown") set(null, "Germline BRCA status unknown");
      else set(d.brca_carrier, `BRCA status ${p.brca}`);
      break;
    case "brain_mets":
      if (excl) {
        if (p.brain_metastases && r.allowed_if_treated) {
          verdict = "unknown";
          explanation = "Brain metastases present; trial allows treated or stable disease, verify";
        } else if (p.brain_metastases) {
          verdict = "not_met";
          explanation = "Brain metastases present";
        } else {
          verdict = "met";
          explanation = "No brain metastases recorded";
        }
      } else set(p.brain_metastases, "Trial requires brain metastases");
      break;
    case "pregnancy":
      verdict = p.pregnant ? "not_met" : "met";
      explanation = p.pregnant ? "Patient is pregnant" : "Not pregnant";
      break;
    case "lvef":
      set(p.lvef == null ? null : p.lvef >= r.min, p.lvef == null ? "LVEF not recorded" : `LVEF ${p.lvef} percent against minimum ${r.min}`);
      break;
    case "prior_therapy": {
      const rec = patientTherapies(p);
      const hits = (r.therapies as string[]).filter((t) => rec.has(t));
      if (!p.prior_therapies.length && p.setting !== "metastatic") set(false, "No prior therapy recorded");
      else set(hits.length > 0, hits.length ? `Received ${hits.join(", ")}` : "No matching prior therapy recorded");
      break;
    }
  }
  return { line: c.line, text: c.text, kind: c.kind, ctype: c.ctype, requirement: r, verdict, explanation };
}

export class BrowserTrialMatcher {
  private tokenSets: Set<string>[];
  constructor(private trials: any[]) {
    this.tokenSets = trials.map((t) => new Set((t.index_text as string).split(" ")));
  }

  get count(): number {
    return this.trials.length;
  }

  match(p: PatientProfile, query = "", topK = 25, includeExcluded = false): TrialMatchResponse {
    const d = derived(p);
    const q = new Set(tokenize(`${query} ${summary(p)}`));
    const rel = this.tokenSets.map((s) => {
      let n = 0;
      for (const t of q) if (s.has(t)) n++;
      return n / Math.max(q.size, 1);
    });
    const relMax = Math.max(...rel, 1e-9);
    const tierOrder: Record<string, number> = { "likely eligible": 0, "needs review": 1, excluded: 2 };
    const rows = this.trials.map((t, j) => {
      const crit = t.criteria.map((c: any) => evaluateCriterion(c, p, d));
      const met = crit.filter((c: CriterionResult) => c.verdict === "met").length;
      const notMet = crit.filter((c: CriterionResult) => c.verdict === "not_met").length;
      const unknown = crit.filter((c: CriterionResult) => c.verdict === "unknown").length;
      const tier: TrialMatch["tier"] = notMet ? "excluded" : unknown === 0 && met >= 2 ? "likely eligible" : "needs review";
      const score = 0.45 * (met / Math.max(crit.length, 1)) + 0.25 * (Math.min(met, 8) / 8) + 0.3 * (rel[j] / relMax);
      return { t, crit, met, notMet, unknown, tier, score };
    });
    rows.sort((a, b) => tierOrder[a.tier] - tierOrder[b.tier] || b.score - a.score);
    const tiers: Record<string, number> = { "likely eligible": 0, "needs review": 0, excluded: 0 };
    rows.forEach((r) => tiers[r.tier]++);
    const out: TrialMatch[] = [];
    for (const r of rows) {
      if (r.tier === "excluded" && !includeExcluded) continue;
      out.push({
        nct_id: r.t.nct_id,
        title: r.t.title,
        url: `https://clinicaltrials.gov/study/${r.t.nct_id}`,
        setting: r.t.setting,
        agents: r.t.agents,
        arms: r.t.arms,
        tier: r.tier,
        score: Math.round(r.score * 1e4) / 1e4,
        criteria_met: r.met,
        criteria_not_met: r.notMet,
        criteria_unknown: r.unknown,
        manual_review_items: r.t.manual_lines,
        criteria: r.crit,
      });
      if (out.length >= topK) break;
    }
    return {
      summary: summary(p),
      tiers,
      trials: out,
      data_note: "Demonstration corpus: 999 breast cancer trials with posted results (NLI4CT). The full service can screen currently recruiting studies from ClinicalTrials.gov.",
    };
  }
}
