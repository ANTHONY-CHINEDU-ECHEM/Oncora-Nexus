// Ports of the guard (oncora/rag/guard.py), the extractive composer
// (oncora/rag/composer.py) and the verifier (oncora/rag/verify.py).

import type { EvidenceHit, ParsedQuery, SentenceCheck, Verification } from "./types";
import { normalise, numbers, splitSentences, tokenize } from "./text";

const PHI: [string, RegExp][] = [
  ["NHS_NUMBER", /\b\d{3}[ ]?\d{3}[ ]?\d{4}\b/g],
  ["EMAIL", /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/g],
  ["PHONE", /(?<!\d)(?:\+44\s?7\d{3}|\(?07\d{3}\)?|\(?0\d{3,4}\)?)\s?\d{3}\s?\d{3,4}(?!\d)/g],
  ["DATE", /\b\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}\b/g],
  ["DATE", /\b\d{1,2}(?:st|nd|rd|th)? (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]* (?:19|20)\d{2}\b/gi],
  ["POSTCODE", /\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b/g],
  ["HOSPITAL_ID", /\b(?:MRN|hospital (?:no|number)|patient id)[:\s#]*[A-Z0-9-]{4,}\b/gi],
  ["NAME", /\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?/g],
];

const SCOPE =
  /breast|cancer|tumou?r|oncolog|carcinoma|mammo|her2|estrogen|oestrogen|progesterone|tamoxifen|letrozole|anastrozole|chemo|radiother|mastectom|lumpectom|metasta|trastuzumab|pertuzumab|brca|lymph|axill|recurrence|survival|trial|biopsy|screening|endocrine|aromatase|palbociclib|ribociclib|abemaciclib|olaparib|pembrolizumab|capecitabine|docetaxel|paclitaxel|anthracycline|neoadjuvant|adjuvant|dcis|lobular|ductal|npi|grade|stage|subtype|luminal|basal|ki67|oncotype|mammaprint|lvef|cardiac|neutropen|nausea|fatigue|lymph?oedema|lymphedema|bone|bisphosphonate|denosumab|fertility|menopaus|hot fl|survivor|palliative|mdt|tumou?r board|ecog|eligib|receptor|\ber\b|\bpr\b|ihc|immunohisto|pam50|hormone|ovar|nodes?\b|sentinel|mri\b|fish\b|margin|radiotherapy|neoplasm|malignan|metastas/i;
const CRISIS = /\b(kill myself|end my life|suicid\w*|self[- ]harm|want to die|overdose on purpose)\b/i;

export const CRISIS_MESSAGE =
  "It sounds like you may be going through something very difficult. If you are in immediate danger, call 999 or go to your nearest A and E. In the UK you can call Samaritans free on 116 123 at any time, or NHS 111. This system is a clinical decision support tool and cannot provide the support you deserve right now.";
export const OUT_OF_SCOPE_MESSAGE =
  "Oncora Nexus answers questions about breast cancer diagnosis, treatment, prognosis, trials and supportive care. This question appears to be outside that scope, so no answer was generated.";

export function redact(text: string): { text: string; counts: Record<string, number> } {
  const counts: Record<string, number> = {};
  let out = text;
  for (const [label, rx] of PHI) {
    out = out.replace(rx, () => {
      counts[label] = (counts[label] ?? 0) + 1;
      return `[${label}]`;
    });
  }
  return { text: out, counts };
}

export function guard(text: string, context?: string) {
  const r = redact(text);
  const crisis = CRISIS.test(text);
  const inScope = SCOPE.test(normalise(`${r.text} ${context ?? ""}`));
  return { ...r, crisis, inScope, blocked: crisis || !inScope, message: crisis ? CRISIS_MESSAGE : !inScope ? OUT_OF_SCOPE_MESSAGE : null };
}

// Composer
const LABEL_RX = /^(BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSIONS):\s*/;
const PREFERRED: Record<string, string[]> = {
  efficacy: ["CONCLUSIONS", "RESULTS"],
  toxicity: ["RESULTS", "CONCLUSIONS"],
  supportive: ["CONCLUSIONS", "RESULTS"],
  recommendation: ["CONCLUSIONS"],
};
const MIN_SENTENCE_SCORE = 0.3;

function trim(s: string): string {
  return s.replace(/\s+([,.;:)\]%])/g, "$1").replace(/([([])\s+/g, "$1").trim().replace(/\.+$/, "");
}

function sentencesOf(hit: EvidenceHit): [string, string][] {
  if (hit.source === "trial")
    return hit.text
      .split(" | ")
      .flatMap((line) => splitSentences(line))
      .filter((s) => s.split(/\s+/).length >= 4)
      .map((s) => ["", s.trim()]);
  const out: [string, string][] = [];
  let label = "";
  for (let sent of splitSentences(hit.text)) {
    const m = sent.match(LABEL_RX);
    if (m) {
      label = m[1];
      sent = sent.slice(m[0].length);
    }
    out.push([label, sent.trim()]);
  }
  return out;
}

function scoreSentence(sent: string, label: string, q: ParsedQuery): number {
  const toks = new Set(tokenize(sent));
  if (!toks.size) return 0;
  const qs = new Set(q.tokens);
  let overlap = 0;
  for (const t of toks) if (qs.has(t)) overlap++;
  overlap /= Math.max(qs.size, 1);
  const prior = (PREFERRED[q.primary_intent] ?? ["CONCLUSIONS", "RESULTS"]).includes(label) ? 0.25 : 0;
  const numeric = /\d/.test(sent) && ["efficacy", "toxicity"].includes(q.primary_intent) ? 0.1 : 0;
  const lengthPen = sent.split(/\s+/).length > 70 ? 0.15 : 0;
  const bgPen = label === "BACKGROUND" || label === "METHODS" ? 0.2 : 0;
  return overlap + prior + numeric - lengthPen - bgPen;
}

export function compose(q: ParsedQuery, hits: EvidenceHit[]): string {
  if (!hits.length) return "BOTTOM LINE: The retrieved evidence does not answer this question.";
  let leadIdx = hits.findIndex((h) => h.source === "guideline");
  if (leadIdx < 0) leadIdx = 0;
  const leadSents = sentencesOf(hits[leadIdx])
    .sort((a, b) => scoreSentence(b[1], b[0], q) - scoreSentence(a[1], a[0], q))
    .slice(0, 2)
    .map(([, s]) => s);
  const bottom = leadSents.map((s) => `${trim(s)} [S${leadIdx + 1}].`).join(" ");
  const bullets: string[] = [];
  hits.forEach((h, i) => {
    if (i === leadIdx) return;
    if (h.section === "Eligibility criteria" && q.primary_intent !== "eligibility") return;
    const scored = sentencesOf(h).sort((a, b) => scoreSentence(b[1], b[0], q) - scoreSentence(a[1], a[0], q));
    if (!scored.length || scoreSentence(scored[0][1], scored[0][0], q) < MIN_SENTENCE_SCORE) return;
    bullets.push(`* ${trim(scored[0][1])} [S${i + 1}].`);
  });
  const sources = new Set(hits.map((h) => h.source));
  const caveats: string[] = [];
  if (!sources.has("guideline")) caveats.push("* No curated guideline digest matched this question closely, so the answer rests on individual trials.");
  if (sources.has("trial")) caveats.push("* Trial registry reports summarise posted results and have not necessarily been peer reviewed.");
  if (q.subtypes.length || q.settings.length) caveats.push("* Check that each cited population matches the patient's subtype and disease setting before applying a finding.");
  caveats.push("* Extractive mode: statements are verbatim evidence sentences selected by relevance; a local language model can produce a fuller synthesis.");
  return `BOTTOM LINE: ${bottom}\nKEY EVIDENCE:\n${bullets.slice(0, 6).join("\n")}\nCAVEATS:\n${caveats.join("\n")}`;
}

// Verifier
const CITE_RX = /\[S(\d+)\]/g;
const HEADER_RX = /^(BOTTOM LINE|KEY EVIDENCE|CAVEATS)\s*:?\s*/i;
const NON_CLAIM_RX = /^(the retrieved evidence does not answer|no (guideline|evidence)|this (is|was) |caveat)/i;

function claims(answer: string): [string, string][] {
  const out: [string, string][] = [];
  let section = "BOTTOM LINE";
  for (const rawLine of answer.split("\n")) {
    const h = rawLine.trim().match(HEADER_RX);
    if (h) section = h[1].toUpperCase();
    const line = rawLine.trim().replace(HEADER_RX, "").replace(/^[*•]+/, "").trim();
    if (!line) continue;
    for (let sent of splitSentences(line)) {
      const lead = sent.match(/^((?:\[S\d+\]\s*)+)/);
      if (lead && out.length) {
        out[out.length - 1][1] += " " + lead[1].trim();
        sent = sent.slice(lead[0].length).trim();
      }
      if (tokenize(sent.replace(CITE_RX, "")).length >= 3) out.push([section, sent]);
    }
  }
  return out;
}

export function verify(answer: string, evidence: string[]): Verification {
  const evTok = evidence.map((t) => new Set(tokenize(t)));
  const evNum = evidence.map((t) => numbers(t));
  const sentences: SentenceCheck[] = [];
  const invalid = new Set<number>();
  const unsupportedNums = new Set<string>();
  let citedAny = 0;
  for (const [section, sent] of claims(answer)) {
    const cites = Array.from(sent.matchAll(CITE_RX)).map((m) => Number(m[1]));
    const clean = sent.replace(CITE_RX, "");
    const toks = new Set(tokenize(clean));
    const valid = cites.filter((c) => c >= 1 && c <= evidence.length);
    cites.filter((c) => !valid.includes(c)).forEach((c) => invalid.add(c));
    let best = 0;
    let bestSrc: number | null = null;
    for (const c of valid) {
      let inter = 0;
      for (const t of toks) if (evTok[c - 1].has(t)) inter++;
      const s = inter / Math.max(toks.size, 1);
      if (s > best) {
        best = s;
        bestSrc = c;
      }
    }
    const sentNums = Array.from(numbers(clean)).filter((n) => !/^\d$/.test(n));
    const pool = new Set<string>();
    valid.forEach((c) => evNum[c - 1].forEach((n) => pool.add(n)));
    const missing = sentNums.filter((n) => !pool.has(n));
    if (missing.length && valid.length) missing.forEach((m) => unsupportedNums.add(m));
    const numsOk = missing.length === 0;
    let status: SentenceCheck["status"];
    if (NON_CLAIM_RX.test(clean.trim()) || (section === "CAVEATS" && !valid.length)) status = "meta";
    else if (!valid.length) status = "uncited";
    else if (best >= 0.5 && numsOk) status = "supported";
    else if (best >= 0.3 && numsOk) status = "partial";
    else status = "unsupported";
    if (valid.length && status !== "meta") citedAny++;
    sentences.push({ text: sent, citations: valid, support: Math.round(best * 1000) / 1000, numbers_ok: numsOk, status, best_source: bestSrc });
  }
  const claimsOnly = sentences.filter((s) => s.status !== "meta");
  const faith = claimsOnly.length
    ? claimsOnly.reduce((a, s) => a + (s.status === "supported" ? 1 : s.status === "partial" ? 0.5 : 0), 0) / claimsOnly.length
    : 0;
  return {
    sentences,
    faithfulness: Math.round(faith * 1000) / 1000,
    citation_coverage: claimsOnly.length ? Math.round((citedAny / claimsOnly.length) * 1000) / 1000 : 0,
    invalid_citations: Array.from(invalid).sort(),
    unsupported_numbers: Array.from(unsupportedNums).sort(),
  };
}

export function confidence(hits: EvidenceHit[], faithfulness: number) {
  const strong = hits.filter((h) => (h.features.query_coverage ?? 0) >= 0.5).length;
  const highLevel = hits.slice(0, 4).some((h) => h.evidence_rank <= 2);
  const score = 0.4 * Math.min(1, strong / 3) + 0.2 * (highLevel ? 1 : 0) + 0.4 * faithfulness;
  const level = score >= 0.75 ? "high" : score >= 0.5 ? "moderate" : "low";
  return { score: Math.round(score * 1000) / 1000, level: level as "high" | "moderate" | "low", strong_passages: strong, high_level_evidence: highLevel, faithfulness };
}
