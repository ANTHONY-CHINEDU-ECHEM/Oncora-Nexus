// Browser retrieval: BM25 over precomputed stemmed tokens, then the exported
// lexical reranker (logistic regression over the same features computed by
// oncora/export/demo_bundle.py), then a per document cap for diversity.

import type { EvidenceHit, ParsedQuery, Source } from "./types";
import { sectionPrior } from "./query";
import { tokenize } from "./text";

export interface Passage {
  chunk_id: string;
  doc_id: string;
  source: Source;
  title: string;
  url: string;
  section: string;
  text: string;
  evidence_rank: number;
  evidence_label: string;
  tokens: string[];
  tokenSet: Set<string>;
  line_start: number;
}

export interface RerankerSpec {
  features: string[];
  mean: number[];
  scale: number[];
  coef: number[];
  intercept: number;
}

const SOURCE: Record<string, Source> = { g: "guideline", p: "pubmed", t: "trial" };

export class BrowserIndex {
  passages: Passage[] = [];
  private postings = new Map<string, { doc: Int32Array; w: Float32Array }>();
  avgdl = 0;

  constructor(raw: { evidence_labels: string[]; passages: any[][] }, private reranker: RerankerSpec) {
    const k1 = 1.2;
    const b = 0.75;
    const tmp = new Map<string, { doc: number[]; tf: number[] }>();
    let total = 0;
    raw.passages.forEach((row, i) => {
      const tokens = (row[9] as string).split(" ").filter(Boolean);
      total += tokens.length;
      this.passages.push({
        chunk_id: row[0],
        doc_id: row[1],
        source: SOURCE[row[2]],
        title: row[3],
        url: row[4],
        section: row[5],
        text: row[6],
        evidence_rank: row[7],
        evidence_label: raw.evidence_labels[row[8]],
        tokens,
        tokenSet: new Set(tokens),
        line_start: row[10],
      });
      const counts = new Map<string, number>();
      for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
      for (const [t, c] of counts) {
        let p = tmp.get(t);
        if (!p) tmp.set(t, (p = { doc: [], tf: [] }));
        p.doc.push(i);
        p.tf.push(c);
      }
    });
    const n = this.passages.length;
    this.avgdl = total / Math.max(n, 1);
    for (const [term, p] of tmp) {
      const df = p.doc.length;
      const idf = Math.log1p((n - df + 0.5) / (df + 0.5));
      const w = new Float32Array(df);
      for (let j = 0; j < df; j++) {
        const dl = this.passages[p.doc[j]].tokens.length;
        const tf = p.tf[j];
        w[j] = ((tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / this.avgdl))) * idf;
      }
      this.postings.set(term, { doc: Int32Array.from(p.doc), w });
    }
  }

  get documents(): number {
    return new Set(this.passages.map((p) => p.doc_id)).size;
  }

  bm25(tokens: string[], sources?: Source[]): Float32Array {
    const scores = new Float32Array(this.passages.length);
    const counts = new Map<string, number>();
    for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
    for (const [t, qtf] of counts) {
      const p = this.postings.get(t);
      if (!p) continue;
      for (let j = 0; j < p.doc.length; j++) scores[p.doc[j]] += p.w[j] * qtf;
    }
    if (sources?.length) {
      const allowed = new Set(sources);
      this.passages.forEach((p, i) => {
        if (!allowed.has(p.source)) scores[i] = 0;
      });
    }
    return scores;
  }

  private features(q: ParsedQuery, cand: number[], bm: Float32Array): number[][] {
    const qset = new Set(q.tokens);
    const qbi: string[][] = [];
    for (let i = 0; i + 1 < q.tokens.length; i++) qbi.push([q.tokens[i], q.tokens[i + 1]]);
    const uniqueBi = Array.from(new Map(qbi.map((b) => [b.join(" "), b])).values());
    const drugs = q.drugs.map((d) => tokenize(d));
    const ctx = [...q.subtypes, ...q.settings, ...q.populations].map((c) => tokenize(c).slice(0, 1)).filter((c) => c.length);
    const bmMax = Math.max(bm[cand[0]] ?? 1, 1e-9);
    return cand.map((i, rank) => {
      const p = this.passages[i];
      const toks = p.tokenSet;
      let cov = 0;
      for (const t of qset) if (toks.has(t)) cov++;
      return [
        bm[i] / bmMax,
        1 / (rank + 1),
        cov / Math.max(qset.size, 1),
        uniqueBi.length ? uniqueBi.filter((b) => b.every((t) => toks.has(t))).length / uniqueBi.length : 0,
        drugs.length ? drugs.filter((d) => d.length && d.every((t) => toks.has(t))).length / drugs.length : 0,
        ctx.length ? ctx.filter((c) => toks.has(c[0])).length / ctx.length : 0,
        sectionPrior(q, p.section),
        p.source === "guideline" ? 1 : 0,
        p.source === "trial" ? 1 : 0,
        (5 - p.evidence_rank) / 4,
      ];
    });
  }

  search(q: ParsedQuery, k = 8, sources?: Source[], perDoc = 2): EvidenceHit[] {
    const bm = this.bm25(q.tokens, sources);
    const order = Array.from(bm.keys())
      .filter((i) => bm[i] > 0)
      .sort((a, b) => bm[b] - bm[a])
      .slice(0, 60);
    if (!order.length) return [];
    const X = this.features(q, order, bm);
    const r = this.reranker;
    const scored = order.map((idx, j) => {
      let s = r.intercept;
      for (let f = 0; f < r.coef.length; f++) s += r.coef[f] * ((X[j][f] - r.mean[f]) / r.scale[f]);
      return { idx, s, x: X[j] };
    });
    scored.sort((a, b) => b.s - a.s);
    const perDocCount = new Map<string, number>();
    const hits: EvidenceHit[] = [];
    for (const c of scored) {
      const p = this.passages[c.idx];
      const used = perDocCount.get(p.doc_id) ?? 0;
      if (used >= perDoc) continue;
      perDocCount.set(p.doc_id, used + 1);
      hits.push({
        chunk_id: p.chunk_id,
        doc_id: p.doc_id,
        source: p.source,
        title: p.title,
        url: p.url,
        section: p.section,
        evidence_label: p.evidence_label,
        evidence_rank: p.evidence_rank,
        score: Math.round(c.s * 1e4) / 1e4,
        rank: hits.length + 1,
        text: p.text,
        features: Object.fromEntries(r.features.map((f, i) => [f, Math.round(c.x[i] * 1e4) / 1e4])),
        meta: p.source === "pubmed" ? { pmid: p.doc_id.replace("PMID:", "") } : p.source === "trial" ? { nct_id: p.doc_id } : {},
        line_start: p.line_start,
      });
      if (hits.length >= k) break;
    }
    return hits;
  }

  documentPassages(docId: string): Passage[] {
    return this.passages.filter((p) => p.doc_id === docId);
  }
}
