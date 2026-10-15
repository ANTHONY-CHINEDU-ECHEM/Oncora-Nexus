import type { AskResponse, Card, Engine, EngineInfo, PathwayRule, PatientProfile, PrognosisResponse, Source, SubtypeResponse, TrialMatchResponse } from "./types";
import { initText, type Lexicon } from "./text";
import { initQuery, parseQuery } from "./query";
import { BrowserIndex } from "./retrieval";
import { compose, confidence, guard, redact, verify } from "./grounding";
import { BrowserTrialMatcher, coxPredict, evaluatePathway, summary, surrogateSubtype } from "./clinical";

// Server engine: talks to the FastAPI service.
class HttpEngine implements Engine {
  constructor(private base: string, private health: any) {}

  private async post<T>(path: string, body: unknown): Promise<T> {
    const r = await fetch(`${this.base}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${path} failed with status ${r.status}: ${await r.text()}`);
    return r.json();
  }

  private async get<T>(path: string): Promise<T> {
    const r = await fetch(`${this.base}${path}`);
    if (!r.ok) throw new Error(`${path} failed with status ${r.status}`);
    return r.json();
  }

  async info(): Promise<EngineInfo> {
    const meta = await this.get<any>("/api/meta");
    return {
      kind: "server",
      label: this.health.llm_available ? `Live service with ${this.health.llm_model}` : "Live service, extractive mode",
      llm: this.health.llm_available ? this.health.llm_model : null,
      passages: meta.index.passages,
      documents: meta.index.documents,
    };
  }
  ask(question: string, opts: { caseContext?: string; sources?: Source[]; mode?: string } = {}): Promise<AskResponse> {
    return this.post("/api/ask", { question, case_context: opts.caseContext, sources: opts.sources?.length ? opts.sources : null, mode: opts.mode ?? "auto" });
  }
  prognosis(patient: PatientProfile): Promise<PrognosisResponse> {
    return this.post("/api/prognosis", { patient });
  }
  subtype(patient: PatientProfile): Promise<SubtypeResponse> {
    return this.post("/api/subtype", { patient });
  }
  async pathway(patient: PatientProfile): Promise<PathwayRule[]> {
    return (await this.post<any>("/api/pathway", { patient })).recommendations;
  }
  trials(patient: PatientProfile, query?: string, includeExcluded = false): Promise<TrialMatchResponse> {
    return this.post("/api/trials/match", { patient, query, top_k: 30, include_excluded: includeExcluded });
  }
  cohorts() {
    return this.get("/api/cohorts");
  }
  evaluation() {
    return this.get("/api/evaluation");
  }
  async cards(): Promise<Card[]> {
    return (await this.get<any>("/api/cards")).cards;
  }
  document(docId: string) {
    return this.get<any>(`/api/documents/${encodeURIComponent(docId)}`);
  }
}

// Browser engine: everything runs locally from the exported demo bundle.
async function fetchJson(url: string): Promise<any> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (status ${r.status})`);
  if (url.endsWith(".gz")) {
    const buf = new Uint8Array(await r.arrayBuffer());
    const isGzip = buf[0] === 0x1f && buf[1] === 0x8b;
    if (!isGzip) return JSON.parse(new TextDecoder().decode(buf));
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return r.json();
}

class BrowserEngine implements Engine {
  private constructor(
    private index: BrowserIndex,
    private matcher: BrowserTrialMatcher,
    private bundle: Record<string, any>,
    private cardMap: Record<string, Card>,
  ) {}

  static async load(base: string, onProgress?: (msg: string) => void): Promise<BrowserEngine> {
    // The static host build serves plain JSON; other builds ship gzip to save bandwidth.
    const gz = import.meta.env.VITE_STATIC_ONLY === "1" ? "" : ".gz";
    onProgress?.("Loading knowledge lexicon and models");
    const [lexicon, reranker, cox, surrogate, rules, cards, cohorts, meta] = await Promise.all(
      ["lexicon.json", "reranker_lexical.json", "cox_core.json", "subtype_surrogate.json", "therapy_rules.json", "cards.json", "cohort_analytics.json", "meta.json"].map((f) =>
        fetchJson(`${base}/${f}`),
      ),
    );
    initText(lexicon as Lexicon);
    initQuery(lexicon as Lexicon);
    onProgress?.("Indexing 13,660 evidence passages in your browser");
    let corpus: any;
    if (gz) corpus = await fetchJson(`${base}/corpus.json.gz`);
    else {
      // Static hosts cap file size, so the plain JSON corpus is published in two parts.
      const parts = await Promise.all([1, 2].map((i) => fetchJson(`${base}/corpus_part${i}.json`)));
      corpus = { ...parts[0], passages: parts.flatMap((p) => p.passages) };
    }
    const index = new BrowserIndex(corpus, reranker);
    onProgress?.("Parsing 999 trial eligibility profiles");
    const trials = await fetchJson(`${base}/trials_parsed.json${gz}`);
    const reports = await fetchJson(`${base}/reports.json${gz}`);
    const cardMap: Record<string, Card> = Object.fromEntries(cards.cards.map((c: Card) => [c.id, c]));
    return new BrowserEngine(index, new BrowserTrialMatcher(trials), { cox, surrogate, rules, cards, cohorts, reports, meta }, cardMap);
  }

  async info(): Promise<EngineInfo> {
    return { kind: "browser", label: "Browser demo engine", llm: null, passages: this.index.passages.length, documents: this.index.documents };
  }

  async ask(question: string, opts: { caseContext?: string; sources?: Source[]; mode?: string } = {}): Promise<AskResponse> {
    const t0 = performance.now();
    const g = guard(question, opts.caseContext);
    const id = Math.random().toString(16).slice(2, 14);
    if (g.blocked) {
      return { request_id: id, question: g.text, mode: "blocked", answer: g.message!, evidence: [], verification: null, confidence: { level: "none", score: 0 }, redactions: g.counts, latency_ms: Math.round(performance.now() - t0) };
    }
    const ctx = opts.caseContext ? redact(opts.caseContext).text : "";
    const q = parseQuery(`${g.text} ${ctx}`.trim());
    const qq = parseQuery(g.text);
    q.intents = qq.intents.length ? qq.intents : q.intents;
    q.primary_intent = q.intents[0] ?? "general";
    const hits = this.index.search(q, 8, opts.sources);
    const answer = compose(q, hits);
    const ver = verify(answer, hits.map((h) => h.text));
    return {
      request_id: id,
      question: g.text,
      mode: "extractive",
      answer,
      query: q,
      evidence: hits,
      verification: ver,
      confidence: confidence(hits, ver.faithfulness),
      redactions: g.counts,
      model: "extractive composer (browser)",
      latency_ms: Math.round(performance.now() - t0),
    };
  }

  async prognosis(patient: PatientProfile): Promise<PrognosisResponse> {
    return coxPredict(this.bundle.cox, patient);
  }

  async subtype(patient: PatientProfile): Promise<SubtypeResponse> {
    return { descriptions: this.bundle.surrogate.descriptions, surrogate: surrogateSubtype(this.bundle.surrogate, patient) };
  }

  async pathway(patient: PatientProfile): Promise<PathwayRule[]> {
    return evaluatePathway(patient, this.bundle.rules, this.cardMap);
  }

  async trials(patient: PatientProfile, query?: string, includeExcluded = false): Promise<TrialMatchResponse> {
    const t0 = performance.now();
    const r = this.matcher.match(patient, query, 30, includeExcluded);
    r.latency_ms = Math.round(performance.now() - t0);
    return r;
  }

  async cohorts() {
    return this.bundle.cohorts;
  }

  async evaluation() {
    return { ...this.bundle.reports, browser_meta: this.bundle.meta };
  }

  async cards(): Promise<Card[]> {
    return Object.values(this.cardMap);
  }

  async document(docId: string) {
    const ps = this.index.documentPassages(docId);
    return { doc_id: docId, source: ps[0]?.source ?? "unknown", passages: ps.map((p) => ({ chunk_id: p.chunk_id, title: p.title, section: p.section, text: p.text, url: p.url })) };
  }
}

export async function createEngine(onProgress?: (msg: string) => void): Promise<Engine> {
  const params = new URLSearchParams(window.location.search);
  const forced = import.meta.env.VITE_STATIC_ONLY === "1" ? "browser" : params.get("engine");
  const apiBase = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
  if (forced !== "browser") {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
      const r = await fetch(`${apiBase}/api/health`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (r.ok && (r.headers.get("content-type") ?? "").includes("json")) {
        onProgress?.("Connected to the Oncora Nexus service");
        return new HttpEngine(apiBase, await r.json());
      }
    } catch {
      // fall through to the browser engine
    }
  }
  const base = (window as any).__ONCORA_DEMO_BASE__ ?? "./demo";
  return BrowserEngine.load(base, onProgress);
}

export { summary };
