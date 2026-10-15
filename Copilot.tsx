import { useEffect, useRef, useState } from "react";
import type { AskResponse, Engine, Source } from "../engine/types";
import { ConfidenceBadge, EvidenceCard, VerificationSummary, VerifiedAnswer } from "../components/Evidence";
import { Icon } from "../components/Icon";

const SUGGESTED = [
  "Is one week of breast radiotherapy as good as three weeks?",
  "Residual invasive HER2 positive disease after neoadjuvant therapy: what adjuvant treatment?",
  "Which node positive luminal patients should get abemaciclib after surgery?",
  "How common is neutropenia with docetaxel in metastatic breast cancer?",
  "Does exercise reduce fatigue during chemotherapy?",
  "Can radiotherapy be omitted for an older woman with a small ER positive tumour?",
];

const SOURCES: [Source, string][] = [
  ["guideline", "Guideline digests"],
  ["pubmed", "PubMed trials"],
  ["trial", "Trial registry"],
];

export function CopilotPanel({ engine, caseContext, compact = false, initialQuestion }: { engine: Engine; caseContext?: string; compact?: boolean; initialQuestion?: string }) {
  const [q, setQ] = useState(initialQuestion ?? "");
  const [res, setRes] = useState<AskResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [useCase, setUseCase] = useState(true);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const run = async (question: string) => {
    if (!question.trim()) return;
    setQ(question);
    setBusy(true);
    setErr(null);
    try {
      setRes(await engine.ask(question, { caseContext: useCase ? caseContext : undefined, sources }));
    } catch (e: any) {
      setErr(`The question could not be answered: ${e.message}. Check that the service is running, then ask again.`);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (initialQuestion) run(initialQuestion);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuestion]);

  const toggleSource = (s: Source) => setSources((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="panel">
        <form
          className="ask-box"
          onSubmit={(e) => {
            e.preventDefault();
            run(q);
          }}
        >
          <label htmlFor="question" className="sr-only">
            Clinical question
          </label>
          <textarea
            id="question"
            ref={taRef}
            rows={2}
            value={q}
            placeholder="Ask a clinical question, for example: does adding pembrolizumab before surgery help stage II triple negative disease?"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                run(q);
              }
            }}
          />
          <button className="btn" type="submit" disabled={busy || !q.trim()}>
            <Icon name="send" className="" />
            {busy ? "Searching evidence" : "Ask"}
          </button>
        </form>
        <div className="spread" style={{ padding: "0 18px 14px", flexWrap: "wrap" }}>
          <div className="row" style={{ gap: 6 }}>
            <span className="sub">Search in</span>
            {SOURCES.map(([s, l]) => (
              <button key={s} type="button" className={`chip ${sources.includes(s) ? "haem" : ""}`} style={{ cursor: "pointer" }} aria-pressed={sources.includes(s)} onClick={() => toggleSource(s)}>
                {l}
              </button>
            ))}
            {!sources.length && <span className="sub">all sources</span>}
          </div>
          {caseContext && (
            <label className="check">
              <input type="checkbox" checked={useCase} onChange={(e) => setUseCase(e.target.checked)} /> Use the current patient as context
            </label>
          )}
        </div>
      </div>

      {!res && !busy && !compact && (
        <div className="stack" style={{ gap: 10 }}>
          <span className="sub">Try one of these</span>
          <div className="suggestions">
            {SUGGESTED.map((s) => (
              <button key={s} onClick={() => run(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {err && (
        <div className="banner warn" role="alert">
          <Icon name="alert" />
          {err}
        </div>
      )}

      {busy && (
        <div className="panel panel-pad stack" aria-busy="true">
          <div className="skeleton" style={{ width: "40%" }} />
          <div className="skeleton" />
          <div className="skeleton" style={{ width: "92%" }} />
          <div className="skeleton" style={{ width: "75%" }} />
        </div>
      )}

      {res && !busy && (
        <div className={compact ? "stack" : "copilot"}>
          <div className="panel panel-pad stack" style={{ gap: 16 }}>
            <div className="spread" style={{ flexWrap: "wrap" }}>
              <div className="row" style={{ gap: 8 }}>
                <ConfidenceBadge res={res} />
                <span className="chip">{res.mode === "llm" ? `Synthesised by ${res.model}` : res.mode === "blocked" ? "Safety guard" : "Extractive evidence brief"}</span>
                {Object.keys(res.redactions ?? {}).length > 0 && <span className="chip warn">Identifiers removed: {Object.entries(res.redactions).map(([k, v]) => `${v} ${k.toLowerCase().replace("_", " ")}`).join(", ")}</span>}
              </div>
              <span className="sub">{res.latency_ms} ms</span>
            </div>
            {res.notice && (
              <div className="banner">
                <Icon name="info" />
                {res.notice}
              </div>
            )}
            {res.mode === "blocked" ? (
              <div className="banner warn">
                <Icon name="shield" />
                {res.answer}
              </div>
            ) : (
              <>
                <VerifiedAnswer res={res} active={active} onCite={setActive} />
                <VerificationSummary res={res} />
                {res.query && (
                  <div className="row" style={{ gap: 6 }}>
                    <span className="sub">Understood as</span>
                    <span className="chip haem">{res.query.primary_intent}</span>
                    {[...res.query.drugs, ...res.query.subtypes, ...res.query.settings, ...res.query.populations].map((t) => (
                      <span key={t} className="chip">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
          {res.evidence.length > 0 && (
            <aside className="evidence-col" aria-label="Cited evidence">
              <div className="spread">
                <h3 style={{ margin: 0 }}>Evidence ({res.evidence.length})</h3>
                <span className="sub">Hover a citation to trace it</span>
              </div>
              {res.evidence.map((h, i) => (
                <EvidenceCard key={h.chunk_id} hit={h} n={i + 1} active={active === i + 1} onHover={setActive} />
              ))}
            </aside>
          )}
        </div>
      )}
    </div>
  );
}

export function CopilotView({ engine }: { engine: Engine }) {
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Evidence copilot</h1>
          <p>Answers drawn only from retrieved guideline digests, randomised trials and trial registry results. Every sentence is checked against the passage it cites.</p>
        </div>
        <div className="legend" aria-label="Underline key">
          <span>
            <i style={{ background: "var(--ok)" }} />
            Supported
          </span>
          <span>
            <i style={{ background: "var(--warn)" }} />
            Partly supported
          </span>
          <span>
            <i style={{ background: "var(--bad)" }} />
            Check before use
          </span>
        </div>
      </header>
      <CopilotPanel engine={engine} />
    </>
  );
}
