import type { EngineInfo } from "../engine/types";
import { PRESETS } from "../lib/presets";
import { Icon } from "../components/Icon";

export function Overview({ info, go, loadPreset, ask }: { info: EngineInfo; go: (v: string) => void; loadPreset: (id: string) => void; ask: (q: string) => void }) {
  return (
    <>
      <section className="hero" style={{ marginBottom: 22 }}>
        <div className="panel panel-pad" style={{ padding: "30px 32px" }}>
          <h1 className="hero-title">Every recommendation, traced to its evidence.</h1>
          <p style={{ color: "var(--text-2)", maxWidth: "60ch", margin: "0 0 20px", fontSize: 16 }}>
            Oncora Nexus prepares a breast cancer case for the multidisciplinary team: pathway options from curated guidance, relapse risk from validated models, eligible trials screened criterion by criterion, and answers that cite {info.passages.toLocaleString()} passages from {info.documents.toLocaleString()} trials, abstracts and guideline digests.
          </p>
          <form
            className="panel"
            style={{ boxShadow: "none" }}
            onSubmit={(e) => {
              e.preventDefault();
              const q = new FormData(e.currentTarget).get("q") as string;
              if (q?.trim()) ask(q);
            }}
          >
            <div className="ask-box">
              <label htmlFor="hero-q" className="sr-only">
                Ask a clinical question
              </label>
              <textarea id="hero-q" name="q" rows={1} placeholder="Ask a clinical question about breast cancer" style={{ minHeight: 44 }} />
              <button className="btn" type="submit">
                Ask
              </button>
            </div>
          </form>
          <div className="row" style={{ marginTop: 16, gap: 8 }}>
            <span className={`chip ${info.kind === "server" ? "ok" : "warn"}`}>{info.label}</span>
            <span className="chip">Runs fully on local infrastructure</span>
            <span className="chip">No patient data leaves the session</span>
          </div>
        </div>
        <div className="panel panel-pad">
          <h2>Start from a teaching case</h2>
          <p className="sub" style={{ marginBottom: 14 }}>
            Fictional profiles that exercise different pathways. Each opens in the case workspace, where every field can be edited.
          </p>
          <div className="case-pick">
            {PRESETS.map((p) => (
              <button key={p.id} onClick={() => loadPreset(p.id)}>
                <strong>{p.name}</strong>
                <span>{p.blurb}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="grid cols-3" style={{ alignItems: "start" }}>
        <div className="panel panel-pad">
          <h2>How a case is prepared</h2>
          <div className="flow" style={{ marginTop: 8 }}>
            {[
              ["Profile", "Structured, deidentified summary; identifiers in free text are removed before anything is stored."],
              ["Pathway", "Declarative rules distilled from 48 guideline digests propose options with their evidence cards."],
              ["Prognosis and subtype", "Cox model trained on METABRIC, externally validated on GBSG2; subtype from routine pathology."],
              ["Trials", "7,000 parsed eligibility criteria across 999 breast cancer trials, each verdict tied to its protocol line."],
              ["Evidence", "Hybrid retrieval, learned reranking and a local language model; every sentence verified against its citation."],
            ].map(([t, d], i) => (
              <div key={t} className="flow-step">
                <span className="n">{i + 1}</span>
                <div>
                  <strong style={{ fontSize: 14 }}>{t}</strong>
                  <div className="sub">{d}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="panel panel-pad stack">
          <h2>Validated, and honest about limits</h2>
          <div className="grid cols-2" style={{ gap: 12 }}>
            <div>
              <div className="metric" style={{ padding: 0 }}>
                <div className="value">0.95</div>
                <div className="label">Recall at 5 on blind clinical questions</div>
              </div>
            </div>
            <div>
              <div className="metric" style={{ padding: 0 }}>
                <div className="value">0.67</div>
                <div className="label">External C index on GBSG2 (NPI 0.66)</div>
              </div>
            </div>
            <div>
              <div className="metric" style={{ padding: 0 }}>
                <div className="value">0</div>
                <div className="label">Invented numbers across 70 test answers</div>
              </div>
            </div>
            <div>
              <div className="metric" style={{ padding: 0 }}>
                <div className="value">5 of 5</div>
                <div className="label">Safety probes passed</div>
              </div>
            </div>
          </div>
          <button className="btn ghost small" style={{ alignSelf: "flex-start" }} onClick={() => go("governance")}>
            <Icon name="governance" className="" /> Review the full evaluation
          </button>
        </div>
        <div className="panel panel-pad stack">
          <h2>What it is not</h2>
          <p className="sub">
            Oncora Nexus supports qualified clinicians. It does not diagnose, it does not choose treatment, and its survival estimates come from historical cohorts. Absolute risks should be reconciled with PREDICT Breast, and every trial verdict needs confirmation by the research team.
          </p>
          <p className="sub">
            The trial corpus here contains studies with posted results, which demonstrate the screening engine. A deployment connects the live ClinicalTrials.gov feed of recruiting studies.
          </p>
          <button className="btn ghost small" style={{ alignSelf: "flex-start" }} onClick={() => go("library")}>
            Browse the knowledge cards
          </button>
        </div>
      </section>
    </>
  );
}
