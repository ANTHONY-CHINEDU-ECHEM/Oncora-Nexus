import { useEffect, useState } from "react";
import type { Engine, PatientProfile, TrialMatch, TrialMatchResponse } from "../engine/types";
import { Icon } from "../components/Icon";
import { PatientForm } from "../components/PatientForm";
import { PRESETS } from "../lib/presets";

const TIER_CHIP: Record<string, string> = { "likely eligible": "ok", "needs review": "warn", excluded: "bad" };
const CTYPE_LABEL: Record<string, string> = {
  age: "Age",
  ecog: "Performance status",
  her2: "HER2",
  hormone: "Hormone receptor",
  setting: "Disease setting",
  menopause: "Menopausal status",
  brca: "BRCA",
  brain_mets: "Brain metastases",
  pregnancy: "Pregnancy",
  lvef: "Cardiac function",
  prior_therapy: "Prior therapy",
};

function VerdictIcon({ v }: { v: string }) {
  const name = v === "met" ? "check" : v === "not_met" ? "cross" : "question";
  const color = v === "met" ? "var(--ok)" : v === "not_met" ? "var(--bad)" : "var(--warn)";
  return (
    <span style={{ color }}>
      <Icon name={name} className="icon" title={v === "met" ? "Met" : v === "not_met" ? "Not met" : "Unknown"} />
    </span>
  );
}

export function TrialCard({ t, defaultOpen = false }: { t: TrialMatch; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <article className="trial">
      <button className="trial-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div className="stack" style={{ gap: 6 }}>
          <div className="row" style={{ gap: 8 }}>
            <span className={`chip ${TIER_CHIP[t.tier]}`}>{t.tier[0].toUpperCase() + t.tier.slice(1)}</span>
            <span className="sub">{t.nct_id}</span>
            <span className="sub">{t.setting}</span>
          </div>
          <span className="title">{t.title}</span>
          {t.arms.length > 0 && <span className="sub">Arms: {t.arms.slice(0, 3).join("; ")}</span>}
        </div>
        <div className="stack" style={{ gap: 4, alignItems: "flex-end", fontSize: 13 }}>
          <span>
            <strong style={{ color: "var(--ok)" }}>{t.criteria_met}</strong> met
          </span>
          {t.criteria_unknown > 0 && (
            <span>
              <strong style={{ color: "var(--warn)" }}>{t.criteria_unknown}</strong> unknown
            </span>
          )}
          {t.criteria_not_met > 0 && (
            <span>
              <strong style={{ color: "var(--bad)" }}>{t.criteria_not_met}</strong> not met
            </span>
          )}
          <span className="muted">{t.manual_review_items} for manual review</span>
        </div>
      </button>
      {open && (
        <div>
          {t.criteria.map((c, i) => (
            <div key={i} className="crit">
              <VerdictIcon v={c.verdict} />
              <div>
                <div>
                  <strong>{CTYPE_LABEL[c.ctype] ?? c.ctype}</strong> <span className="muted">({c.kind})</span>: {c.explanation}
                </div>
                <div className="src">{c.line >= 0 ? `Line ${c.line + 1}: ${c.text}` : c.text}</div>
              </div>
            </div>
          ))}
          <div className="crit" style={{ gridTemplateColumns: "1fr" }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="sub">Laboratory values, consent and investigator judgement items are left for the research team.</span>
              <a className="linklike" href={t.url} target="_blank" rel="noreferrer">
                Open on ClinicalTrials.gov <Icon name="external" />
              </a>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}

export function TrialResults({ engine, patient, query }: { engine: Engine; patient: PatientProfile; query?: string }) {
  const [res, setRes] = useState<TrialMatchResponse | null>(null);
  const [showExcluded, setShowExcluded] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    engine
      .trials(patient, query, showExcluded)
      .then((r) => live && (setRes(r), setErr(null)))
      .catch((e) => live && setErr(e.message));
    return () => {
      live = false;
    };
  }, [engine, patient, query, showExcluded]);
  if (err) return <div className="banner warn">{err}</div>;
  if (!res) return <div className="skeleton" style={{ height: 120 }} />;
  return (
    <div className="stack">
      <div className="grid cols-3">
        {(["likely eligible", "needs review", "excluded"] as const).map((t) => (
          <div key={t} className="panel metric">
            <div className="value" style={{ color: t === "likely eligible" ? "var(--ok)" : t === "needs review" ? "var(--warn)" : "var(--text-3)" }}>
              {res.tiers[t] ?? 0}
            </div>
            <div className="label">{t[0].toUpperCase() + t.slice(1)}</div>
          </div>
        ))}
      </div>
      <div className="spread">
        <span className="sub">{res.data_note}</span>
        <label className="check">
          <input type="checkbox" checked={showExcluded} onChange={(e) => setShowExcluded(e.target.checked)} /> Show excluded trials
        </label>
      </div>
      {res.trials.length === 0 ? (
        <div className="panel empty">
          <strong>No trials to show yet</strong>
          Add receptor status, disease setting and age to screen the trial corpus.
        </div>
      ) : (
        res.trials.map((t, i) => <TrialCard key={t.nct_id} t={t} defaultOpen={i === 0} />)
      )}
    </div>
  );
}

export function TrialsView({ engine, patient, setPatient }: { engine: Engine; patient: PatientProfile; setPatient: (p: PatientProfile) => void }) {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query), 300);
    return () => clearTimeout(t);
  }, [query]);
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Trial screening</h1>
          <p>Each eligibility criterion is parsed into a typed rule and checked against the patient. Every verdict links back to the line of the protocol it came from.</p>
        </div>
      </header>
      <div className="workspace">
        <PatientForm patient={patient} onChange={setPatient} onPreset={(id) => setPatient(PRESETS.find((p) => p.id === id)!.patient)} />
        <div className="stack">
          <div className="field">
            <label htmlFor="tq">Agent or keyword of interest (optional)</label>
            <input id="tq" className="input" value={query} placeholder="for example: palbociclib, trastuzumab deruxtecan, exercise" onChange={(e) => setQuery(e.target.value)} />
          </div>
          <TrialResults engine={engine} patient={patient} query={debounced} />
        </div>
      </div>
    </>
  );
}
