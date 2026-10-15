import { useEffect, useMemo, useState } from "react";
import type { Engine, PathwayRule, PatientProfile, PrognosisResponse, SubtypeResponse } from "../engine/types";
import { PatientForm } from "../components/PatientForm";
import { Contributions, HBars, StepChart } from "../components/Charts";
import { Icon } from "../components/Icon";
import { PRESETS } from "../lib/presets";
import { summary } from "../engine";
import { TrialResults } from "./Trials";
import { CopilotPanel } from "./Copilot";

type Tab = "brief" | "prognosis" | "subtype" | "trials" | "evidence";

const RISK_COLOR: Record<string, string> = { low: "var(--risk-low)", intermediate: "var(--risk-mid)", high: "var(--risk-high)" };

function Pathway({ rules, onOpenCard }: { rules: PathwayRule[]; onOpenCard: (id: string) => void }) {
  const groups = useMemo(() => {
    const g = new Map<string, PathwayRule[]>();
    rules.forEach((r) => g.set(r.category, [...(g.get(r.category) ?? []), r]));
    return Array.from(g.entries());
  }, [rules]);
  if (!rules.length) {
    return (
      <div className="empty">
        <strong>No pathway options yet</strong>
        Enter the disease setting and receptor status, or load a teaching case.
      </div>
    );
  }
  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      {groups.map(([cat, rs]) => (
        <section key={cat} className="panel panel-pad">
          <h3>{cat}</h3>
          {rs.map((r) => (
            <div key={r.id} className={`rule ${r.strength}`}>
              <div className="bar" />
              <div>
                <div className="spread" style={{ alignItems: "flex-start" }}>
                  <span className="title">{r.title}</span>
                  <span className={`chip ${r.strength === "standard" ? "haem" : r.strength === "caution" ? "warn" : ""}`}>
                    {r.strength === "standard" ? "Standard option" : r.strength === "consider" ? "Consider" : "Caution"}
                  </span>
                </div>
                <p>{r.text}</p>
                <div className="row" style={{ gap: 6 }}>
                  {r.cards.map((c) => (
                    <button key={c} className="chip" style={{ cursor: "pointer" }} onClick={() => onOpenCard(c)} title="Open the evidence card">
                      <Icon name="doc" />
                      {r.card_titles?.[c] ?? c}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

function Prognosis({ res, patient }: { res: PrognosisResponse | null; patient: PatientProfile }) {
  if (patient.setting === "metastatic")
    return (
      <div className="panel empty">
        <strong>Prognostic model applies to early disease</strong>
        The relapse free survival model was trained on patients with operable breast cancer, so it is not shown for metastatic disease.
      </div>
    );
  if (!res) return <div className="skeleton" style={{ height: 240 }} />;
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  return (
    <div className="stack">
      <div className="grid cols-4">
        {(["36m", "60m", "120m"] as const).map((h) => (
          <div key={h} className="panel metric">
            <div className="value">{pct(res.survival_at[h])}</div>
            <div className="label">Relapse free at {Number(h.replace("m", "")) / 12} years</div>
          </div>
        ))}
        <div className="panel metric">
          <div className="value" style={{ color: RISK_COLOR[res.risk_group] }}>
            {res.risk_group[0].toUpperCase() + res.risk_group.slice(1)}
          </div>
          <div className="label">Risk group, relative hazard {res.relative_hazard.toFixed(2)}</div>
          {res.npi && (
            <div className="note">
              NPI {res.npi.value} ({res.npi.group} prognostic group)
            </div>
          )}
        </div>
      </div>
      <div className="grid cols-2" style={{ alignItems: "start" }}>
        <section className="panel panel-pad">
          <h3>Predicted relapse free survival</h3>
          <StepChart
            ariaLabel="Predicted relapse free survival curve for this patient against the cohort baseline"
            series={[
              { name: "This patient", color: "var(--haem)", x: res.curve.time, y: res.curve.survival, step: false },
              { name: "Cohort baseline", color: "var(--text-3)", x: res.curve.time, y: res.curve.baseline, step: false, dashed: true },
            ]}
            xMax={180}
            yMin={0}
            xLabel="Months from diagnosis"
            yLabel="Relapse free"
            xTicks={[0, 36, 60, 120, 180]}
          />
        </section>
        <section className="panel panel-pad">
          <h3>What drives this estimate</h3>
          <p className="sub" style={{ marginBottom: 14 }}>
            Each bar is the feature's contribution to the log hazard relative to the average METABRIC patient.
          </p>
          <Contributions items={res.contributions.map((c) => ({ label: c.label, value: c.log_hazard, note: c.imputed ? "(median used)" : undefined }))} />
        </section>
      </div>
      <div className="banner">
        <Icon name="info" />
        <div>
          {res.caveats.map((c) => (
            <p key={c} style={{ margin: "0 0 4px" }}>
              {c}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

function Subtype({ res }: { res: SubtypeResponse | null }) {
  if (!res?.surrogate) return <div className="skeleton" style={{ height: 200 }} />;
  const s = res.surrogate;
  const items = Object.entries(s.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label: label === "claudin-low" ? "Claudin low" : label === "LumA" ? "Luminal A" : label === "LumB" ? "Luminal B" : label === "Her2" ? "HER2 enriched" : label === "Basal" ? "Basal like" : "Normal like", value }));
  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <section className="panel panel-pad">
        <h3>Likely intrinsic subtype from routine pathology</h3>
        <p className="sub" style={{ marginBottom: 14 }}>
          {s.description}
        </p>
        <HBars items={items} highlight={items[0].label} />
      </section>
      <section className="panel panel-pad stack">
        <h3>How far to trust this</h3>
        <p className="sub">{s.model_note}</p>
        <p className="sub">
          Molecular subtype is defined by gene expression (PAM50). Immunohistochemistry and grade recover it imperfectly, most often confusing Luminal A with Luminal B and Normal like tumours. When the top two probabilities are close, a genomic assay is the most likely to change the label.
        </p>
        <div className="banner">
          <Icon name="info" />
          The service also accepts PAM50 expression values through the API, where the expression model reaches about 77 percent accuracy against the reference label.
        </div>
      </section>
    </div>
  );
}

export function CaseWorkspace({ engine, patient, setPatient, openCard }: { engine: Engine; patient: PatientProfile; setPatient: (p: PatientProfile) => void; openCard: (id: string) => void }) {
  const [tab, setTab] = useState<Tab>("brief");
  const [rules, setRules] = useState<PathwayRule[]>([]);
  const [prog, setProg] = useState<PrognosisResponse | null>(null);
  const [sub, setSub] = useState<SubtypeResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const t = setTimeout(async () => {
      try {
        const [r, p, s] = await Promise.all([
          engine.pathway(patient),
          patient.setting === "metastatic" ? Promise.resolve(null) : engine.prognosis(patient),
          engine.subtype(patient),
        ]);
        if (!live) return;
        setRules(r);
        setProg(p);
        setSub(s);
        setErr(null);
      } catch (e: any) {
        if (live) setErr(e.message);
      }
    }, 150);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [engine, patient]);

  const text = summary(patient);
  const tabs: [Tab, string, number | null][] = [
    ["brief", "Tumour board brief", rules.length],
    ["prognosis", "Prognosis", null],
    ["subtype", "Subtype", null],
    ["trials", "Trials", null],
    ["evidence", "Ask about this patient", null],
  ];

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Case workspace</h1>
          <p>{text}</p>
        </div>
        {import.meta.env.VITE_STATIC_ONLY !== "1" && (
          <button className="btn ghost" onClick={() => window.print()}>
            <Icon name="doc" className="" /> Print brief
          </button>
        )}
      </header>
      <div className="workspace">
        <PatientForm patient={patient} onChange={setPatient} onPreset={(id) => setPatient(PRESETS.find((p) => p.id === id)!.patient)} />
        <div style={{ minWidth: 0 }}>
          <div className="tabs" role="tablist">
            {tabs.map(([id, label, count]) => (
              <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>
                {label}
                {count ? <span className="count">{count}</span> : null}
              </button>
            ))}
          </div>
          {err && <div className="banner warn">{err}</div>}
          {tab === "brief" && (
            <div className="stack">
              <div className="banner">
                <Icon name="shield" />
                Options for multidisciplinary discussion, derived from curated guideline digests. They are not treatment decisions; open each linked card to see the evidence behind it.
              </div>
              <Pathway rules={rules} onOpenCard={openCard} />
            </div>
          )}
          {tab === "prognosis" && <Prognosis res={prog} patient={patient} />}
          {tab === "subtype" && <Subtype res={sub} />}
          {tab === "trials" && <TrialResults engine={engine} patient={patient} />}
          {tab === "evidence" && <CopilotPanel engine={engine} caseContext={text} compact />}
        </div>
      </div>
    </>
  );
}
