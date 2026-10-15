import { useEffect, useState } from "react";
import { createEngine } from "./engine";
import type { Engine, EngineInfo, PatientProfile } from "./engine/types";
import { BrandMark, Icon } from "./components/Icon";
import { PRESETS } from "./lib/presets";
import { Overview } from "./views/Overview";
import { CaseWorkspace } from "./views/CaseWorkspace";
import { CopilotPanel, CopilotView } from "./views/Copilot";
import { TrialsView } from "./views/Trials";
import { CohortsView } from "./views/Cohorts";
import { LibraryView } from "./views/Library";
import { GovernanceView } from "./views/Governance";

const STATIC_ONLY = import.meta.env.VITE_STATIC_ONLY === "1";

type View = "overview" | "case" | "copilot" | "trials" | "cohorts" | "library" | "governance";

const NAV: [View, string, string][] = [
  ["overview", "Overview", "overview"],
  ["case", "Case workspace", "case"],
  ["copilot", "Evidence copilot", "copilot"],
  ["trials", "Trial screening", "trials"],
  ["cohorts", "Cohort explorer", "cohort"],
  ["library", "Knowledge library", "library"],
  ["governance", "Model governance", "governance"],
];

function readHash(): View {
  const h = window.location.hash.replace("#/", "") as View;
  return NAV.some(([v]) => v === h) ? h : "overview";
}

export default function App() {
  const [engine, setEngine] = useState<Engine | null>(null);
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [progress, setProgress] = useState("Starting Oncora Nexus");
  const [failure, setFailure] = useState<string | null>(null);
  const [view, setView] = useState<View>(readHash());
  const [patient, setPatient] = useState<PatientProfile>(PRESETS[0].patient);
  const [cardFocus, setCardFocus] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | undefined>();
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);

  useEffect(() => {
    createEngine(setProgress)
      .then(async (e) => {
        setEngine(e);
        setInfo(await e.info());
      })
      .catch((e) => setFailure(String(e.message ?? e)));
    const onHash = () => setView(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const go = (v: string) => {
    if (!STATIC_ONLY) window.location.hash = `#/${v}`;
    setView(v as View);
    window.scrollTo({ top: 0 });
  };

  const openCard = (id: string) => {
    setCardFocus(id);
    go("library");
  };

  const isDark = theme ? theme === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;

  if (failure) {
    return (
      <div className="loading">
        <BrandMark className="loading-mark" />
        <div>
          <strong>Oncora Nexus could not start</strong>
          <p>{failure}</p>
          <p className="sub">The demo needs the files in the demo folder next to this page. Reload once your connection is back, or run the service locally with docker compose up.</p>
        </div>
      </div>
    );
  }

  if (!engine || !info) {
    return (
      <div className="loading" role="status" aria-live="polite">
        <BrandMark className="loading-mark" />
        <div>
          <div style={{ fontFamily: "var(--prose)", fontSize: 24, fontWeight: 600, color: "var(--text)" }}>Oncora Nexus</div>
          <p style={{ margin: "6px 0 0" }}>{progress}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <BrandMark className="brand-mark" />
          <div className="brand-name">
            Oncora Nexus
            <span>Breast oncology evidence engine</span>
          </div>
        </div>
        <nav className="nav" aria-label="Main">
          {NAV.map(([v, label, icon]) => (
            <button key={v} aria-current={view === v ? "page" : undefined} onClick={() => go(v)}>
              <Icon name={icon} />
              {label}
            </button>
          ))}
        </nav>
        <div className="rail-foot">
          <div className="engine-status">
            <span className={`dot ${info.kind === "browser" ? "browser" : ""}`} />
            <div>
              <div style={{ color: "var(--text)", fontWeight: 600 }}>{info.label}</div>
              <div>{info.kind === "browser" ? "Retrieval and models run in this tab" : `${info.passages.toLocaleString()} passages indexed`}</div>
            </div>
          </div>
          <button className="theme-toggle" onClick={() => setTheme(isDark ? "light" : "dark")}>
            {isDark ? "Use light theme" : "Use dark theme"}
          </button>
        </div>
      </aside>
      <div>
        <nav className="mobile-nav" aria-label="Main">
          {NAV.map(([v, label]) => (
            <button key={v} aria-current={view === v ? "page" : undefined} onClick={() => go(v)}>
              {label}
            </button>
          ))}
        </nav>
        <main className="main" id="main">
          {view === "overview" && (
            <Overview
              info={info}
              go={go}
              loadPreset={(id) => {
                setPatient(PRESETS.find((p) => p.id === id)!.patient);
                go("case");
              }}
              ask={(q) => {
                setPendingQuestion(q);
                go("copilot");
              }}
            />
          )}
          {view === "case" && <CaseWorkspace engine={engine} patient={patient} setPatient={setPatient} openCard={openCard} />}
          {view === "copilot" &&
            (pendingQuestion ? (
              <>
                <header className="page-head">
                  <div>
                    <h1>Evidence copilot</h1>
                    <p>Answers drawn only from retrieved evidence, with every sentence checked against the passage it cites.</p>
                  </div>
                </header>
                <CopilotPanel engine={engine} initialQuestion={pendingQuestion} />
              </>
            ) : (
              <CopilotView engine={engine} />
            ))}
          {view === "trials" && <TrialsView engine={engine} patient={patient} setPatient={setPatient} />}
          {view === "cohorts" && <CohortsView engine={engine} />}
          {view === "library" && <LibraryView engine={engine} focus={cardFocus} />}
          {view === "governance" && <GovernanceView engine={engine} info={info} />}
          <p className="disclaimer">
            Oncora Nexus is a research and portfolio system for clinical decision support. It is not a medical device, has not been clinically validated for patient care, and does not replace the judgement of the multidisciplinary team. Teaching cases are fictional.
          </p>
        </main>
      </div>
    </div>
  );
}
