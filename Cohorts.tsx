import { useEffect, useState } from "react";
import type { Engine, KMCurve } from "../engine/types";
import { Columns, HBars, StepChart } from "../components/Charts";

const SUBTYPE_COLOR: Record<string, string> = {
  LumA: "var(--s1)",
  LumB: "var(--s2)",
  Her2: "var(--s3)",
  Basal: "var(--s4)",
  "claudin-low": "var(--s5)",
  Normal: "var(--s6)",
  "HR positive, HER2 negative": "var(--s1)",
  "HER2 positive": "var(--s3)",
  "Triple negative": "var(--s4)",
};
const NAME: Record<string, string> = { LumA: "Luminal A", LumB: "Luminal B", Her2: "HER2 enriched", Basal: "Basal like", "claudin-low": "Claudin low", Normal: "Normal like" };

type View = "metabric_rfs" | "metabric_os" | "metabric_receptor" | "tcga_os";

export function CohortsView({ engine }: { engine: Engine }) {
  const [data, setData] = useState<any>(null);
  const [view, setView] = useState<View>("metabric_rfs");
  const [table, setTable] = useState(false);
  useEffect(() => {
    engine.cohorts().then(setData);
  }, [engine]);
  if (!data) return <div className="skeleton" style={{ height: 300 }} />;
  const block =
    view === "metabric_rfs"
      ? data.metabric.rfs_by_subtype
      : view === "metabric_os"
        ? data.metabric.os_by_subtype
        : view === "metabric_receptor"
          ? data.metabric.rfs_by_receptor_group
          : data.tcga.os_by_subtype;
  const curves: [string, KMCurve][] = Object.entries(block.curves);
  const xMax = view === "tcga_os" ? 180 : 240;
  const p = block.logrank.p_value;
  const cohort = view === "tcga_os" ? data.tcga : data.metabric;
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Cohort explorer</h1>
          <p>Survival by intrinsic subtype and receptor group in METABRIC (n = {data.metabric.n.toLocaleString()}, median follow up {Math.round(data.metabric.median_follow_up_months / 12)} years) and TCGA BRCA (n = {data.tcga.n.toLocaleString()}).</p>
        </div>
      </header>
      <section className="panel panel-pad stack" style={{ marginBottom: 18 }}>
        <div className="spread" style={{ flexWrap: "wrap" }}>
          <div className="seg" role="group" aria-label="Cohort and endpoint">
            {(
              [
                ["metabric_rfs", "METABRIC relapse free"],
                ["metabric_os", "METABRIC overall"],
                ["metabric_receptor", "METABRIC by receptor group"],
                ["tcga_os", "TCGA overall"],
              ] as [View, string][]
            ).map(([v, l]) => (
              <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>
                {l}
              </button>
            ))}
          </div>
          <div className="row">
            <span className="sub">Log rank p {p < 1e-6 ? "below 0.000001" : `= ${p.toPrecision(2)}`}</span>
            <button className="btn ghost small" onClick={() => setTable(!table)}>
              {table ? "Show chart" : "Show table"}
            </button>
          </div>
        </div>
        {table ? (
          <table className="tableview">
            <thead>
              <tr>
                <th>Group</th>
                <th className="num">Patients</th>
                <th className="num">Events</th>
                <th className="num">5 years</th>
                <th className="num">10 years</th>
              </tr>
            </thead>
            <tbody>
              {curves.map(([name, c]) => {
                const at = (m: number) => c.survival[c.time.findIndex((t) => t >= m)] ?? c.survival[c.survival.length - 1];
                return (
                  <tr key={name}>
                    <td>{NAME[name] ?? name}</td>
                    <td className="num">{c.n}</td>
                    <td className="num">{c.events}</td>
                    <td className="num">{Math.round(at(60) * 100)}%</td>
                    <td className="num">{Math.round(at(120) * 100)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <StepChart
            ariaLabel="Kaplan Meier survival curves by group"
            series={curves.map(([name, c]) => ({ name: `${NAME[name] ?? name} (n=${c.n})`, color: SUBTYPE_COLOR[name] ?? "var(--text-3)", x: c.time, y: c.survival }))}
            xMax={xMax}
            height={330}
            xLabel="Months from diagnosis"
            yLabel={view.includes("rfs") || view === "metabric_receptor" ? "Relapse free survival" : "Overall survival"}
            xTicks={view === "tcga_os" ? [0, 36, 60, 120, 180] : [0, 60, 120, 180, 240]}
          />
        )}
      </section>
      <div className="grid cols-3">
        <section className="panel panel-pad">
          <h3>Intrinsic subtype mix</h3>
          <HBars
            items={Object.entries(cohort.subtype_counts as Record<string, number>)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => ({ label: NAME[k] ?? k, value: v, color: SUBTYPE_COLOR[k] }))}
            max={Math.max(...(Object.values(cohort.subtype_counts) as number[]))}
            format={(v) => v.toLocaleString()}
          />
        </section>
        <section className="panel panel-pad">
          <h3>Age at diagnosis</h3>
          <p className="sub">Median {cohort.median_age} years</p>
          <Columns edges={cohort.age_histogram.edges} counts={cohort.age_histogram.counts} label="Age histogram" />
        </section>
        <section className="panel panel-pad">
          {view === "tcga_os" ? (
            <>
              <h3>Pathological stage</h3>
              <HBars items={Object.entries(data.tcga.stage_counts as Record<string, number>).map(([k, v]) => ({ label: k, value: v }))} max={Math.max(...(Object.values(data.tcga.stage_counts) as number[]))} format={(v) => v.toLocaleString()} />
            </>
          ) : (
            <>
              <h3>Treatment received in METABRIC</h3>
              <HBars
                items={Object.entries(data.metabric.treatment_rates as Record<string, number>).map(([k, v]) => ({ label: k.replace("_", " ").replace(/^./, (c) => c.toUpperCase()), value: v }))}
              />
              <p className="sub" style={{ marginTop: 12 }}>
                Patients were diagnosed between 1977 and 2005, before routine HER2 targeted therapy, which is why survival models trained here are conservative for today's patients.
              </p>
            </>
          )}
        </section>
      </div>
    </>
  );
}
