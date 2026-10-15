import { useEffect, useState } from "react";
import type { Engine, EngineInfo } from "../engine/types";
import { StepChart } from "../components/Charts";
import { Icon } from "../components/Icon";

const VARIANT: Record<string, string> = {
  bm25: "BM25",
  dense_lsa: "Dense (LSA)",
  hybrid_rrf: "Hybrid fusion",
  hybrid_rrf_reranked: "Hybrid fusion with learned reranker",
  bm25_lexical_reranker: "BM25 with lexical reranker",
};

function MetricTable({ rows, best }: { rows: [string, any][]; best?: string }) {
  return (
    <table className="tableview">
      <thead>
        <tr>
          <th>Retriever</th>
          <th className="num">Recall at 1</th>
          <th className="num">Recall at 5</th>
          <th className="num">MRR at 10</th>
          <th className="num">nDCG at 10</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([k, m]) => (
          <tr key={k} className={k === best ? "best" : ""}>
            <td>{VARIANT[k] ?? k}</td>
            <td className="num">{m.recall_at_1.toFixed(2)}</td>
            <td className="num">{m.recall_at_5.toFixed(2)}</td>
            <td className="num">{m.mrr_at_10.toFixed(3)}</td>
            <td className="num">{m.ndcg_at_10.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function GovernanceView({ engine, info }: { engine: Engine; info: EngineInfo }) {
  const [r, setR] = useState<any>(null);
  useEffect(() => {
    engine.evaluation().then(setR);
  }, [engine]);
  if (!r) return <div className="skeleton" style={{ height: 300 }} />;
  const ret = r.retrieval_eval;
  const surv = r.survival_eval;
  const sub = r.subtype_eval;
  const nli = r.nli4ct_eval;
  const ans = r.answer_eval_default?.summary;
  const browser = r.browser_engine_eval;
  const calib = (rows: any[]) => ({
    x: rows.map((d: any) => d.predicted),
    y: rows.map((d: any) => d.observed),
  });
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Model governance</h1>
          <p>How every component was evaluated, on which data, and where it falls short. All figures are reproduced by <code>python scripts/run_evaluation.py</code>.</p>
        </div>
        <span className={`chip ${info.kind === "server" ? "ok" : "warn"}`}>{info.label}</span>
      </header>

      <div className="stack" style={{ gap: 18 }}>
        {ans && (
          <section className="grid cols-4">
            <div className="panel metric">
              <div className="value">{Math.round(ans.grounded_hit_rate * 100)}%</div>
              <div className="label">Answers citing a relevant guideline digest</div>
              <div className="note">{ans.questions} held out clinical questions</div>
            </div>
            <div className="panel metric">
              <div className="value">{ans.mean_faithfulness.toFixed(2)}</div>
              <div className="label">Mean faithfulness of claims</div>
              <div className="note">{ans.modes.includes("llm") ? "Local LLM synthesis" : "Extractive mode, faithful by construction; run the same script with --mode llm to score the language model"}</div>
            </div>
            <div className="panel metric">
              <div className="value">{ans.answers_with_invented_numbers}</div>
              <div className="label">Answers with numbers absent from sources</div>
            </div>
            <div className="panel metric">
              <div className="value">{ans.safety_probes_passed.replace("/", " of ")}</div>
              <div className="label">Safety probes passed</div>
              <div className="note">Out of scope, crisis language, identifiers, prompt injection</div>
            </div>
          </section>
        )}

        {ret && (
          <section className="panel panel-pad stack">
            <h2>Retrieval</h2>
            <p className="sub">
              Index of {ret.index.passages.toLocaleString()} passages from {ret.index.documents.toLocaleString()} documents. The reranker is a logistic regression over eleven interpretable features, trained on {ret.reranker_training.training_queries.toLocaleString()} synthetic queries and never on the clinical question sets.
            </p>
            <div className="grid cols-2" style={{ alignItems: "start" }}>
              <div>
                <h3>Blind clinical questions (n = {ret.clinical_questions_blind?.bm25.queries})</h3>
                <MetricTable rows={Object.entries(ret.clinical_questions_blind ?? {})} best="hybrid_rrf_reranked" />
              </div>
              <div>
                <h3>Development clinical questions (n = {ret.clinical_questions_development.bm25.queries})</h3>
                <MetricTable rows={Object.entries(ret.clinical_questions_development)} best="hybrid_rrf_reranked" />
              </div>
              <div>
                <h3>PubMed known item queries (n = {ret.pubmed_known_item_test.bm25.queries})</h3>
                <MetricTable rows={Object.entries(ret.pubmed_known_item_test)} best="bm25" />
                <p className="sub" style={{ marginTop: 8 }}>
                  Known item queries are sampled from the abstract itself, which favours exact lexical matching; BM25 is strongest here by design of the test.
                </p>
              </div>
              {browser && (
                <div>
                  <h3>Browser demo engine, blind questions</h3>
                  <MetricTable rows={Object.entries(browser.clinical_blind)} best="bm25_lexical_reranker" />
                  <p className="sub" style={{ marginTop: 8 }}>
                    The hosted demo omits the dense encoder to stay light, so its recall is lower than the full service. The gap is reported rather than hidden.
                  </p>
                </div>
              )}
            </div>
          </section>
        )}

        {surv && (
          <section className="panel panel-pad stack">
            <h2>Prognostic model</h2>
            <p className="sub">
              Endpoint: {surv.endpoint}. Developed on {surv.development_cohort.n.toLocaleString()} METABRIC patients ({surv.development_cohort.events} events); externally validated on {surv.external_cohort.n} GBSG2 patients. {surv.external_cohort.note}
            </p>
            <div className="grid cols-2" style={{ alignItems: "start" }}>
              <table className="tableview">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="num">C index, cross validated</th>
                    <th className="num">C index, external</th>
                    <th className="num">Brier score, external</th>
                  </tr>
                </thead>
                <tbody>
                  {(["cox_ph", "random_survival_forest", "gradient_boosted_survival", "npi_baseline"] as const).map((m) => (
                    <tr key={m} className={m === "cox_ph" ? "best" : ""}>
                      <td>{{ cox_ph: "Cox proportional hazards (deployed)", random_survival_forest: "Random survival forest", gradient_boosted_survival: "Gradient boosted survival", npi_baseline: "Nottingham Prognostic Index" }[m]}</td>
                      <td className="num">
                        {surv.core_cross_validation[m].c_index_harrell_mean.toFixed(3)} ± {surv.core_cross_validation[m].c_index_harrell_sd.toFixed(3)}
                      </td>
                      <td className="num">{surv.external_validation[m].c_index_harrell.toFixed(3)}</td>
                      <td className="num">{surv.external_validation[m].integrated_brier_score?.toFixed(3) ?? "not applicable"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div>
                <h3>Calibration</h3>
                <StepChart
                  ariaLabel="Calibration: predicted versus observed survival by risk quintile"
                  series={[
                    { name: "Ideal", color: "var(--text-3)", x: [0, 1], y: [0, 1], step: false, dashed: true },
                    { name: "METABRIC, 10 years", color: "var(--s1)", ...calib(surv.internal_calibration_120m), step: false },
                    { name: "GBSG2, 5 years", color: "var(--s2)", ...calib(surv.external_calibration_60m), step: false },
                  ]}
                  xMax={1}
                  height={260}
                  xLabel="Predicted relapse free survival"
                  yLabel="Observed"
                  xTicks={[0, 0.25, 0.5, 0.75, 1]}
                  formatX={(v) => `Predicted ${Math.round(v * 100)}%`}
                />
                <p className="sub">Internally the model is well calibrated. On GBSG2 observed survival is consistently lower than predicted: discrimination transfers, absolute risk does not, so recalibration is required before local use.</p>
              </div>
            </div>
          </section>
        )}

        <div className="grid cols-2" style={{ alignItems: "start" }}>
          {sub && (
            <section className="panel panel-pad stack">
              <h2>Intrinsic subtype</h2>
              <table className="tableview">
                <thead>
                  <tr>
                    <th>Model, five fold cross validation</th>
                    <th className="num">Accuracy</th>
                    <th className="num">Macro F1</th>
                    <th className="num">Kappa</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["PAM50 expression, logistic regression", sub.expression_pam50.logistic_regression_cv],
                    ["PAM50 expression, nearest centroid", sub.expression_pam50.nearest_centroid_cv],
                    ["Routine pathology, logistic regression", sub.surrogate_pathology.logistic_regression_cv],
                    ["Routine pathology, gradient boosting", sub.surrogate_pathology.gradient_boosting_cv],
                    ["St Gallen rule baseline", sub.surrogate_pathology.st_gallen_rule_baseline],
                  ].map(([n, m]: any) => (
                    <tr key={n}>
                      <td>{n}</td>
                      <td className="num">{m.accuracy.toFixed(2)}</td>
                      <td className="num">{m.macro_f1.toFixed(2)}</td>
                      <td className="num">{m.cohen_kappa.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="sub">Reference labels in METABRIC were themselves derived from expression, so the expression score measures portability of a compact model. The pathology score shows how often immunohistochemistry alone gets the molecular subtype wrong.</p>
            </section>
          )}
          {nli && (
            <section className="panel panel-pad stack">
              <h2>Evidence checking on NLI4CT</h2>
              <p className="sub">Expert statements about breast cancer trial reports (SemEval 2023), gold test split of {nli.splits.test} statements.</p>
              <table className="tableview">
                <tbody>
                  <tr>
                    <td>Evidence line ranking, mean average precision</td>
                    <td className="num">{nli.evidence_ranking_test.mean_average_precision.toFixed(2)}</td>
                    <td className="num muted">random {nli.evidence_ranking_test.mean_average_precision_random.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Evidence lines, precision at 3</td>
                    <td className="num">{nli.evidence_ranking_test.precision_at_3.toFixed(2)}</td>
                    <td className="num muted">random {nli.evidence_ranking_test.precision_at_3_random.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Claim verification accuracy</td>
                    <td className="num">{nli.claim_verification_test.accuracy.toFixed(2)}</td>
                    <td className="num muted">majority {nli.claim_verification_test.majority_baseline_accuracy.toFixed(2)}</td>
                  </tr>
                </tbody>
              </table>
              <div className="banner">
                <Icon name="info" />
                {nli.interpretation}
              </div>
            </section>
          )}
        </div>

        <section className="panel panel-pad stack">
          <h2>Known limitations</h2>
          <ul style={{ margin: 0, paddingLeft: 18, color: "var(--text-2)", lineHeight: 1.7 }}>
            <li>The literature corpus is the breast oncology subset of PubMed 200k RCT (abstracts to 2017). Trials published later, such as KATHERINE or KEYNOTE-522, are represented through the curated digests, not their abstracts. The ingestion supports full PubMed through E-utilities for production.</li>
            <li>Knowledge cards are authored summaries, reviewed October 2026; they must be checked against current NICE, ESMO or local protocols.</li>
            <li>The prognostic model reflects historical treatment and overestimates relapse free survival on GBSG2; it is prognostic, not causal.</li>
            <li>Trial screening covers structured criteria (age, performance status, receptors, setting, menopause, BRCA, brain metastases, pregnancy, LVEF, prior therapy). Laboratory thresholds and investigator judgement remain manual.</li>
            <li>Claim verification is lexical and numeric; it catches unsupported wording and invented numbers but cannot prove logical entailment.</li>
          </ul>
        </section>
      </div>
    </>
  );
}
