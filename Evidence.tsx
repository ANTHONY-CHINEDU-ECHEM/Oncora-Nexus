import { Fragment, useState } from "react";
import type { AskResponse, EvidenceHit, SentenceCheck } from "../engine/types";
import { Icon } from "./Icon";

const STATUS_LABEL: Record<SentenceCheck["status"], string> = {
  supported: "Supported by the cited passage",
  partial: "Partly supported; check the cited passage",
  unsupported: "Not supported by the cited passage",
  uncited: "No citation",
  meta: "Caveat",
};

function cleanAnswer(answer: string) {
  const sections: { title: string; lines: string[] }[] = [];
  let cur = { title: "Bottom line", lines: [] as string[] };
  for (const raw of answer.split("\n")) {
    const line = raw.trim();
    const m = line.match(/^(BOTTOM LINE|KEY EVIDENCE|CAVEATS)\s*:?\s*(.*)$/i);
    if (m) {
      if (cur.lines.length) sections.push(cur);
      const t = m[1].toUpperCase();
      cur = { title: t === "BOTTOM LINE" ? "Bottom line" : t === "KEY EVIDENCE" ? "Key evidence" : "Caveats", lines: [] };
      if (m[2]) cur.lines.push(m[2]);
    } else if (line) cur.lines.push(line);
  }
  if (cur.lines.length) sections.push(cur);
  return sections;
}

function normaliseKey(s: string) {
  return s.replace(/\[S\d+\]/g, "").replace(/[^a-z0-9]/gi, "").toLowerCase().slice(0, 60);
}

// Renders an answer with every claim underlined by its verification status and
// every citation as a button linked to the evidence list.
export function VerifiedAnswer({ res, active, onCite }: { res: AskResponse; active: number | null; onCite: (n: number | null) => void }) {
  const checks = res.verification?.sentences ?? [];
  const lookup = new Map(checks.map((c) => [normaliseKey(c.text), c]));
  const renderText = (text: string) => {
    const parts = text.split(/(\[S\d+\])/g);
    return parts.map((p, i) => {
      const m = p.match(/\[S(\d+)\]/);
      if (m) {
        const n = Number(m[1]);
        return (
          <button key={i} className={`cite ${active === n ? "active" : ""}`} onMouseEnter={() => onCite(n)} onMouseLeave={() => onCite(null)} onFocus={() => onCite(n)} onBlur={() => onCite(null)} onClick={() => document.getElementById(`ev-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" })} aria-label={`Source ${n}`}>
            S{n}
          </button>
        );
      }
      return <Fragment key={i}>{p}</Fragment>;
    });
  };
  const renderSentence = (s: string, key: number) => {
    const chk = lookup.get(normaliseKey(s));
    const cls = chk && chk.status !== "meta" ? `claim ${chk.status}` : "";
    const isActive = chk && active !== null && chk.citations.includes(active);
    return (
      <span key={key} className={`${cls} ${isActive ? "active" : ""}`} title={chk ? `${STATUS_LABEL[chk.status]}${chk.status !== "meta" && chk.citations.length ? `, overlap ${Math.round(chk.support * 100)}%` : ""}${chk.numbers_ok ? "" : ", contains a number not found in the source"}` : undefined}>
        {renderText(s)}{" "}
      </span>
    );
  };
  const splitSent = (line: string) => line.split(/(?<=[.!?])\s+(?=[A-Z0-9(])/);
  return (
    <div className="answer">
      {cleanAnswer(res.answer).map((sec) => (
        <div key={sec.title}>
          <h4>{sec.title}</h4>
          {sec.title === "Bottom line" ? (
            <p>{sec.lines.flatMap(splitSent).map(renderSentence)}</p>
          ) : (
            <ul>
              {sec.lines.map((l, i) => (
                <li key={i}>{splitSent(l.replace(/^[*•]\s*/, "")).map(renderSentence)}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
}

const SOURCE_LABEL: Record<string, string> = { guideline: "Guideline digest", pubmed: "PubMed abstract", trial: "Trial registry" };

export function EvidenceCard({ hit, n, active, onHover }: { hit: EvidenceHit; n: number; active: boolean; onHover: (n: number | null) => void }) {
  const [open, setOpen] = useState(false);
  const ident = hit.source === "pubmed" ? `PMID ${String(hit.meta.pmid ?? hit.doc_id.replace("PMID:", ""))}` : hit.source === "trial" ? hit.doc_id : hit.doc_id;
  const href = hit.url || (hit.source === "pubmed" ? `https://pubmed.ncbi.nlm.nih.gov/${hit.doc_id.replace("PMID:", "")}/` : undefined);
  return (
    <article id={`ev-${n}`} className={`evidence-card ${active ? "active" : ""}`} onMouseEnter={() => onHover(n)} onMouseLeave={() => onHover(null)}>
      <div className="spread">
        <span className="row" style={{ gap: 8 }}>
          <span className="num">S{n}</span>
          <span className={`chip ${hit.source === "guideline" ? "eosin" : hit.source === "trial" ? "" : "haem"}`}>{SOURCE_LABEL[hit.source]}</span>
        </span>
        <span className="src">{hit.section}</span>
      </div>
      <div className="title">{hit.title}</div>
      <div className={`body ${open ? "open" : ""}`}>{hit.text.replace(/\s+([,.;:)%])/g, "$1").replace(/\(\s+/g, "(")}</div>
      <div className="spread" style={{ marginTop: 8 }}>
        <span className="src">
          {hit.evidence_label}, {ident}
        </span>
        <span className="row" style={{ gap: 12 }}>
          <button className="linklike" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? "Show less" : "Read passage"}
          </button>
          {href && (
            <a className="linklike" href={href} target="_blank" rel="noreferrer">
              Source <Icon name="external" className="" />
            </a>
          )}
        </span>
      </div>
    </article>
  );
}

export function ConfidenceBadge({ res }: { res: AskResponse }) {
  const lvl = res.confidence.level;
  const cls = lvl === "high" ? "ok" : lvl === "moderate" ? "warn" : lvl === "low" ? "bad" : "";
  const label = lvl === "none" ? "Not answered" : `${lvl[0].toUpperCase()}${lvl.slice(1)} confidence`;
  return (
    <span className={`chip ${cls}`} title={`Score ${res.confidence.score}`}>
      <Icon name="shield" />
      {label}
    </span>
  );
}

export function VerificationSummary({ res }: { res: AskResponse }) {
  const v = res.verification;
  if (!v) return null;
  const claims = v.sentences.filter((s) => s.status !== "meta");
  const count = (st: string) => claims.filter((s) => s.status === st).length;
  return (
    <div className="row" style={{ gap: 8 }}>
      <span className="chip ok" title="Claims whose wording and numbers are found in the cited passage">
        <Icon name="check" /> {count("supported")} supported
      </span>
      {count("partial") > 0 && <span className="chip warn">{count("partial")} partly supported</span>}
      {count("unsupported") + count("uncited") > 0 && (
        <span className="chip bad">
          <Icon name="alert" /> {count("unsupported") + count("uncited")} need checking
        </span>
      )}
      {v.unsupported_numbers.length > 0 && <span className="chip bad">Numbers not in sources: {v.unsupported_numbers.join(", ")}</span>}
    </div>
  );
}
