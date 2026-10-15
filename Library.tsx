import { useEffect, useMemo, useState } from "react";
import type { Card, Engine } from "../engine/types";

export function LibraryView({ engine, focus }: { engine: Engine; focus: string | null }) {
  const [cards, setCards] = useState<Card[]>([]);
  const [q, setQ] = useState("");
  const [cat, setCat] = useState<string>("All");
  useEffect(() => {
    engine.cards().then(setCards);
  }, [engine]);
  useEffect(() => {
    if (focus && cards.length) setTimeout(() => document.getElementById(`card-${focus}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  }, [focus, cards]);
  const cats = useMemo(() => ["All", ...Array.from(new Set(cards.map((c) => c.category)))], [cards]);
  const shown = cards.filter((c) => (cat === "All" || c.category === cat) && (!q || `${c.title} ${c.body} ${c.tags.join(" ")}`.toLowerCase().includes(q.toLowerCase())));
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Knowledge library</h1>
          <p>Forty eight authored evidence digests that ground the pathway rules and lead most answers. Each summarises guideline positions and landmark trials and lists its sources; verify against the current guideline before clinical use.</p>
        </div>
      </header>
      <div className="row" style={{ marginBottom: 16 }}>
        <input className="input" style={{ maxWidth: 360 }} placeholder="Filter by drug, trial or topic" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter cards" />
        <select className="input" style={{ maxWidth: 260 }} value={cat} onChange={(e) => setCat(e.target.value)} aria-label="Category">
          {cats.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <span className="sub">{shown.length} cards</span>
      </div>
      <div className="grid cols-2" style={{ alignItems: "start" }}>
        {shown.map((c) => (
          <article key={c.id} id={`card-${c.id}`} className="panel panel-pad" style={focus === c.id ? { borderColor: "var(--eosin)", boxShadow: "0 0 0 3px var(--eosin-soft)" } : undefined}>
            <div className="spread">
              <span className="chip eosin">{c.category}</span>
              <span className="sub">{c.id}</span>
            </div>
            <h2 style={{ fontFamily: "var(--prose)", fontSize: 19, margin: "12px 0 8px" }}>{c.title}</h2>
            <p style={{ fontFamily: "var(--prose)", fontSize: 15.5, lineHeight: 1.6, margin: "0 0 12px" }}>{c.body}</p>
            <div className="sub">
              <strong style={{ color: "var(--text-2)" }}>Sources: </strong>
              {c.references.join("; ")}
            </div>
          </article>
        ))}
      </div>
    </>
  );
}
