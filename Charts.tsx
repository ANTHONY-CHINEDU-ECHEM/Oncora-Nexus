import { useEffect, useMemo, useRef, useState } from "react";

// Measure the container so SVG text renders at its true pixel size.
function useWidth(fallback = 640) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => setW(Math.max(280, Math.round(entries[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export interface Series {
  name: string;
  color: string;
  x: number[];
  y: number[];
  lower?: number[];
  upper?: number[];
  dashed?: boolean;
  step?: boolean;
}

interface StepChartProps {
  series: Series[];
  xMax?: number;
  yMin?: number;
  height?: number;
  xLabel: string;
  yLabel: string;
  xTicks?: number[];
  formatY?: (v: number) => string;
  markers?: { x: number; label: string }[];
  ariaLabel: string;
  formatX?: (v: number) => string;
}


function pathFor(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number, step: boolean): string {
  let d = "";
  xs.forEach((x, i) => {
    const X = sx(x);
    const Y = sy(ys[i]);
    if (i === 0) d += `M${X},${Y}`;
    else if (step) d += `H${X}V${Y}`;
    else d += `L${X},${Y}`;
  });
  return d;
}

function bandFor(xs: number[], lo: number[], hi: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  const top = xs.map((x, i) => `${i === 0 ? "M" : "L"}${sx(x)},${sy(hi[i])}`).join("");
  const bottom = xs
    .map((_, j) => xs.length - 1 - j)
    .map((i) => `L${sx(xs[i])},${sy(lo[i])}`)
    .join("");
  return `${top}${bottom}Z`;
}

// Survival style chart: step lines, optional confidence bands, crosshair tooltip
// that reads every series at the hovered time.
export function StepChart({ series, xMax, yMin = 0, height = 300, xLabel, yLabel, xTicks, formatY = (v) => `${Math.round(v * 100)}%`, markers = [], ariaLabel, formatX = (v) => `${Math.round(v)} months` }: StepChartProps) {
  const [wrapRef, W] = useWidth();
  const pad = { l: 46, r: 18, t: 12, b: 40 };
  const H = Math.min(height, Math.max(220, W * 0.62));
  const maxX = xMax ?? Math.max(...series.flatMap((s) => s.x));
  const sx = (v: number) => pad.l + (v / maxX) * (W - pad.l - pad.r);
  const sy = (v: number) => pad.t + (1 - (v - yMin) / (1 - yMin)) * (H - pad.t - pad.b);
  const ticks = xTicks ?? Array.from({ length: 6 }, (_, i) => Math.round((maxX / 5) * i));
  const yTicks = [0, 0.25, 0.5, 0.75, 1].filter((v) => v >= yMin);
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);

  const clipped = useMemo(
    () =>
      series.map((s) => {
        const idx = s.x.map((x, i) => (x <= maxX ? i : -1)).filter((i) => i >= 0);
        return { ...s, x: idx.map((i) => s.x[i]), y: idx.map((i) => s.y[i]), lower: s.lower && idx.map((i) => s.lower![i]), upper: s.upper && idx.map((i) => s.upper![i]) };
      }),
    [series, maxX],
  );

  const valueAt = (s: Series, t: number) => {
    let v = s.y[0];
    for (let i = 0; i < s.x.length && s.x[i] <= t; i++) v = s.y[i];
    return v;
  };

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const t = ((px - pad.l) / (W - pad.l - pad.r)) * maxX;
    setHover(t >= 0 && t <= maxX ? t : null);
  };

  return (
    <div className="chart" style={{ position: "relative" }} ref={wrapRef}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <g className="grid">
          {yTicks.map((v) => (
            <line key={v} x1={pad.l} x2={W - pad.r} y1={sy(v)} y2={sy(v)} />
          ))}
        </g>
        {yTicks.map((v) => (
          <text key={v} x={pad.l - 8} y={sy(v) + 4} textAnchor="end">
            {formatY(v)}
          </text>
        ))}
        {ticks.map((t) => (
          <text key={t} x={sx(t)} y={H - pad.b + 18} textAnchor="middle">
            {t}
          </text>
        ))}
        <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle">
          {xLabel}
        </text>
        <text transform={`translate(12 ${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle">
          {yLabel}
        </text>
        {markers.map((m) => (
          <g key={m.label}>
            <line x1={sx(m.x)} x2={sx(m.x)} y1={pad.t} y2={H - pad.b} stroke="var(--line-strong)" strokeDasharray="3 4" />
            <text x={sx(m.x) + 4} y={pad.t + 10}>
              {m.label}
            </text>
          </g>
        ))}
        {clipped.map((s) =>
          s.lower && s.upper ? <path key={`${s.name}-band`} d={bandFor(s.x, s.lower, s.upper, sx, sy)} fill={s.color} opacity={0.12} /> : null,
        )}
        {clipped.map((s) => (
          <path key={s.name} d={pathFor(s.x, s.y, sx, sy, s.step ?? true)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 5" : undefined} strokeLinejoin="round" />
        ))}
        {hover !== null && (
          <g>
            <line x1={sx(hover)} x2={sx(hover)} y1={pad.t} y2={H - pad.b} stroke="var(--text-3)" strokeWidth={1} />
            {clipped.map((s) => (
              <circle key={s.name} cx={sx(hover)} cy={sy(valueAt(s, hover))} r={4} fill={s.color} stroke="var(--panel)" strokeWidth={2} />
            ))}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${Math.min(80, (sx(hover) / W) * 100 + 2)}%`, top: 8 }}>
          <div style={{ fontWeight: 650, marginBottom: 4 }}>{formatX(hover)}</div>
          {clipped.map((s) => (
            <div key={s.name} className="spread">
              <span>
                <i style={{ display: "inline-block", width: 10, height: 3, background: s.color, marginRight: 6, verticalAlign: "middle" }} />
                {s.name}
              </span>
              <strong>{formatY(valueAt(s, hover))}</strong>
            </div>
          ))}
        </div>
      )}
      {series.length > 1 && (
        <div className="legend" style={{ marginTop: 6 }}>
          {series.map((s) => (
            <span key={s.name}>
              <i style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// Horizontal bars for probabilities and metrics, values labelled at the bar end.
export function HBars({ items, max = 1, format = (v: number) => `${Math.round(v * 100)}%`, highlight }: { items: { label: string; value: number; color?: string }[]; max?: number; format?: (v: number) => string; highlight?: string }) {
  return (
    <div className="stack" style={{ gap: 9 }}>
      {items.map((it) => (
        <div key={it.label} style={{ display: "grid", gridTemplateColumns: "minmax(110px, 34%) 1fr 56px", gap: 10, alignItems: "center", fontSize: 13.5 }}>
          <span style={{ fontWeight: highlight === it.label ? 650 : 500, color: highlight === it.label ? "var(--text)" : "var(--text-2)" }}>{it.label}</span>
          <div className="meter" style={{ height: 10 }}>
            <span style={{ width: `${Math.max(1.5, (it.value / max) * 100)}%`, background: it.color ?? (highlight === it.label ? "var(--haem)" : "var(--risk-low)") }} />
          </div>
          <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>{format(it.value)}</span>
        </div>
      ))}
    </div>
  );
}

// Diverging contribution chart (log hazard relative to the cohort average).
export function Contributions({ items }: { items: { label: string; value: number; note?: string }[] }) {
  const max = Math.max(0.05, ...items.map((i) => Math.abs(i.value)));
  return (
    <div className="stack" style={{ gap: 8 }}>
      {items.map((it) => {
        const w = (Math.abs(it.value) / max) * 50;
        const up = it.value > 0;
        return (
          <div key={it.label} style={{ display: "grid", gridTemplateColumns: "minmax(140px, 38%) 1fr 64px", gap: 10, alignItems: "center", fontSize: 13.5 }}>
            <span style={{ color: "var(--text-2)" }}>
              {it.label}
              {it.note && <span className="muted"> {it.note}</span>}
            </span>
            <div style={{ position: "relative", height: 12 }}>
              <div style={{ position: "absolute", left: "50%", top: -3, bottom: -3, width: 1, background: "var(--line-strong)" }} />
              <div
                style={{
                  position: "absolute",
                  top: 1,
                  height: 10,
                  borderRadius: 3,
                  background: up ? "var(--s2)" : "var(--s1)",
                  left: up ? "50%" : `${50 - w}%`,
                  width: `${Math.max(w, 0.6)}%`,
                }}
              />
            </div>
            <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 600, color: up ? "var(--s2)" : "var(--s1)" }}>
              {up ? "+" : ""}
              {it.value.toFixed(2)}
            </span>
          </div>
        );
      })}
      <p className="sub" style={{ fontSize: 12.5, margin: "4px 0 0" }}>
        Orange bars raise the hazard and blue bars lower it, measured in log hazard against the average patient.
      </p>
    </div>
  );
}

// Simple column histogram.
export function Columns({ edges, counts, label }: { edges: number[]; counts: number[]; label: string }) {
  const [wrapRef, W] = useWidth();
  const max = Math.max(...counts, 1);
  const H = 150;
  const bw = W / counts.length;
  return (
    <div className="chart" ref={wrapRef}>
      <svg viewBox={`0 0 ${W} ${H + 30}`} role="img" aria-label={label}>
        {counts.map((c, i) => {
          const h = (c / max) * H;
          return (
            <g key={i}>
              <rect x={i * bw + 2} y={H - h} width={bw - 4} height={h} rx={3} fill="var(--risk-mid)" opacity={0.85}>
                <title>{`${edges[i]} to ${edges[i + 1]}: ${c} patients`}</title>
              </rect>
              {i % 2 === 0 && (
                <text x={i * bw + bw / 2} y={H + 16} textAnchor="middle">
                  {edges[i]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
