// Inline stroke icons (24 px grid, 1.75 stroke) so the interface ships with no icon font.
const PATHS: Record<string, string> = {
  overview: "M4 13h6V4H4zM14 20h6v-9h-6zM4 20h6v-3H4zM14 4v3h6V4z",
  case: "M9 4h6a1 1 0 0 1 1 1v2H8V5a1 1 0 0 1 1-1zM6 7h12a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1zM9 12h6M9 16h4",
  copilot: "M5 5h14v10H9l-4 4zM9 9h6M9 12h3",
  trials: "M9 3v5l-5 9a2 2 0 0 0 1.8 3h12.4a2 2 0 0 0 1.8-3l-5-9V3M8 3h8M7 14h10",
  cohort: "M3 20h18M6 16V9M11 16V5M16 16v-4M21 16V8",
  library: "M5 4h4v16H5zM10 4h4v16h-4zM15 5l4 1-3 14-4-1z",
  governance: "M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6zM9 12l2 2 4-4",
  check: "M5 12l4 4 10-10",
  cross: "M6 6l12 12M18 6L6 18",
  question: "M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18zM9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14M12 17.2v.1",
  info: "M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18zM12 11v6M12 7.5v.1",
  alert: "M12 4l9 16H3zM12 10v4M12 17v.1",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  send: "M4 12l16-8-6 16-2-6z",
  spark: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6",
  chevron: "M9 6l6 6-6 6",
  shield: "M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z",
  doc: "M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM14 3v5h5M9 13h6M9 17h6",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z",
};

export function Icon({ name, className, title }: { name: keyof typeof PATHS | string; className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" aria-hidden={title ? undefined : true} role={title ? "img" : undefined}>
      {title && <title>{title}</title>}
      <path d={PATHS[name] ?? PATHS.info} />
    </svg>
  );
}

// Brand mark: a stylised cell nucleus (haematoxylin) within eosin stained cytoplasm,
// two arcs suggesting the connection between evidence and the patient.
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <circle cx="16" cy="16" r="14" fill="var(--eosin-soft)" />
      <path d="M5.5 19.5C8 25 15 27.5 21 25" stroke="var(--eosin)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M26.5 12.5C24 7 17 4.5 11 7" stroke="var(--eosin)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <circle cx="16" cy="16" r="6" fill="var(--haem)" />
      <circle cx="18" cy="14" r="1.6" fill="var(--panel)" opacity="0.85" />
    </svg>
  );
}
