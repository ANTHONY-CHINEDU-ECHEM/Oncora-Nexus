// Line for line port of oncora/rag/text.py. The dictionaries are loaded from
// the exported lexicon so that both engines normalise text identically.

export interface Lexicon {
  uk_to_us: Record<string, string>;
  synonyms: Record<string, string>;
  stopwords: string[];
  keep_short: string[];
  intent_rules: [string, string][];
  section_priors: Record<string, Record<string, number>>;
  subtype_rules: Record<string, string>;
  setting_rules: Record<string, string>;
  population_rules: Record<string, string>;
  drugs: string[];
}

let phrasePatterns: [RegExp, string][] = [];
let stopwords = new Set<string>();
let keepShort = new Set<string>();

const escapeRx = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function initText(lex: Lexicon): void {
  const pairs = [...Object.entries(lex.uk_to_us), ...Object.entries(lex.synonyms)];
  pairs.sort((a, b) => b[0].length - a[0].length);
  phrasePatterns = pairs.map(([k, v]) => [new RegExp(`(?<![a-z0-9])${escapeRx(k)}(?![a-z0-9])`, "g"), v]);
  stopwords = new Set(lex.stopwords);
  keepShort = new Set(lex.keep_short);
}

export function normalise(text: string): string {
  let low = ` ${text.toLowerCase()} `;
  for (const [rx, repl] of phrasePatterns) {
    rx.lastIndex = 0;
    low = low.replace(rx, repl);
  }
  return low.trim();
}

const SUFFIXES: [string, string][] = [
  ["ational", "ate"],
  ["ization", "ize"],
  ["ies", "y"],
  ["sses", "ss"],
  ["ness", ""],
  ["ments", "ment"],
  ["ings", ""],
  ["ing", ""],
  ["edly", ""],
  ["ed", ""],
  ["es", "e"],
  ["s", ""],
];

export function stem(token: string): string {
  if (token.length <= 4 || /^\d+$/.test(token)) return token;
  if (/(shes|ches|xes|zes)$/.test(token)) return token.slice(0, -2);
  for (const [suffix, repl] of SUFFIXES) {
    if (token.endsWith(suffix) && token.length - suffix.length >= 3) {
      if (suffix === "s" && /(ss|us|is)$/.test(token)) return token;
      return token.slice(0, token.length - suffix.length) + repl;
    }
  }
  return token;
}

const TOKEN_RX = /[a-z0-9]+(?:\.[0-9]+)?/g;

export function tokenize(text: string, keepStopwords = false): string[] {
  const out: string[] = [];
  const matches = normalise(text).match(TOKEN_RX) ?? [];
  for (const tok of matches) {
    if (!keepStopwords && stopwords.has(tok)) continue;
    if (tok.length < 3 && !keepShort.has(tok) && !/^\d+$/.test(tok)) continue;
    out.push(stem(tok));
  }
  return out;
}

const NUMBER_RX = /(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(%|percent|mg|gy|months?|years?|weeks?|days?)?/g;

export function numbers(text: string): Set<string> {
  const found = new Set<string>();
  const joined = text.replace(/(\d)\s+\.(\d)/g, "$1.$2").replace(/(\d)\s+%/g, "$1%");
  for (const m of joined.toLowerCase().matchAll(NUMBER_RX)) {
    let v = m[1];
    if (v.includes(".")) v = v.replace(/0+$/, "").replace(/\.$/, "");
    found.add(v);
    if (m[2]) found.add(v + (m[2] === "%" || m[2] === "percent" ? "%" : m[2][0]));
  }
  return found;
}

const SENT_SPLIT = /(?<=[.!?])\s+(?=[A-Z0-9(])/;

export function splitSentences(text: string): string[] {
  const parts = text.trim().split(SENT_SPLIT).map((p) => p.trim()).filter(Boolean);
  return parts.length ? parts : text.trim() ? [text.trim()] : [];
}
