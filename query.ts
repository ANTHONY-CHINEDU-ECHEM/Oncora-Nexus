import type { ParsedQuery } from "./types";
import { normalise, tokenize, type Lexicon } from "./text";

let lex: Lexicon;
let intentRules: [string, RegExp][] = [];
let subtypeRules: [string, RegExp][] = [];
let settingRules: [string, RegExp][] = [];
let populationRules: [string, RegExp][] = [];

export function initQuery(l: Lexicon): void {
  lex = l;
  intentRules = l.intent_rules.map(([n, p]) => [n, new RegExp(p, "i")]);
  subtypeRules = Object.entries(l.subtype_rules).map(([k, p]) => [k, new RegExp(p, "i")]);
  settingRules = Object.entries(l.setting_rules).map(([k, p]) => [k, new RegExp(p, "i")]);
  populationRules = Object.entries(l.population_rules).map(([k, p]) => [k, new RegExp(p, "i")]);
}

const escapeRx = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function parseQuery(text: string): ParsedQuery {
  const norm = normalise(text);
  const drugs = lex.drugs.filter((d) => new RegExp(`(?<![a-z])${escapeRx(d)}(?![a-z])`).test(norm));
  if (norm.includes("aromatase inhibitor")) drugs.push("aromatase inhibitor");
  if (norm.includes("cdk4 inhibitor")) drugs.push("cdk4 inhibitor");
  const intents = intentRules.filter(([, rx]) => rx.test(norm)).map(([n]) => n);
  return {
    raw: text,
    tokens: tokenize(text),
    intents,
    drugs,
    subtypes: subtypeRules.filter(([, rx]) => rx.test(norm)).map(([k]) => k),
    settings: settingRules.filter(([, rx]) => rx.test(norm)).map(([k]) => k),
    populations: populationRules.filter(([, rx]) => rx.test(norm)).map(([k]) => k),
    primary_intent: intents[0] ?? "general",
  };
}

export function sectionPrior(q: ParsedQuery, section: string): number {
  let best = 0;
  for (const intent of q.intents.slice(0, 2)) best = Math.max(best, lex.section_priors[intent]?.[section] ?? 0);
  return best;
}
