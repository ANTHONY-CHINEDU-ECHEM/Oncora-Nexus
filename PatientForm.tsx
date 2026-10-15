import type { PatientProfile } from "../engine/types";
import { PRESETS, THERAPY_OPTIONS } from "../lib/presets";

type Setter = (p: PatientProfile) => void;

function num(v: string): number | null {
  if (v.trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function triBool(v: string): boolean | null {
  return v === "yes" ? true : v === "no" ? false : null;
}

function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: [T, string][]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, l]) => (
        <button key={v} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>
          {l}
        </button>
      ))}
    </div>
  );
}

export function PatientForm({ patient, onChange, onPreset }: { patient: PatientProfile; onChange: Setter; onPreset: (id: string) => void }) {
  const set = <K extends keyof PatientProfile>(k: K, v: PatientProfile[K]) => onChange({ ...patient, [k]: v });
  const toggleTherapy = (t: string) => {
    const has = patient.prior_therapies.includes(t);
    set("prior_therapies", has ? patient.prior_therapies.filter((x) => x !== t) : [...patient.prior_therapies, t]);
  };
  const triVal = (v: boolean | null | undefined) => (v === true ? "yes" : v === false ? "no" : "");

  return (
    <form className="panel profile" onSubmit={(e) => e.preventDefault()} aria-label="Patient profile">
      <div className="profile-head">
        <h2>Patient profile</h2>
        <p className="sub">Deidentified summary. Every panel updates as you edit; leave unknown fields blank.</p>
        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="preset">Load a teaching case</label>
          <select id="preset" defaultValue="" onChange={(e) => e.target.value && onPreset(e.target.value)}>
            <option value="">Choose a case</option>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <fieldset>
        <legend>Patient</legend>
        <div className="field">
          <label htmlFor="age">Age</label>
          <input id="age" type="number" min={15} max={110} value={patient.age ?? ""} onChange={(e) => set("age", num(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="ecog">ECOG</label>
          <select id="ecog" value={patient.ecog ?? ""} onChange={(e) => set("ecog", num(e.target.value))}>
            <option value="">Unknown</option>
            {[0, 1, 2, 3, 4].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
        <div className="field full">
          <label>Sex</label>
          <Seg label="Sex" value={patient.sex} options={[["female", "Female"], ["male", "Male"]]} onChange={(v) => set("sex", v)} />
        </div>
        <div className="field full">
          <label>Menopausal status</label>
          <Seg label="Menopausal status" value={patient.menopausal_status} options={[["premenopausal", "Pre"], ["postmenopausal", "Post"], ["unknown", "Unknown"]]} onChange={(v) => set("menopausal_status", v)} />
        </div>
      </fieldset>

      <fieldset>
        <legend>Disease</legend>
        <div className="field full">
          <label>Setting</label>
          <Seg label="Disease setting" value={patient.setting} options={[["early", "Early"], ["locally_advanced", "Locally advanced"], ["metastatic", "Metastatic"]]} onChange={(v) => set("setting", v)} />
        </div>
        <div className="field">
          <label htmlFor="size">Tumour size (mm)</label>
          <input id="size" type="number" min={0} value={patient.tumour_size_mm ?? ""} onChange={(e) => set("tumour_size_mm", num(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="grade">Grade</label>
          <select id="grade" value={patient.grade ?? ""} onChange={(e) => set("grade", num(e.target.value))}>
            <option value="">Unknown</option>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="nodes">Positive nodes</label>
          <input id="nodes" type="number" min={0} value={patient.positive_nodes ?? ""} onChange={(e) => set("positive_nodes", num(e.target.value))} />
        </div>
        <div className="field full">
          <label htmlFor="residual">Residual disease after neoadjuvant therapy</label>
          <select id="residual" value={triVal(patient.residual_disease)} onChange={(e) => set("residual_disease", triBool(e.target.value))}>
            <option value="">Not applicable or unknown</option>
            <option value="yes">Yes, residual invasive disease</option>
            <option value="no">No, complete response</option>
          </select>
        </div>
      </fieldset>

      <fieldset>
        <legend>Receptors and biomarkers</legend>
        <div className="field">
          <label htmlFor="er">ER</label>
          <select id="er" value={patient.er_status} onChange={(e) => set("er_status", e.target.value as PatientProfile["er_status"])}>
            <option value="unknown">Unknown</option>
            <option value="positive">Positive</option>
            <option value="low">Low (1 to 10%)</option>
            <option value="negative">Negative</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="pr">PR</label>
          <select id="pr" value={patient.pr_status} onChange={(e) => set("pr_status", e.target.value as PatientProfile["pr_status"])}>
            <option value="unknown">Unknown</option>
            <option value="positive">Positive</option>
            <option value="negative">Negative</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="her2">HER2</label>
          <select id="her2" value={patient.her2_status} onChange={(e) => set("her2_status", e.target.value as PatientProfile["her2_status"])}>
            <option value="unknown">Unknown</option>
            <option value="positive">Positive</option>
            <option value="low">Low (IHC 1+ or 2+ ISH negative)</option>
            <option value="negative">Negative (IHC 0)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="brca">Germline BRCA</label>
          <select id="brca" value={patient.brca} onChange={(e) => set("brca", e.target.value as PatientProfile["brca"])}>
            <option value="unknown">Unknown</option>
            <option value="negative">No variant</option>
            <option value="brca1">BRCA1</option>
            <option value="brca2">BRCA2</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="rs">Recurrence score</label>
          <input id="rs" type="number" min={0} max={100} value={patient.recurrence_score ?? ""} onChange={(e) => set("recurrence_score", num(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="cps">PD-L1 CPS</label>
          <input id="cps" type="number" min={0} max={100} value={patient.pdl1_cps ?? ""} onChange={(e) => set("pdl1_cps", num(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="pik3ca">PIK3CA</label>
          <select id="pik3ca" value={triVal(patient.pik3ca_mutation)} onChange={(e) => set("pik3ca_mutation", triBool(e.target.value))}>
            <option value="">Not tested</option>
            <option value="yes">Mutated</option>
            <option value="no">Wild type</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="esr1">ESR1</label>
          <select id="esr1" value={triVal(patient.esr1_mutation)} onChange={(e) => set("esr1_mutation", triBool(e.target.value))}>
            <option value="">Not tested</option>
            <option value="yes">Mutated</option>
            <option value="no">Wild type</option>
          </select>
        </div>
      </fieldset>

      <fieldset>
        <legend>History and safety</legend>
        <div className="field full">
          <label>Prior systemic therapy</label>
          <div className="row" style={{ gap: 6 }}>
            {THERAPY_OPTIONS.map(([v, l]) => (
              <button key={v} type="button" className={`chip ${patient.prior_therapies.includes(v) ? "haem" : ""}`} style={{ cursor: "pointer" }} aria-pressed={patient.prior_therapies.includes(v)} onClick={() => toggleTherapy(v)}>
                {l}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label htmlFor="lvef">LVEF (%)</label>
          <input id="lvef" type="number" min={10} max={90} value={patient.lvef ?? ""} onChange={(e) => set("lvef", num(e.target.value))} />
        </div>
        <div className="field" style={{ justifyContent: "flex-end", gap: 8 }}>
          <label className="check">
            <input type="checkbox" checked={patient.brain_metastases} onChange={(e) => set("brain_metastases", e.target.checked)} /> Brain metastases
          </label>
          <label className="check">
            <input type="checkbox" checked={patient.visceral_crisis} onChange={(e) => set("visceral_crisis", e.target.checked)} /> Visceral crisis
          </label>
          <label className="check">
            <input type="checkbox" checked={patient.pregnant} onChange={(e) => set("pregnant", e.target.checked)} /> Pregnant
          </label>
        </div>
      </fieldset>
    </form>
  );
}
