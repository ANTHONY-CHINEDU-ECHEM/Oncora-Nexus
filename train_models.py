"""Train every predictive component and write the model registry.

Outputs (data/models):
    cox_core.json, cox_extended.json, survival_core_models.joblib
    subtype_models.joblib, subtype_surrogate.json
    trials_parsed.json, cohort_analytics.json
    registry.json   versions, training data fingerprints and headline metrics
Reports (reports):
    survival_eval.json, subtype_eval.json, trial_parsing.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.config import settings  # noqa: E402
from oncora.data.cohorts import load_gbsg2, load_metabric, load_pam50_expression, load_tcga  # noqa: E402
from oncora.data.trials import load_trials  # noqa: E402
from oncora.models import cohort_analytics, subtype, survival  # noqa: E402
from oncora.models.trial_matcher import TrialMatcher  # noqa: E402

log = logging.getLogger("train")


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = settings.paths.ensure()
    seed = settings.random_seed
    t0 = time.time()

    metabric = load_metabric(paths.raw)
    gbsg = load_gbsg2()
    tcga = load_tcga(paths.raw)

    log.info("Training survival models")
    surv = survival.train_and_report(metabric, gbsg, paths.models, seed)
    (paths.reports / "survival_eval.json").write_text(json.dumps(surv, indent=1))

    log.info("Training subtype models")
    sub = subtype.train_all(load_pam50_expression(paths.raw), metabric, paths.models, seed)
    (paths.reports / "subtype_eval.json").write_text(json.dumps(sub, indent=1))

    log.info("Building cohort analytics")
    cohort = cohort_analytics.build(metabric, tcga)
    (paths.models / "cohort_analytics.json").write_text(json.dumps(cohort))

    log.info("Parsing trial eligibility criteria")
    matcher = TrialMatcher(load_trials(paths.raw))
    (paths.models / "trials_parsed.json").write_text(json.dumps(matcher.export_parsed()))
    stats = matcher.parsing_stats()
    (paths.reports / "trial_parsing.json").write_text(json.dumps(stats, indent=2))

    registry = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "training_seconds": round(time.time() - t0, 1),
        "data_fingerprints": {
            "metabric_patient": fingerprint(paths.raw / "metabric" / "data_clinical_patient.txt"),
            "metabric_sample": fingerprint(paths.raw / "metabric" / "data_clinical_sample.txt"),
            "metabric_expression_pam50": fingerprint(paths.raw / "metabric" / "expression_pam50.tsv"),
            "tcga_patient": fingerprint(paths.raw / "tcga_brca" / "data_clinical_patient.txt"),
        },
        "models": {
            "prognosis_cox_core": {
                "version": "1.0.0",
                "endpoint": "relapse free survival",
                "cv_c_index": surv["core_cross_validation"]["cox_ph"]["c_index_harrell_mean"],
                "external_c_index_gbsg2": surv["external_validation"]["cox_ph"]["c_index_harrell"],
                "npi_c_index_gbsg2": surv["external_validation"]["npi_baseline"]["c_index_harrell"],
            },
            "subtype_expression_lr": {
                "version": "1.0.0",
                "cv_accuracy": sub["expression_pam50"]["logistic_regression_cv"]["accuracy"],
                "cv_macro_f1": sub["expression_pam50"]["logistic_regression_cv"]["macro_f1"],
            },
            "subtype_surrogate_lr": {
                "version": "1.0.0",
                "cv_accuracy": sub["surrogate_pathology"]["logistic_regression_cv"]["accuracy"],
                "cv_macro_f1": sub["surrogate_pathology"]["logistic_regression_cv"]["macro_f1"],
            },
            "trial_criteria_parser": {"version": "1.0.0", **{k: stats[k] for k in ("trials", "structured_criteria", "criteria_per_trial")}},
        },
    }
    (paths.models / "registry.json").write_text(json.dumps(registry, indent=2))
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
