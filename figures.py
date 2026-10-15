"""Report figures for the README and model card.

Colours follow a validated categorical order (blue, orange, aqua, yellow,
magenta, green, violet, red) for identity, a single blue ramp for ordered
magnitude, and neutral ink for text. Every multi series figure carries a legend
and direct labels so identity never depends on colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED_BAR = "#c9c7c1"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = {"250": "#86b6ef", "450": "#2a78d6", "650": "#104281", "100": "#cde2fb", "700": "#0d366b"}
RISK_COLOURS = {"low": BLUE_RAMP["250"], "intermediate": BLUE_RAMP["450"], "high": BLUE_RAMP["650"]}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def retrieval_ablation(report: dict, out: Path) -> None:
    blocks = [
        ("pubmed_known_item_test", "PubMed known item (n=500)"),
        ("clinical_questions_development", "Clinical questions, development (n=50)"),
        ("clinical_questions_blind", "Clinical questions, blind (n=20)"),
    ]
    blocks = [b for b in blocks if b[0] in report]
    names = {"bm25": "BM25", "dense_lsa": "Dense (LSA)", "hybrid_rrf": "Hybrid RRF", "hybrid_rrf_reranked": "Hybrid RRF + reranker"}
    fig, axes = plt.subplots(1, len(blocks), figsize=(4.2 * len(blocks), 3.4), sharex=True)
    for ax, (key, title) in zip(np.atleast_1d(axes), blocks, strict=False):
        res = report[key]
        variants = list(names)
        vals = [res[v]["mrr_at_10"] for v in variants]
        best = int(np.argmax(vals))
        colours = [CATEGORICAL[0] if v == "hybrid_rrf_reranked" else MUTED_BAR for v in variants]
        y = np.arange(len(variants))[::-1]
        ax.barh(y, vals, color=colours, height=0.6, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals, strict=False):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [names[v] for v in variants])
        ax.set_xlim(0, 1.08)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("MRR at 10")
        _ = best
    fig.suptitle("Retrieval ablation: learned reranking over hybrid fusion", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, out)


def survival_risk_groups(report: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    panels = [
        ("risk_groups_metabric", "risk_groups_metabric_logrank", "METABRIC development cohort", 180),
        ("risk_groups_gbsg2", "risk_groups_gbsg2_logrank", "GBSG2 external validation", 84),
    ]
    for ax, (key, lr_key, title, xmax) in zip(axes, panels, strict=False):
        for g in ("low", "intermediate", "high"):
            if g not in report[key]:
                continue
            km = report[key][g]
            t = np.array(km["time"])
            m = t <= xmax
            ax.step(t[m], np.array(km["survival"])[m], where="post", color=RISK_COLOURS[g], linewidth=2, label=f"{g.title()} risk (n={km['n']})")
            ax.fill_between(t[m], np.array(km["lower"])[m], np.array(km["upper"])[m], step="post", color=RISK_COLOURS[g], alpha=0.12, linewidth=0)
            ax.text(t[m][-1] + xmax * 0.01, np.array(km["survival"])[m][-1], g.title(), color=INK_2, fontsize=8.5, va="center")
        p = report[lr_key]["p_value"]
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.text(0.02, 0.03, f"Log rank p {'< 1e-6' if p < 1e-6 else f'= {p:.2g}'}", transform=ax.transAxes, fontsize=8.5, color=INK_2)
        ax.set_xlim(0, xmax * 1.12)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Months from diagnosis")
        ax.set_ylabel("Relapse free survival")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.07))
    fig.suptitle("Cox model risk groups separate relapse free survival in both cohorts", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, out)


def calibration(report: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    panels = [("internal_calibration_120m", "METABRIC, 10 year RFS (internal)"), ("external_calibration_60m", "GBSG2, 5 year RFS (external)")]
    for ax, (key, title) in zip(axes, panels, strict=False):
        rows = report[key]
        pred = np.array([r["predicted"] for r in rows])
        obs = np.array([r["observed"] for r in rows])
        lo = np.array([r["observed_lower"] for r in rows])
        hi = np.array([r["observed_upper"] for r in rows])
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label="Perfect calibration")
        ax.errorbar(pred, obs, yerr=[obs - lo, hi - obs], fmt="o", color=CATEGORICAL[0], ecolor=CATEGORICAL[0], elinewidth=1.4, capsize=3, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Risk quintile (95% CI)")
        ax.set_xlim(0.15, 0.95)
        ax.set_ylim(0.15, 0.95)
        ax.set_xlabel("Predicted survival")
        ax.set_ylabel("Observed (Kaplan Meier)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.legend(loc="upper left")
    fig.suptitle("Calibration: accurate internally, optimistic on the external cohort", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, out)


def discrimination(report: dict, out: Path) -> None:
    names = {"cox_ph": "Cox PH (deployed)", "random_survival_forest": "Random survival forest", "gradient_boosted_survival": "Gradient boosted survival", "npi_baseline": "Nottingham Prognostic Index"}
    internal = {k: report["core_cross_validation"][k]["c_index_harrell_mean"] for k in names}
    sd = {k: report["core_cross_validation"][k].get("c_index_harrell_sd", 0) for k in names}
    external = {k: report["external_validation"][k]["c_index_harrell"] for k in names}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, data, title, err in ((axes[0], internal, "METABRIC five fold cross validation", sd), (axes[1], external, "GBSG2 external validation", None)):
        keys = list(names)
        y = np.arange(len(keys))[::-1]
        vals = [data[k] for k in keys]
        colours = [MUTED_BAR if k == "npi_baseline" else CATEGORICAL[0] if k == "cox_ph" else BLUE_RAMP["250"] for k in keys]
        ax.barh(y, [v - 0.5 for v in vals], left=0.5, color=colours, height=0.6, edgecolor=SURFACE, linewidth=2, xerr=[err[k] for k in keys] if err else None, error_kw={"ecolor": INK_2, "elinewidth": 1})
        for yi, v in zip(y, vals, strict=False):
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [names[k] for k in keys])
        ax.set_xlim(0.5, 0.74)
        ax.set_xlabel("Harrell C index (0.5 is chance)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Discrimination for relapse free survival versus the NPI baseline", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, out)


def hazard_ratios(report: dict, out: Path) -> None:
    rows = sorted(report["cox_hazard_ratios"], key=lambda r: r["hr_per_sd"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(rows))
    hr = [r["hr_per_sd"] for r in rows]
    ax.axvline(1.0, color=INK_2, linewidth=1)
    for yi, v in zip(y, hr, strict=False):
        ax.plot([1.0, v], [yi, yi], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
    ax.scatter(hr, y, s=70, color=[CATEGORICAL[1] if v > 1 else CATEGORICAL[0] for v in hr], edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, v in zip(y, hr, strict=False):
        ax.text(v + (0.012 if v >= 1 else -0.012), yi + 0.28, f"{v:.2f}", ha="left" if v >= 1 else "right", fontsize=8.5, color=INK)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xscale("log")
    ax.set_xticks([0.8, 0.9, 1.0, 1.25, 1.5, 1.75], ["0.8", "0.9", "1.0", "1.25", "1.5", "1.75"])
    ax.set_xlabel("Hazard ratio per standard deviation (log scale)")
    ax.set_title("Cox model hazard ratios (orange raises risk, blue lowers it)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    _save(fig, out)


def subtype_confusion(report: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    panels = [
        (report["expression_pam50"]["logistic_regression_cv"], "PAM50 expression model"),
        (report["surrogate_pathology"]["logistic_regression_cv"], "Pathology surrogate model"),
    ]
    for ax, (res, title) in zip(axes, panels, strict=False):
        cm = np.array(res["confusion_matrix"]["matrix"], dtype=float)
        labels = res["confusion_matrix"]["labels"]
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", BLUE_RAMP["100"], BLUE_RAMP["250"], BLUE_RAMP["450"], BLUE_RAMP["650"]])
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference PAM50 label")
        ax.grid(False)
        ax.set_title(f"{title}\naccuracy {res['accuracy']:.2f}, macro F1 {res['macro_f1']:.2f}", loc="left", fontsize=10)
    fig.suptitle("Intrinsic subtype: expression recovers the label, routine pathology often cannot", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, out)


def subtype_survival(cohort: dict, out: Path) -> None:
    curves = cohort["metabric"]["rfs_by_subtype"]["curves"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for i, (name, km) in enumerate(curves.items()):
        t = np.array(km["time"])
        m = t <= 240
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        ax.step(t[m], np.array(km["survival"])[m], where="post", color=colour, linewidth=2, label=f"{name} (n={km['n']})")
        ax.text(t[m][-1] + 3, np.array(km["survival"])[m][-1], name, color=INK_2, fontsize=8.5, va="center")
    p = cohort["metabric"]["rfs_by_subtype"]["logrank"]["p_value"]
    ax.set_xlim(0, 268)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months from diagnosis")
    ax.set_ylabel("Relapse free survival")
    ax.set_title(f"METABRIC relapse free survival by intrinsic subtype (log rank p {'< 1e-6' if p < 1e-6 else f'= {p:.2g}'})", loc="left", fontsize=11)
    ax.legend(loc="lower left", ncol=2)
    _save(fig, out)


def render_all(reports: Path, figures: Path) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    made = []

    def load(name):
        p = reports / name
        return json.loads(p.read_text()) if p.exists() else None

    if r := load("retrieval_eval.json"):
        retrieval_ablation(r, figures / "retrieval_ablation.png")
        made.append("retrieval_ablation.png")
    if s := load("survival_eval.json"):
        survival_risk_groups(s, figures / "survival_risk_groups.png")
        calibration(s, figures / "calibration.png")
        discrimination(s, figures / "discrimination.png")
        hazard_ratios(s, figures / "hazard_ratios.png")
        made += ["survival_risk_groups.png", "calibration.png", "discrimination.png", "hazard_ratios.png"]
    if t := load("subtype_eval.json"):
        subtype_confusion(t, figures / "subtype_confusion.png")
        made.append("subtype_confusion.png")
    cohort_path = reports.parent / "data" / "models" / "cohort_analytics.json"
    if cohort_path.exists():
        subtype_survival(json.loads(cohort_path.read_text()), figures / "subtype_survival.png")
        made.append("subtype_survival.png")
    return made
