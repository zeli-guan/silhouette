#!/usr/bin/env python3
"""Generate publication-grade vector PDFs (fig3..fig6) and 200 DPI PNG previews from artifact_release/stats/."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette.jsonl_io import iter_jsonl
from stats.reproduce_stats import paired_bootstrap, vec

# Publication typography & Okabe-Ito colorblind-safe palette
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 10.6,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6C757D"
DARK = "#222222"


def load_sources() -> dict:
    outcomes_bm = defaultdict(lambda: defaultdict(dict))
    dbs = {}
    for r in iter_jsonl(ROOT / "stats/per_question_outcomes.jsonl"):
        outcomes_bm[r["benchmark"]][r["condition"]][r["question_id"]] = r["outcome"]
        dbs[r["question_id"]] = r["db_id"]

    toks_by_bm_cond = defaultdict(list)
    for r in iter_jsonl(ROOT / "stats/sql_and_audit_receipts.jsonl.gz"):
        if r.get("total_tokens") is not None:
            toks_by_bm_cond[(r["benchmark"], r["condition"])].append(r["total_tokens"])

    bird_440 = json.loads((ROOT / "manifests/bird_minidev_2026_manifest.json").read_text(encoding="utf-8"))["evaluation"]
    ehr_600 = [x["ehrsql_id"] for x in json.loads((ROOT / "manifests/ehrsql_2024_manifest.json").read_text(encoding="utf-8"))["confirmation"]]
    strata = json.loads((ROOT / "manifests/question_strata_and_remap.json").read_text(encoding="utf-8"))
    scaling = json.loads((ROOT / "stats/stage6_scaling_and_survival.json").read_text(encoding="utf-8"))
    fig_tel = json.loads((ROOT / "stats/fig_telemetry_v6.json").read_text(encoding="utf-8"))
    clinical_survival = json.loads((ROOT / "stats/clinical_survival_3stage_verified.json").read_text(encoding="utf-8"))
    return {
        "outcomes_bm": outcomes_bm,
        "dbs": dbs,
        "toks": toks_by_bm_cond,
        "bird_440": bird_440,
        "ehr_600": ehr_600,
        "strata": strata,
        "scaling": scaling,
        "fig_tel": fig_tel,
        "clinical_survival": clinical_survival,
    }


def _save_fig(fig: plt.Figure, pdf_path: Path, png_path: Path | None, dpi: int = 150) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_fig3_frontier(src: dict, pdf_dir: Path, png_dir: Path | None, dpi: int = 150) -> None:
    outcomes_bm = src["outcomes_bm"]
    toks = src["toks"]
    bird_440 = src["bird_440"]
    ehr_600 = src["ehr_600"]

    def bm_ex(bm: str, qids: list, conds: list[str] | str) -> float:
        return 100.0 * mean(vec(outcomes_bm[bm], conds, qids))

    def bm_tok(bm: str, conds: list[str]) -> float:
        vals = []
        for c in conds:
            ts = toks[(bm, c)]
            if not ts:
                raise KeyError(f"Missing total_tokens receipts for ({bm}, {c})")
            vals.append(mean(ts))
        return mean(vals)

    # Colors & markers by paradigm
    COL_SILH = "#005AB5"
    COL_SILH_HALO = "#56B4E9"
    COL_SILH_TXT = "#004494"
    COL_1D = "#009E73"
    COL_1D_TXT = "#005A40"
    COL_IDP = "#7B3294"
    COL_IDP_TXT = "#511D66"
    COL_SYNTH = "#5C6B7A"
    COL_SYNTH_TXT = "#334155"
    COL_RAW = "#E69F00"
    COL_RAW_TXT = "#8A5300"
    COL_SCHEMA_LINE = "#888888"
    COL_FULL_LINE = "#334155"

    # Reference bands (horizontal dashed lines, NOT scatter points)
    pa_schema_y = bm_ex("bird", bird_440, ["C0/r0", "C0/r2"])
    pa_full_y = bm_ex("bird", bird_440, ["C4/r0", "C4/r2"])
    pb_schema_y = bm_ex("ehrsql_600", ehr_600, "C0/r0")
    pb_full_y = bm_ex("ehrsql_600", ehr_600, "C4/r0")
    pc_schema_y = bm_ex("bird", bird_440, "SOL_C0/r0")
    pc_full_y = bm_ex("bird", bird_440, "SOL_C4/r0")

    # Clean Table 2 & Table 7 representative points ONLY, with exact 95% CIs
    # tuple: (label, x, y, lo, hi, kind, anchor_y_mode, (dx, dy), ha, va)
    pa_2way_x = bm_tok("bird", ["SYN8/r0"])
    pa_2way_y = bm_ex("bird", bird_440, "SYN8/r0")
    pa_1way_x = bm_tok("bird", ["IND8/r0", "IND8_PROBE_R2/r0", "IND8_PROBE_R3/r0"])
    pa_1way_y = bm_ex("bird", bird_440, ["IND8/r0", "IND8_PROBE_R2/r0", "IND8_PROBE_R3/r0"])

    pa_points = [
        ("DP-Query", bm_tok("bird", ["DP_QUERY_ROW_TP8_V2_R1/r0", "DP_QUERY_ROW_TP8_V2_R2/r0", "DP_QUERY_ROW_TP8_V2_R3/r0"]),
         bm_ex("bird", bird_440, ["DP_QUERY_ROW_TP8_V2_R1/r0", "DP_QUERY_ROW_TP8_V2_R2/r0", "DP_QUERY_ROW_TP8_V2_R3/r0"]),
         29.4, 36.9, "idp", "pt", (-8, 0), "right", "center"),
        ("1-/2-way Synth†", pa_2way_x, pa_2way_y, 32.5, 41.6, "synth", (2620, 42.3), (0, 0), "left", "bottom"),
        ("", pa_1way_x, pa_1way_y, 33.3, 40.7, "synth", "pt", (0, 0), "center", "center"),
        ("FK Synth†", bm_tok("bird", ["SYN8_FK_DAG_R1/r0"]), bm_ex("bird", bird_440, "SYN8_FK_DAG_R1/r0"),
         33.2, 42.3, "synth", "pt", (10, -21), "left", "top"),
        ("Select 3", bm_tok("bird", ["S3/r0"]), bm_ex("bird", bird_440, "S3/r0"),
         35.7, 44.8, "raw", "pt", (9, -2), "left", "center"),
        ("DAWA+DP-SIPS", bm_tok("bird", ["DAWA_CLEAN_BIRD_R1/r0", "DAWA_CLEAN_BIRD_R2/r0", "DAWA_CLEAN_BIRD_R3/r0"]),
         bm_ex("bird", bird_440, ["DAWA_CLEAN_BIRD_R1/r0", "DAWA_CLEAN_BIRD_R2/r0", "DAWA_CLEAN_BIRD_R3/r0"]),
         41.7, 49.3, "u1d", "hi", (-4, 4), "right", "bottom"),
        ("1-way Profile", bm_tok("bird", ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"]),
         bm_ex("bird", bird_440, ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"]),
         42.1, 50.0, "u1d", "lo", (0, -5), "center", "top"),
        ("Silhouette", bm_tok("bird", ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]),
         bm_ex("bird", bird_440, ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]),
         43.0, 50.9, "silh", "hi", (0, 4), "center", "bottom"),
    ]

    pb_points = [
        ("2-way Synth†", bm_tok("ehrsql_600", ["SYN8P/r0"]), bm_ex("ehrsql_600", ehr_600, "SYN8P/r0"),
         41.0, 49.0, "synth", "pt", (-9, 0), "right", "center"),
        ("1-way Synth†", bm_tok("ehrsql_600", ["IND8P/r0"]), bm_ex("ehrsql_600", ehr_600, "IND8P/r0"),
         40.5, 48.5, "synth", "lo", (4, -5), "left", "top"),
        ("Select 3", bm_tok("ehrsql_600", ["S3/r0"]), bm_ex("ehrsql_600", ehr_600, "S3/r0"),
         42.5, 50.5, "raw", "pt", (10, -2), "left", "center"),
        ("DAWA+DP-SIPS†", bm_tok("ehrsql_600", ["DAWA_CLEAN_EHR600_R1/r0"]), bm_ex("ehrsql_600", ehr_600, "DAWA_CLEAN_EHR600_R1/r0"),
         47.0, 55.0, "u1d", "pt", (-10, 8), "right", "bottom"),
        ("1-way Profile†", bm_tok("ehrsql_600", ["EHR_IND8_V3_R1/r0"]), bm_ex("ehrsql_600", ehr_600, "EHR_IND8_V3_R1/r0"),
         45.3, 53.3, "u1d", "lo", (0, -5), "center", "top"),
        ("Silhouette", bm_tok("ehrsql_600", ["EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0"]),
         bm_ex("ehrsql_600", ehr_600, ["EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0"]),
         46.0, 53.2, "silh", "hi", (-2, 4), "center", "bottom"),
    ]

    pc_points = [
        ("Static Synth†", bm_tok("bird", ["SOL_PUBCAP_IND8_ROWS_R1/r0"]), bm_ex("bird", bird_440, "SOL_PUBCAP_IND8_ROWS_R1/r0"),
         73.6, 81.4, "synth", "pt", (-9, -10), "right", "top"),
        ("1-way Synth†", bm_tok("bird", ["SOL_IND8_PROBE_R1/r0"]), bm_ex("bird", bird_440, "SOL_IND8_PROBE_R1/r0"),
         74.8, 82.5, "synth", "lo", (0, -5), "center", "top"),
        ("1-way Profile†", bm_tok("bird", ["SOL_IND8_PROMPT/r0"]), bm_ex("bird", bird_440, "SOL_IND8_PROMPT/r0"),
         75.0, 82.7, "u1d", "pt", (10, -2), "left", "center"),
        ("Silhouette", bm_tok("bird", ["SOL_DG8_CLEAN_R1/r0", "SOL_DG8_CLEAN_R2/r0", "SOL_DG8_CLEAN_R3/r0"]),
         bm_ex("bird", bird_440, ["SOL_DG8_CLEAN_R1/r0", "SOL_DG8_CLEAN_R2/r0", "SOL_DG8_CLEAN_R3/r0"]),
         76.5, 83.6, "silh", "pt", (-15, 16), "right", "bottom"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.45))
    panels = [
        (axes[0], pa_points, pa_schema_y, (10550, pa_schema_y + 0.42, "right", "bottom"),
         pa_full_y, (2550, pa_full_y + 0.42, "left", "bottom"),
         "(a) BIRD Mini-Dev (n=440, glm-5.3)", "Mean total tokens / question", "Official Set Execution Accuracy (%)",
         (2400, 10850), (28.0, 53.8)),
        (axes[1], pb_points, pb_schema_y, (16200, pb_schema_y + 0.35, "right", "bottom"),
         pb_full_y, (3850, pb_full_y + 0.35, "left", "bottom"),
         "(b) EHRSQL Confirm. (n=600, glm-5.3)", "Mean total tokens / question", "Official Execution Accuracy (%)",
         (3600, 16500), (34.8, 56.8)),
        (axes[2], pc_points, pc_schema_y, (66300, pc_schema_y + 0.25, "right", "bottom"),
         pc_full_y, (66300, pc_full_y + 0.25, "right", "bottom"),
         "(c) BIRD Mini-Dev (n=440, gpt-6.1-sol cached)", "Mean total tokens / question (incl. cached)", "Official Set Execution Accuracy (%)",
         (46200, 66800), (71.2, 86.2)),
    ]

    for ax, pts, sch_y, sch_pos, ful_y, ful_pos, title, xlabel, ylabel, xlim, ylim in panels:
        ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
        ax.axhline(sch_y, color=COL_SCHEMA_LINE, linestyle="--", linewidth=1.15, zorder=1)
        ax.axhline(ful_y, color=COL_FULL_LINE, linestyle="--", linewidth=1.25, zorder=1)
        ax.text(sch_pos[0], sch_pos[1], f"Schema only ({sch_y:.1f}%)", ha=sch_pos[2], va=sch_pos[3],
                fontsize=8.6, color="#555555", zorder=3,
                bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.8))
        ax.text(ful_pos[0], ful_pos[1], f"Full access ({ful_y:.1f}%)", ha=ful_pos[2], va=ful_pos[3],
                fontsize=8.6, fontweight="bold", color=COL_FULL_LINE, zorder=3,
                bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.8))

        style_map = {
            "silh": (COL_SILH, COL_SILH_TXT),
            "u1d": (COL_1D, COL_1D_TXT),
            "idp": (COL_IDP, COL_IDP_TXT),
            "synth": (COL_SYNTH, COL_SYNTH_TXT),
            "raw": (COL_RAW, COL_RAW_TXT),
        }

        for label, x, y, lo, hi, kind, anchor_mode, (dx, dy), ha, va in pts:
            col, txt_col = style_map[kind]
            if kind == "silh":
                ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=COL_SILH,
                            elinewidth=1.35, capsize=2.4, alpha=0.58, zorder=4)
                ax.scatter([x], [y], s=520, marker="o", color=COL_SILH_HALO, alpha=0.22, edgecolor="none", zorder=6)
                ax.scatter([x], [y], s=230, marker="*", color=COL_SILH, edgecolor="white", linewidth=0.9, zorder=7)
            else:
                ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col,
                            elinewidth=1.1, capsize=2.2, alpha=0.40, zorder=2)
                mk_map = {"u1d": ("o", 64), "idp": ("D", 60), "synth": ("^", 68), "raw": ("s", 60)}
                mk, sz = mk_map[kind]
                ax.scatter([x], [y], s=sz, marker=mk, color=col, edgecolor="white", linewidth=0.8, zorder=5)

            if label:
                if isinstance(anchor_mode, tuple):
                    ax_x, ay = anchor_mode
                else:
                    ax_x = x
                    ay = hi if anchor_mode == "hi" else (lo if anchor_mode == "lo" else y)
                fs = 10.0 if kind == "silh" else 8.8
                fw = "bold" if kind == "silh" else "normal"
                ax.annotate(label, (ax_x, ay), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fs, fontweight=fw, color=txt_col, zorder=8)

        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if xlim[1] > 20000:
            ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:g}k")
        else:
            ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{int(v)}")

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", label="Silhouette (role-allocated 1D DP)",
               markerfacecolor=COL_SILH, markeredgecolor="white", markersize=13.5),
        Line2D([0], [0], marker="o", color="w", label="Uniform 1D DP profile",
               markerfacecolor=COL_1D, markeredgecolor="white", markersize=8.5),
        Line2D([0], [0], marker="D", color="w", label="Interactive DP",
               markerfacecolor=COL_IDP, markeredgecolor="white", markersize=7.5),
        Line2D([0], [0], marker="^", color="w", label="DP synthetic DB",
               markerfacecolor=COL_SYNTH, markeredgecolor="white", markersize=8.5),
        Line2D([0], [0], marker="s", color="w", label="Raw sample rows (Select 3)",
               markerfacecolor=COL_RAW, markeredgecolor="white", markersize=7.8),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#CCCCCC",
        fontsize=9.3,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_fig(fig, pdf_dir / "fig3_frontier.pdf", png_dir / "fig3_frontier.png" if png_dir else None, dpi=dpi)


def plot_fig4_budget(src: dict, pdf_dir: Path, png_dir: Path | None, dpi: int = 150) -> None:
    bird_out = src["outcomes_bm"]["bird"]
    bird_440 = src["bird_440"]
    scaling = src["scaling"]
    fig4b = scaling["fig4b_retention_m20"]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.35))

    # Panel a: R=1 Budget Sweep (eps=1,2,4,8 + Noiseless v3) vs Equal Public Cap (R=3) vs w/o Values
    ax = axes[0]
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)

    c0_pct = 100.0 * mean(vec(bird_out, ["C0/r0", "C0/r2"], bird_440))
    c4_pct = 100.0 * mean(vec(bird_out, ["C4/r0", "C4/r2"], bird_440))
    ax.axhline(c0_pct, color=GRAY, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(c4_pct, color=DARK, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(-0.42, c0_pct + 0.55, f"Schema only ({c0_pct:.1f}%)", ha="left", fontsize=8.5, color=GRAY,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.5))
    ax.annotate(f"Full access ({c4_pct:.1f}%)", (-0.25, c4_pct), xytext=(-0.35, 52.0), ha="left", va="bottom",
                fontsize=8.5, fontweight="bold", color=DARK, zorder=6,
                bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", pad=0.6),
                arrowprops=dict(arrowstyle="-", color=DARK, lw=0.75, shrinkA=0, shrinkB=2))

    eps_conds = [(1, "DG1_CLEAN_R1/r0"), (2, "DG2_CLEAN_R1/r0"), (4, "DG4_CLEAN_R1/r0"), (8, "DG8_CLEAN_R1/r0")]
    x_curve = [0, 1, 2, 3]
    y_curve, lo_c, hi_c = [], [], []
    for _eps, cond in eps_conds:
        v = vec(bird_out, cond, bird_440)
        lo, hi = paired_bootstrap(v)
        m = 100.0 * mean(v)
        y_curve.append(m)
        lo_c.append(m - 100.0 * lo)
        hi_c.append(100.0 * hi - m)
    ax.errorbar(x_curve, y_curve, yerr=[lo_c, hi_c], fmt="-o", color=BLUE, capsize=3, linewidth=1.8, markersize=5.2, zorder=4)

    # Noiseless v3 reference on R=1 sweep (x=4)
    v_nl = vec(bird_out, "BIRD_NOISELESS_V3/r0", bird_440)
    lo_nl, hi_nl = paired_bootstrap(v_nl)
    m_nl = 100.0 * mean(v_nl)
    ax.plot([3, 4], [y_curve[-1], m_nl], linestyle=":", color=SKY, linewidth=1.4, zorder=3)
    ax.errorbar([4], [m_nl], yerr=[[m_nl - 100.0 * lo_nl], [100.0 * hi_nl - m_nl]], fmt="D", color=SKY,
                markeredgecolor=DARK, markeredgewidth=0.6, capsize=3, markersize=5.5, zorder=5)
    ax.annotate(f"{m_nl:.1f}%", (4, 100.0 * hi_nl), xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=8.6, color=BLUE, fontweight="bold")

    ax.axvline(4.65, color="#BBBBBB", linestyle="-", linewidth=1.0, zorder=1)
    ax.text(1.95, 59.6, "R=1 Sweep", ha="center", fontsize=8.8, fontweight="bold", color=BLUE)
    ax.text(8.2, 59.6, "Equal Public Cap (R=3, 95% CI)", ha="center", fontsize=8.8, fontweight="bold", color=VERMILION)

    pubcap_specs = [
        ("Static Synth", ["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"], ORANGE),
        ("1-way Cap", ["PUBCAP_IND8_PROFILE_R1/r0", "PUBCAP_IND8_PROFILE_R2/r0", "PUBCAP_IND8_PROFILE_R3/r0"], GREEN),
        ("w/o Key prune", ["PUBCAP_UNPRUNED_R1/r0", "PUBCAP_UNPRUNED_R2/r0", "PUBCAP_UNPRUNED_R3/r0"], SKY),
        ("w/o Role wts", ["PUBCAP_OMEGA1_EQDELTA_R1/r0", "PUBCAP_OMEGA1_EQDELTA_R2/r0", "PUBCAP_OMEGA1_EQDELTA_R3/r0"], PURPLE),
        ("Flat wts", ["PUBCAP_FLATW_R1/r0", "PUBCAP_FLATW_R2/r0", "PUBCAP_FLATW_R3/r0"], VERMILION),
        ("Silh.-C", ["PUBCAP_SILHOUETTE_R1/r0", "PUBCAP_SILHOUETTE_R2/r0", "PUBCAP_SILHOUETTE_R3/r0"], BLUE),
    ]
    x_bars = [5.4, 6.4, 7.4, 8.4, 9.4, 10.4]
    for xb, (lbl, conds, col) in zip(x_bars, pubcap_specs):
        v = vec(bird_out, conds, bird_440)
        lo, hi = paired_bootstrap(v)
        m = 100.0 * mean(v)
        ax.bar([xb], [m], width=0.72, color=col, alpha=0.84 if col != BLUE else 0.96, edgecolor=DARK, linewidth=0.6, zorder=3)
        ax.errorbar([xb], [m], yerr=[[m - 100.0 * lo], [100.0 * hi - m]], fmt="none", ecolor=DARK, capsize=2.5, linewidth=0.9, zorder=4)
        if lbl == "Silh.-C":
            ax.annotate(f"{m:.1f}%", (xb, 100.0 * hi), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.6, color=BLUE, fontweight="bold")

    # Uncapped R=3 w/o Values comparison bar (x=11.7)
    ax.axvline(11.05, color="#CCCCCC", linestyle=":", linewidth=1.0, zorder=1)
    noval_conds = ["DG8_NOVAL_POST_R1/r0", "DG8_NOVAL_POST_R2/r0", "DG8_NOVAL_POST_R3/r0"]
    v_nv = vec(bird_out, noval_conds, bird_440)
    lo_nv, hi_nv = paired_bootstrap(v_nv)
    m_nv = 100.0 * mean(v_nv)
    ax.bar([11.7], [m_nv], width=0.72, color="#7F8C8D", alpha=0.90, edgecolor=DARK, linewidth=0.6, hatch="//", zorder=3)
    ax.errorbar([11.7], [m_nv], yerr=[[m_nv - 100.0 * lo_nv], [100.0 * hi_nv - m_nv]], fmt="none", ecolor=DARK, capsize=2.5, linewidth=0.9, zorder=4)
    ax.annotate(f"{m_nv:.1f}%", (11.7, 100.0 * hi_nv), xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=8.5, color=DARK, fontweight="bold")
    ax.text(11.7, 52.8, "Uncapped\nw/o Values", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#555555")

    ax.set_xticks(x_curve + [4] + x_bars + [11.7])
    ax.set_xticklabels(
        ["$\\varepsilon_T=1$", "$\\varepsilon_T=2$", "$\\varepsilon_T=4$", "$\\varepsilon_T=8$", "Noiseless"] + [s[0] for s in pubcap_specs] + ["w/o Values"],
        fontsize=8.1, rotation=32, ha="right",
    )
    ax.set_xlim(-0.55, 12.45)
    ax.set_ylim(18.5, 62.5)
    ax.set_ylabel("BIRD Official Set EX (%)")
    ax.set_title("(a) Budget Sweep (R=1), Cap_pub & w/o Values")
    leg_a = [
        Line2D([0], [0], color=BLUE, marker="o", lw=1.8, markersize=5, label="R=1 curve ($\\varepsilon_T=1..8$)"),
        Line2D([0], [0], color=SKY, marker="D", lw=1.2, ls=":", markeredgecolor=DARK, markersize=5, label="Noiseless ($R=1$)"),
        Patch(facecolor=BLUE, edgecolor=DARK, alpha=0.92, label="Equal Cap_pub (R=3, 95% CI)"),
        Patch(facecolor="#7F8C8D", edgecolor=DARK, hatch="//", alpha=0.9, label="Uncapped w/o Values (40.5%)"),
    ]
    ax.legend(handles=leg_a, loc="lower left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.5)

    # Panel b: M=20 retention over eps in {1, 2, 4, 8}
    ax = axes[1]
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
    eps_vals = [1, 2, 4, 8]
    for mode, label_mode, col, mk, ls in [
        ("DG-v3", "Silhouette", BLUE, "o", "-"),
        ("DG-FlatW", "Flat weights", ORANGE, "s", "--"),
    ]:
        means = [fig4b["by_epsilon"][str(e)][mode]["mean_pct"] for e in eps_vals]
        sds = [fig4b["by_epsilon"][str(e)][mode]["sd_pp"] for e in eps_vals]
        ax.errorbar(eps_vals, means, yerr=sds, fmt=f"{ls}{mk}", color=col, capsize=3.5, linewidth=1.8, markersize=5.5, label=f"{label_mode} (mean $\\pm$ SD, M=20)")
        for e, m in zip(eps_vals, means):
            ax.annotate(f"{m:.1f}%", (e, m), xytext=(0, 10 if mode == "DG-v3" else -15), textcoords="offset points", ha="center", fontsize=9.0, color=col, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_vals)
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.8, 10.5)
    ax.set_ylim(26, 74)
    ax.set_xlabel("Privacy budget $\\varepsilon_T$ per table")
    ax.set_ylabel("Target Frequent-Value Retention (%)")
    ax.set_title(f"(b) Frequent-Value Retention (n={fig4b['n_target_c1_values']:,} targets)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.1)

    # Panel c: Table-size bucket retention (75 BIRD tables, M=20, eps=8)
    ax = axes[2]
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
    bkt = scaling["fig4c_table_size_buckets"]["by_mode_and_table_size_bin"]
    bins = ["<1k", "1k-100k", ">100k"]
    x_idx = np.arange(len(bins))
    w = 0.35
    dg_m = [bkt["DG-v3"][b]["retention_mean_pct"] for b in bins]
    dg_sd = [bkt["DG-v3"][b]["retention_sd_pp"] for b in bins]
    fw_m = [bkt["DG-FlatW"][b]["retention_mean_pct"] for b in bins]
    fw_sd = [bkt["DG-FlatW"][b]["retention_sd_pp"] for b in bins]
    ax.bar(x_idx - w / 2, dg_m, width=w, yerr=dg_sd, capsize=3, color=BLUE, edgecolor=DARK, linewidth=0.6, label="Silhouette", zorder=3)
    ax.bar(x_idx + w / 2, fw_m, width=w, yerr=fw_sd, capsize=3, color=ORANGE, edgecolor=DARK, linewidth=0.6, label="Flat weights", zorder=3)
    for i in range(len(bins)):
        ax.text(x_idx[i] - w / 2 - 0.03, dg_m[i] + dg_sd[i] + 1.5, f"{dg_m[i]:.1f}%", ha="center", va="bottom", fontsize=8.6, color=BLUE, fontweight="bold")
        ax.text(x_idx[i] + w / 2 + 0.03, fw_m[i] + fw_sd[i] + 1.5, f"{fw_m[i]:.1f}%", ha="center", va="bottom", fontsize=8.6, color="#A86800", fontweight="bold")
    x_labels = [f"{b}\n(n={bkt['DG-v3'][b]['n_tables']} tbls)" for b in bins]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, fontsize=9.6)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Target Frequent-Value Retention (%)")
    ax.set_title("(c) Retention by Table Size ($\\varepsilon_T=8$, 75 tables)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.3)

    fig.tight_layout()
    _save_fig(fig, pdf_dir / "fig4_budget.pdf", png_dir / "fig4_budget.png" if png_dir else None, dpi=dpi)


def plot_fig5_scaling(src: dict, pdf_dir: Path, png_dir: Path | None, dpi: int = 150) -> None:
    sc = src["scaling"]
    ft = src["fig_tel"]
    raw_row = sc["bird_raw_scaling_165"]
    cov_row = sc["fig5b_scaling_rows_cov_165"]
    col_p = ft["fig5_columns_prefix"]

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.2))

    fracs = [0.01, 0.03, 0.1, 0.3, 1.0]
    fkey = ["0.01", "0.03", "0.1", "0.3", "1.0"]

    # Panel a: 165-run BIRD row-scaling median tokens with IQR and P95
    ax = axes[0]
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
    p50 = [raw_row[k]["tokens"]["p50"] for k in fkey]
    q1 = [raw_row[k]["tokens"]["iqr"][0] for k in fkey]
    q3 = [raw_row[k]["tokens"]["iqr"][1] for k in fkey]
    p95 = [raw_row[k]["tokens"]["p95"] for k in fkey]
    ax.fill_between(fracs, q1, q3, color=BLUE, alpha=0.22, label="IQR [Q1, Q3]")
    ax.plot(fracs, p95, linestyle="--", color=SKY, linewidth=1.3, label="P95")
    ax.plot(fracs, p50, "-o", color=BLUE, linewidth=1.8, markersize=4.8, label="Median (P50)")
    ax.annotate(f"{int(p50[0]):,}", (fracs[0], p50[0]), xytext=(4, 7), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.annotate(f"{int(p50[-1]):,}", (fracs[-1], p50[-1]), xytext=(-24, 7), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.set_ylim(0, 10200)
    ax.set_xlabel("Row fraction")
    ax.set_ylabel("DB profile tokens")
    ax.set_title("(a) Profile Tokens vs. Rows")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.7)

    # Panel b: Literal coverage vs row fraction (from stage6_scaling_and_survival.json)
    ax = axes[1]
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
    cmed = [cov_row[k]["cov_median_pct"] for k in fkey]
    cmin = [cov_row[k]["cov_min_pct"] for k in fkey]
    cmax = [cov_row[k]["cov_max_pct"] for k in fkey]
    base_cov = sc["fig5_baseline_cov_pct"]
    ax.axhline(base_cov, color=GRAY, linestyle="--", linewidth=1.1, label=f"Schema only ({base_cov:.1f}%)")
    ax.fill_between(fracs, cmin, cmax, color=BLUE, alpha=0.20)
    ax.plot(fracs, cmed, "-o", color=BLUE, linewidth=1.8, markersize=4.8, label="Silhouette ($\\varepsilon_T=8$)")
    ax.annotate(f"{cmed[0]:.1f}%", (fracs[0], cmed[0]), xytext=(0, 9), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.annotate(f"{cmed[-1]:.1f}%", (fracs[-1], cmed[-1]), xytext=(-24, 6), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.set_ylim(56, 91)
    ax.set_xlabel("Row fraction")
    ax.set_ylabel("Literal coverage (%)")
    ax.set_title("(b) Literal Coverage vs. Rows")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.7)

    # Panel c: Column-prefix exact integer tau (4, 8, 16, 32, all)
    col_keys = ["4", "8", "16", "32", "all"]
    cx = np.arange(len(col_keys))
    ax = axes[2]
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
    tau_h = [col_p[k]["tau_hist_median"] for k in col_keys]
    tau_pool = [col_p[k].get("tau_all_stats_pooled_median", col_p[k]["tau_hist_median"]) for k in col_keys]
    tau_v = [col_p[k]["tau_median"] for k in col_keys]
    tau_q1 = [col_p[k]["tau_q1"] for k in col_keys]
    tau_q3 = [col_p[k]["tau_q3"] for k in col_keys]
    ax.plot(cx, tau_h, "-o", color=BLUE, linewidth=1.8, markersize=4.8, label="Table-median $\\tau$ (int)")
    ax.plot(cx, tau_pool, ":^", color=SKY, linewidth=1.4, markersize=4.2, label="Pooled-hist. $\\tau$ (int)")
    ax.fill_between(cx, tau_q1, tau_q3, color=ORANGE, alpha=0.18)
    ax.plot(cx, tau_v, "--s", color=ORANGE, linewidth=1.4, markersize=4.2, label="Val. $\\tau$ (median + IQR)")
    ax.annotate(f"{int(tau_h[0])}", (cx[0], tau_h[0]), xytext=(2, 9), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.annotate(f"{int(tau_h[-1])}", (cx[-1], tau_h[-1]), xytext=(-18, -14), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.annotate(f"{int(tau_pool[-1])}", (cx[-1], tau_pool[-1]), xytext=(-18, 5), textcoords="offset points", fontsize=8.9, color=BLUE)
    ax.set_xticks(cx)
    ax.set_xticklabels(["4", "8", "16", "32", "all\n(115)"])
    ax.set_ylim(5, 58)
    ax.set_xlabel("Column prefix")
    ax.set_ylabel("Median integer threshold $\\tau$")
    ax.set_title("(c) Exact Integer $\\tau$ vs. Width")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.3)

    # Panel d: Column-prefix table tokens
    ax = axes[3]
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
    tmed = [col_p[k]["tokens_median"] for k in col_keys]
    tq1 = [col_p[k]["tokens_q1"] for k in col_keys]
    tq3 = [col_p[k]["tokens_q3"] for k in col_keys]
    ax.fill_between(cx, tq1, tq3, color=BLUE, alpha=0.22, label="IQR [Q1, Q3]")
    ax.plot(cx, tmed, "-o", color=BLUE, linewidth=1.8, markersize=4.8, label="Median tokens")
    ax.annotate(f"{tmed[0]}", (cx[0], tmed[0]), xytext=(4, 7), textcoords="offset points", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.annotate(f"{tmed[-1]:,}", (cx[-1], tmed[-1]), xytext=(-12, -17), textcoords="offset points", ha="center", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.set_xticks(cx)
    ax.set_xticklabels(["4", "8", "16", "32", "all\n(115)"])
    ax.set_ylim(0, 1850)
    ax.set_xlabel("Column prefix")
    ax.set_ylabel("Table profile tokens")
    ax.set_title("(d) Table Tokens vs. Width")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.7)

    fig.tight_layout()
    _save_fig(fig, pdf_dir / "fig5_scaling.pdf", png_dir / "fig5_scaling.png" if png_dir else None, dpi=dpi)


def plot_fig6_interactive(src: dict, pdf_dir: Path, png_dir: Path | None, dpi: int = 150) -> None:
    bird_out = src["outcomes_bm"]["bird"]
    bird_440 = src["bird_440"]
    strata = src["strata"]
    surv = src["clinical_survival"]["schedules"]

    fixed_55 = strata["fixed_55_qids"]
    q15_163 = strata["dp_query_subsets"]["q15_163"]
    dg8_clean = ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]
    tp8_v2 = ["DP_QUERY_ROW_TP8_V2_R1/r0", "DP_QUERY_ROW_TP8_V2_R2/r0", "DP_QUERY_ROW_TP8_V2_R3/r0"]
    c0_conds = ["C0/r0", "C0/r2"]
    ind8_conds = ["IND8/r0", "IND8_PROBE_R2/r0", "IND8_PROBE_R3/r0"]

    fig = plt.figure(figsize=(13.8, 4.45))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.48, wspace=0.29)

    # Panel a1 & a2: Fixed-55 Horizon Sweep and Nested Stream Prefixes
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a1.grid(True, linestyle=":", alpha=0.35, zorder=0)
    qps = [5, 15, 50, 200]
    f55_means = [100.0 * mean(vec(bird_out, [f"DPQ_FIXED55_QP{q}_R{r}/r0" for r in (1, 2, 3)], fixed_55)) for q in qps]
    silh_55 = 100.0 * mean(vec(bird_out, dg8_clean, fixed_55))
    ax_a1.axhline(silh_55, color=BLUE, linestyle="--", linewidth=1.3, label=f"Silhouette ({silh_55:.1f}%)")
    ax_a1.plot(range(len(qps)), f55_means, "-s", color=VERMILION, linewidth=1.6, markersize=5, label="DP-Query")
    for i, m in enumerate(f55_means):
        ax_a1.annotate(f"{m:.1f}%", (i, m), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.9, color=VERMILION, fontweight="bold")
    ax_a1.set_xticks(range(len(qps)))
    ax_a1.set_xticklabels([f"$Q_p$={q}" for q in qps], fontsize=9.3)
    ax_a1.set_ylim(45, 70)
    ax_a1.set_ylabel("Set EX (%)")
    ax_a1.set_title("(a1) Fixed-55 Horizon Sweep (n=55)", fontsize=10.9)
    ax_a1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.6)

    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_a2.grid(True, linestyle=":", alpha=0.35, zorder=0)
    prefixes = [("n=55", fixed_55), ("n=163", q15_163), ("n=440", bird_440)]
    silh_p = [100.0 * mean(vec(bird_out, dg8_clean, qset)) for _, qset in prefixes]
    tp8_p = [100.0 * mean(vec(bird_out, tp8_v2, qset)) for _, qset in prefixes]
    px = np.arange(len(prefixes))
    ax_a2.plot(px, silh_p, "-o", color=BLUE, linewidth=1.8, markersize=5.2, label="Silhouette")
    ax_a2.plot(px, tp8_p, "-s", color=VERMILION, linewidth=1.8, markersize=5.2, label="DP-Query")
    for i in range(len(prefixes)):
        ax_a2.annotate(f"{silh_p[i]:.1f}%", (px[i], silh_p[i]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.9, color=BLUE, fontweight="bold")
        ax_a2.annotate(f"{tp8_p[i]:.1f}%", (px[i], tp8_p[i]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.9, color=VERMILION, fontweight="bold")
    ax_a2.set_xticks(px)
    ax_a2.set_xticklabels([p[0] for p in prefixes], fontsize=9.6)
    ax_a2.set_ylim(24, 72)
    ax_a2.set_ylabel("Set EX (%)")
    ax_a2.set_title("(a2) Nested Stream Prefixes", fontsize=10.9)
    ax_a2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.6)

    # Panel b (full height middle): Whole-Patient Bootstrap 3-stage Clinical Survival (M=200 cohorts, C=5)
    ax_b = fig.add_subplot(gs[:, 1])
    ax_b.grid(True, linestyle=":", alpha=0.35, zorder=0)
    cohorts = ["94", "1000", "10000", "100000"]
    cx_b = np.arange(len(cohorts))

    def extract_surv(sched_key: str, stage_key: str, denom: str = "all_125"):
        ys, lo, hi = [], [], []
        for c in cohorts:
            cell = surv[sched_key]["cohorts"][c][stage_key][denom]
            y = cell["mean_pct"]
            c_lo, c_hi = cell["ci95_pct"]
            assert c_lo <= y <= c_hi, f"Marker outside band for {sched_key} {stage_key} {denom} N={c}: {y} not in [{c_lo}, {c_hi}]"
            ys.append(y)
            lo.append(c_lo)
            hi.append(c_hi)
        return ys, lo, hi

    s1_y, s1_lo, s1_hi = extract_surv("db_global_eps8", "stage1_occupied_gt_tau", "all_125")
    s2_y, s2_lo, s2_hi = extract_surv("db_global_eps8", "stage2_dhat_le_50_gate", "all_125")
    s3_y, s3_lo, s3_hi = extract_surv("db_global_eps8", "stage3_eq10_top15_published", "all_125")
    s3_r, _, _ = extract_surv("db_global_eps8", "stage3_eq10_top15_published", "reachable_120")
    pt_y, pt_lo, pt_hi = extract_surv("per_table_eps8", "stage3_eq10_top15_published", "all_125")
    pt_r, _, _ = extract_surv("per_table_eps8", "stage3_eq10_top15_published", "reachable_120")

    ax_b.fill_between(cx_b, s1_lo, s1_hi, color=SKY, alpha=0.20)
    ax_b.plot(cx_b, s1_y, "--^", color=BLUE, linewidth=1.6, markersize=5, label="Stage-1 Threshold (db_global $\\varepsilon=8$)")
    ax_b.fill_between(cx_b, s2_lo, s2_hi, color=ORANGE, alpha=0.16)
    ax_b.plot(cx_b, s2_y, "-.d", color=ORANGE, linewidth=1.5, markersize=4.8, label="Stage-2 Card.$\\leq$50 (db_global $\\varepsilon=8$)")
    ax_b.fill_between(cx_b, s3_lo, s3_hi, color=VERMILION, alpha=0.18)
    ax_b.plot(cx_b, s3_y, "-o", color=VERMILION, linewidth=1.8, markersize=5.2, label="Stage-3 Published (db_global $\\varepsilon=8$)")
    ax_b.fill_between(cx_b, pt_lo, pt_hi, color=GREEN, alpha=0.16)
    ax_b.plot(cx_b, pt_y, "-s", color=GREEN, linewidth=1.5, markersize=4.8, label="Stage-3 Published (per_table $\\varepsilon_T=8$)")
    ax_b.plot(cx_b, pt_r, ":s", color=GREEN, linewidth=1.1, markersize=3.5, alpha=0.85, label="Reachable 120 (per_table $\\varepsilon_T=8$)")

    ax_b.annotate(f"{s3_y[0]:.1f}% ({s3_r[0]:.1f}% reach.)", (cx_b[0], s3_y[0]), xytext=(6, -13), textcoords="offset points", ha="left", fontsize=8.3, color=VERMILION, fontweight="bold")
    ax_b.annotate(f"{pt_y[0]:.1f}%", (cx_b[0], pt_y[0]), xytext=(2, 11), textcoords="offset points", ha="center", fontsize=8.5, color=GREEN, fontweight="bold")
    ax_b.annotate(f"{s1_y[-1]:.1f}%", (cx_b[-1], s1_y[-1]), xytext=(-10, 5), textcoords="offset points", ha="right", fontsize=8.7, color=BLUE, fontweight="bold")
    ax_b.annotate(f"{s2_y[-1]:.1f}%", (cx_b[-1], s2_y[-1]), xytext=(-10, 4), textcoords="offset points", ha="right", fontsize=8.6, color="#A86800", fontweight="bold")
    ax_b.annotate(f"{s3_y[-1]:.1f}% ({s3_r[-1]:.1f}% reach.)", (cx_b[-1], s3_y[-1]), xytext=(-8, -14), textcoords="offset points", ha="right", fontsize=8.4, color=VERMILION, fontweight="bold")
    ax_b.annotate(f"{pt_r[-1]:.1f}% reach. ({pt_y[-1]:.1f}%)", (cx_b[-1], pt_y[-1]), xytext=(-8, -12), textcoords="offset points", ha="right", fontsize=8.4, color=GREEN, fontweight="bold")

    ax_b.set_xticks(cx_b)
    ax_b.set_xticklabels(["94\n(demo)", "1k", "10k", "100k"])
    ax_b.set_xlim(-0.28, 3.25)
    ax_b.set_ylim(-6, 130)
    ax_b.set_xlabel("Clinical Cohort Size $N$ (patients)")
    ax_b.set_ylabel("Projected Target-Value Survival (%)")
    ax_b.set_title("(b) Clinical Survival (Whole-Patient M=200, C=5)", fontsize=10.6)
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    handles_b.append(Patch(facecolor=GRAY, alpha=0.22, edgecolor="none", label="Shaded bands: 2.5–97.5th trial percentiles"))
    ax_b.legend(handles=handles_b, loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.6)

    # Panel c (full height right): Gold Hidden Literals & Parsed Base Tables (with R=3 1-way Synth)
    ax_c = fig.add_subplot(gs[:, 2])
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
    hidden_106 = strata["gold_hidden_literal_106"]["hidden_106"]
    no_hidden_334 = strata["gold_hidden_literal_106"]["no_hidden_334"]
    pb = strata["parsed_base_tables"]
    strata_groups = [
        ("Hidden\nLit.\n(n=106)", hidden_106),
        ("No-Hidden\nLit.\n(n=334)", no_hidden_334),
        ("1 Base\nTable\n(n=50)", pb["1_base_table"]),
        ("2 Base\nTables\n(n=242)", pb["2_base_tables"]),
        ("3+ Base\nTables\n(n=148)", pb["3+_base_tables"]),
    ]
    gx = np.arange(len(strata_groups))
    bw = 0.25
    c0_vals = [100.0 * mean(vec(bird_out, c0_conds, qids)) for _, qids in strata_groups]
    ind_vals = [100.0 * mean(vec(bird_out, ind8_conds, qids)) for _, qids in strata_groups]
    sil_vals = [100.0 * mean(vec(bird_out, dg8_clean, qids)) for _, qids in strata_groups]

    ax_c.bar(gx - bw, c0_vals, width=bw, color=GRAY, edgecolor=DARK, linewidth=0.5, label="Schema only", zorder=3)
    ax_c.bar(gx, ind_vals, width=bw, color=ORANGE, edgecolor=DARK, linewidth=0.5, label="1-way Synth", zorder=3)
    ax_c.bar(gx + bw, sil_vals, width=bw, color=BLUE, edgecolor=DARK, linewidth=0.5, label="Silhouette", zorder=3)
    for i in range(len(strata_groups)):
        ax_c.text(gx[i] + bw, sil_vals[i] + 1.0, f"{sil_vals[i]:.1f}", ha="center", va="bottom", fontsize=8.6, color=BLUE, fontweight="bold")
    ax_c.axvline(1.5, color="#BBBBBB", linestyle="-", linewidth=1.0)
    ax_c.set_xticks(gx)
    ax_c.set_xticklabels([g[0] for g in strata_groups], fontsize=8.9)
    ax_c.set_ylim(0, 62)
    ax_c.set_ylabel("BIRD Official Set EX (%)")
    ax_c.set_title("(c) Gold Literal & Parsed Table Strata")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.7)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.14)
    _save_fig(fig, pdf_dir / "fig6_interactive.pdf", png_dir / "fig6_interactive.png" if png_dir else None, dpi=dpi)


def main(argv: list[str] | None = None) -> int:
    default_pdf_dir = ROOT.parent / "vldb_2027" / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir, help="Destination directory for vector PDFs.")
    parser.add_argument("--png-dir", type=Path, default=None, help="Optional directory for PNG previews.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG preview DPI (default: 150).")
    args = parser.parse_args(argv)

    src = load_sources()
    plot_fig3_frontier(src, args.pdf_dir, args.png_dir, dpi=args.dpi)
    plot_fig4_budget(src, args.pdf_dir, args.png_dir, dpi=args.dpi)
    plot_fig5_scaling(src, args.pdf_dir, args.png_dir, dpi=args.dpi)
    plot_fig6_interactive(src, args.pdf_dir, args.png_dir, dpi=args.dpi)
    print(f"Generated fig3..fig6 vector PDFs in {args.pdf_dir}")
    if args.png_dir is not None:
        print(f"Generated {args.dpi} DPI PNG previews in {args.png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
