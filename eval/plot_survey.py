#!/usr/bin/env python3
"""Thesis figures and tables for the listening study (survey v5).

    ~/.conda/envs/music_EBT/bin/python eval/plot_survey.py \
        --analysis ~/orcd/scratch/rebcecca/music_EBT_logs/survey/analysis \
        --out eth-mit-master-thesis/figures/survey \
        --tables eth-mit-master-thesis/tables/survey

Reads main.json / strict.json / by_training.json written by
demo/survey/analyze_survey.py and demo/survey/analyze_by_training.py.
Figures: survey_part1.pdf (unguided best-worst), survey_part2.pdf (steering
heard in the requested direction), survey_caco.pdf (cacophony ticks,
unguided vs steered). Tables (LaTeX, booktabs): part2 cells, part3,
robustness, training groups.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Same ink as eval/plot_corpus_attributes.py; EBT takes categorical slot 1,
# the AR baselines share one neutral, the human original is ink.
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
EBT_C, AR_C = "#2a78d6", "#8d8c85"

NAMES = {"original": "Human original", "ebt": "EBT", "llama": "Llama", "gpt2": "GPT-2",
         "llama_pplm": "Llama PPLM", "llama_tilt": "Llama tilting", "llama_bon": "Llama best-of-$N$",
         "gpt2_tilt": "GPT-2 tilting", "gpt2_bon": "GPT-2 best-of-$N$"}
TEX_NAMES = {k: v for k, v in NAMES.items()}
TOK = {"remi": "REMI", "ant": "Anticipation"}
ATTR = {"velocity": "Velocity", "duration": "Duration", "pitch_register": "Pitch register"}
ORDER = ["ebt", "llama_pplm", "llama_tilt", "llama_bon", "gpt2_tilt", "gpt2_bon"]


def color(system):
    return INK if system == "original" else EBT_C if system == "ebt" else AR_C


def style():
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})


def dot_ci(ax, y, m, lo, hi, c, filled=True, size=4.5):
    ax.plot([lo, hi], [y, y], color=c, linewidth=1.4, solid_capstyle="round", zorder=2)
    ax.plot([m], [y], "o", color=c, markersize=size, zorder=3,
            markerfacecolor=c if filled else "white", markeredgewidth=1.2)


def fig_part1(main, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.75), sharex=True, constrained_layout=True)
    for ax, tok in zip(axes, ["remi", "ant"]):
        rows = sorted([r for r in main["part1"] if r["group"] == tok], key=lambda r: r["score"])
        for y, r in enumerate(rows):
            dot_ci(ax, y, r["score"], r["lo"], r["hi"], color(r["system"]))
            ax.text(1.0, y, f"{r['score']:+.2f}", ha="right", va="center", color=MUTED, fontsize=7)
        ax.set_yticks(range(len(rows)), [NAMES[r["system"]] for r in rows])
        ax.axvline(0, color=GRID, linewidth=0.8, zorder=1)
        ax.set_xlim(-0.6, 1.0)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(TOK[tok], color=INK, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("best–worst score")
    fig.savefig(out / "survey_part1.pdf")
    plt.close(fig)


def fig_part2(main, out):
    cells = {tuple(r["group"]): r for r in main["part2"] if r["grouping"] == "tok x attribute x system"}
    fam = {tuple(r["group"]): r for r in main["part2"] if r["grouping"] == "tok x family"}
    panels = [("remi", "velocity"), ("remi", "duration"), ("remi", "pitch_register"),
              ("ant", "duration"), ("ant", "pitch_register"), None]
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.6), sharex=True, constrained_layout=True)
    for ax, p in zip(axes.flat, panels):
        if p is None:   # pooled panel: EBT vs AR per tokenizer
            rows = [(f"{TOK[t]} {f}", fam[(t, f)], EBT_C if f == "EBT" else AR_C)
                    for t in ("remi", "ant") for f in ("EBT", "AR")]
            title = "Pooled over attributes"
        else:
            tok, attr = p
            rows = [(NAMES[s], cells[(tok, attr, s)], color(s)) for s in ORDER if (tok, attr, s) in cells]
            title = f"{TOK[tok]} · {ATTR[attr]}"
        rows = rows[::-1]
        for y, (label, r, c) in enumerate(rows):
            dot_ci(ax, y, r["right"], r["right_lo"], r["right_hi"], c)
            ax.plot([r["wrong"]], [y], "x", color=c, markersize=4, markeredgewidth=1.0, zorder=3)
        ax.set_yticks(range(len(rows)), [lab.replace("Llama ", "L. ").replace("GPT-2 ", "G. ")
                                         if p is not None else lab.replace(" AR", " AR baselines")
                                         for lab, _, _ in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.grid(axis="x", which="both", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=7)
        ax.set_title(title, color=INK, fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("share of answers")
    h = [plt.Line2D([], [], marker="o", color=MUTED, linestyle="-", label="heard in the requested direction (95% CI)"),
         plt.Line2D([], [], marker="x", color=MUTED, linestyle="none", label="heard in the opposite direction")]
    fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.savefig(out / "survey_part2.pdf")
    plt.close(fig)


def fig_caco(main, out):
    """Part 2 pairs only: the same trials give the reference and the steered tick rate."""
    pairs = {(r["tok"], r["system"]): r for r in main["caco_part2"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharex=True, constrained_layout=True)
    for ax, tok in zip(axes, ["remi", "ant"]):
        systems = [s for s in ORDER if (tok, s) in pairs][::-1]
        for y, s in enumerate(systems):
            u, st, c = pairs[(tok, s)]["reference"], pairs[(tok, s)]["steered"], color(s)
            ax.plot([u["share"], st["share"]], [y, y], color=c, linewidth=1.2, alpha=0.6, zorder=2)
            ax.plot([u["share"]], [y], "o", color=c, markerfacecolor="white", markersize=4.5,
                    markeredgewidth=1.2, zorder=3)
            ax.plot([st["share"]], [y], "o", color=c, markersize=4.5, zorder=3)
            ax.text(0.25, y, f"+{100 * (st['share'] - u['share']):.0f} pts", ha="right", va="center",
                    color=MUTED, fontsize=7)
        ax.set_yticks(range(len(systems)), [NAMES[s] for s in systems])
        ax.set_ylim(-0.6, len(systems) - 0.4)
        ax.set_xlim(0, 0.25)
        ax.set_xticks([0, 0.1, 0.2], ["0", "10%", "20%"])
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(TOK[tok], color=INK, fontsize=8.5)
        ax.set_xlabel("share ticked “cacophonous”")
    h = [plt.Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="none",
                    label="unguided reference"),
         plt.Line2D([], [], marker="o", color=MUTED, linestyle="none", label="steered version (same trials)")]
    fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.savefig(out / "survey_caco.pdf")
    plt.close(fig)


def pct(x):
    return f"{100 * x:.0f}"


def tab_part2(main, path):
    rows = [r for r in main["part2"] if r["grouping"] == "tok x attribute x system"]
    lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
             r"Tokenizer & Attribute & System & Signed (CI) & Right \% (CI) & Opp.\ \% & None \% & $n$ \\",
             r"\midrule"]
    prev = None
    for tok in ("remi", "ant"):
        for attr in ("velocity", "duration", "pitch_register"):
            for s in ORDER:
                r = next((x for x in rows if tuple(x["group"]) == (tok, attr, s)), None)
                if not r:
                    continue
                if prev and prev != (tok, attr):
                    lines.append(r"\addlinespace")
                prev = (tok, attr)
                lines.append(f"{TOK[tok]} & {ATTR[attr]} & {TEX_NAMES[s]} & "
                             f"${r['signed']:+.2f}$ [${r['lo']:+.2f}$, ${r['hi']:+.2f}$] & "
                             f"{pct(r['right'])} [{pct(r['right_lo'])}, {pct(r['right_hi'])}] & "
                             f"{pct(r['wrong'])} & {pct(r['none'])} & {r['n']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def tab_part3(main, path):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Tokenizer & System & Score (CI) & Best \% & Worst \% & $n$ \\", r"\midrule"]
    for i, tok in enumerate(("remi", "ant")):
        if i:
            lines.append(r"\addlinespace")
        for r in [x for x in main["part3_tok"] if x["group"] == tok]:
            lines.append(f"{TOK[tok]} & {TEX_NAMES[r['system']]} & ${r['score']:+.2f}$ "
                         f"[${r['lo']:+.2f}$, ${r['hi']:+.2f}$] & {pct(r['best'])} & {pct(r['worst'])} & {r['n']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def tab_robust(variants, path):
    """variants: [(label, json)]; key numbers side by side."""
    def p1(j, tok, s):
        return next(r["score"] for r in j["part1"] if r["group"] == tok and r["system"] == s)

    def p2(j, grouping, g):
        return next(r["right"] for r in j["part2"] if r["grouping"] == grouping and r["group"] == g)
    rows = [("Participants", lambda j: f"{j['n']}")]
    for tok in ("remi", "ant"):
        for s in ("original", "ebt", "llama", "gpt2"):
            rows.append((f"Part 1 {TOK[tok]}, {NAMES[s]}", lambda j, tok=tok, s=s: f"${p1(j, tok, s):+.2f}$"))
    rows.append(("Part 2 right direction, all", lambda j: pct(p2(j, "overall", "all"))))
    rows.append(("Part 2 right direction, EBT", lambda j: pct(p2(j, "system family", "EBT"))))
    rows.append(("Part 2 right direction, AR", lambda j: pct(p2(j, "system family", "AR baselines"))))
    lines = [r"\begin{tabular}{l" + "r" * len(variants) + "}", r"\toprule",
             " & " + " & ".join(lab for lab, _ in variants) + r" \\", r"\midrule"]
    for lab, f in rows:
        lines.append(lab + " & " + " & ".join(f(j) for _, j in variants) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def tab_training(bt, path):
    keep = [("caco: all clips", "Cacophonous, all clips"), ("caco: original", "Cacophonous, human originals"),
            ("caco: unguided", "Cacophonous, unguided"), ("caco: steered", "Cacophonous, steered"),
            ("caco: broken catch clip", "Cacophonous, broken catch clip"),
            ("catch_unguided pass", "Broken clip picked as least musical"),
            ("catch_change 'no difference'", "Identical pair: ``no difference''"),
            ("P2 right direction (all)", "Part 2: requested direction"),
            ("P1 best-worst remi original", "Part 1: REMI original score"),
            ("P1 best-worst remi ebt", "Part 1: REMI EBT score")]
    rows = {r["metric"]: r for r in bt["rows"]}
    c = bt["counts"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             f"Measure & None ({c['None']}) & Some ({c['Some']}) & Extensive ({c['Extensive']}) & Ext.\\ $-$ None (CI) \\\\",
             r"\midrule"]
    for k, lab in keep:
        r = rows[k]
        score = k.startswith("P1")
        f = (lambda x: f"${x:+.2f}$") if score else pct
        d = r["diff_ext_none"]
        dfmt = (f"${d[0]:+.2f}$ [${d[1]:+.2f}$, ${d[2]:+.2f}$]" if score
                else f"${100 * d[0]:+.0f}$ [${100 * d[1]:+.0f}$, ${100 * d[2]:+.0f}$]").replace("-0$", "0$").replace("+0$", "0$")
        lines.append(f"{lab} & {f(r['None'][0])} & {f(r['Some'][0])} & {f(r['Extensive'][0])} & {dfmt} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    load = lambda n: json.loads((args.analysis / n).read_text())
    main_, strict, keep, bt = load("main.json"), load("strict.json"), load("keep_caco.json"), load("by_training.json")
    style()
    fig_part1(main_, args.out)
    fig_part2(main_, args.out)
    fig_caco(main_, args.out)
    tab_part2(main_, args.tables / "part2_cells.tex")
    tab_part3(main_, args.tables / "part3.tex")
    variants = [("Main", main_), ("Strict exclusion", strict)]
    no686 = args.analysis / "no686.json"
    if no686.exists():
        variants.append(("Without 686e30f1", json.loads(no686.read_text())))
    tab_robust(variants, args.tables / "robustness.tex")
    tab_training(bt, args.tables / "training.tex")
    print("wrote", sorted(p.name for p in args.out.glob("survey_*.pdf")), sorted(p.name for p in args.tables.glob("*.tex")))


if __name__ == "__main__":
    main()
