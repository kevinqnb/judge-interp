"""Calibration report shared by ``entity_detection.py`` and ``relation_set_detection.py``: per error type, a
calibration diagram and a table of smECE / Brier skill (BSS) / AUROC, every interval a percentile interval over
``report.n_boot`` resamples of DOCUMENTS (a resample keeps every row of each drawn document).

Nothing here changes a metric the analyses already report: the pooled smECE / BSS / AUROC computed here are
asserted equal to the analyses' own (``assert_matches_existing``), and the intervals are new outputs.

An evaluation set (``evset``) is a dict: ``y`` (bool, valid), ``docs`` and ``error_type`` (str arrays, one entry per
row) and ``probs`` (``{method label: P(valid)}``, float arrays). Per error type, per method:

* point estimate = the metric on the type's full rows; interval = percentiles of the document resamples. The
  resamples are drawn once per type and shared by every method, so methods are compared on the same resamples.
* BSS reference = the (re)sample's own valid rate, as in the analyses (``evaluation_set_rate``).
* calibration curve = relplot's reflected-Gaussian kernel smoother at the bandwidth smECE selects on the full rows
  (held fixed over resamples); the band is the pointwise percentile interval. Both are drawn everywhere, faded by
  the predictions' kernel density (as relplot's own plot): opacity = ``report.alpha_floor`` + (1 - ``alpha_floor``) *
  density / max density (rounded to ``report.alpha_levels`` levels), so regions with few predictions are lightly drawn.
* shuffled-label baseline: the labels permuted once within the type, each method's probabilities scored against
  them on the same resamples; the baseline row is the mean over methods.

Usage (re-draw from saved predictions): ``python analysis/calibration_report.py <config.yaml> <predictions.npz> <out_dir> <stem>
[--control]`` (``--control``: the npz holds a ``control`` row mask; also draw the control report ``control_<stem>``).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import numpy as np
import relplot
import yaml
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from relplot.kernels import ReflectedGaussianKernel
from sklearn.metrics import roc_auc_score

METRICS = ("smece", "bss", "auroc")
HEADERS = {"smece": r"smECE $\downarrow$", "bss": r"BSS $\uparrow$", "auroc": r"AUROC $\uparrow$"}
SHUFFLED = "shuffled"
REPORT_KEYS = ("n_boot", "methods", "error_type_names", "font", "white_frac", "mesh_pts", "alpha_floor", "band_alpha", "alpha_levels",
               "shuffled_auroc_tolerance", "figsize")
TAB10 = plt.get_cmap("tab10").colors


# ------------------------------------------------------------------------------ evaluation sets
def check_evset(ev: dict, labels: list[str]) -> None:
    """Shapes, dtypes, finite probabilities in [0, 1], and exactly the expected methods."""
    n = len(ev["y"])
    assert ev["y"].dtype == bool and n > 0, ev["y"].dtype
    assert ev["docs"].shape == ev["error_type"].shape == (n,), (ev["docs"].shape, ev["error_type"].shape, n)
    assert sorted(ev["probs"]) == sorted(labels), f"methods {sorted(ev['probs'])} != report methods {sorted(labels)}"
    for label, f in ev["probs"].items():
        assert f.shape == (n,) and f.dtype == np.float64, (label, f.shape, f.dtype)
        assert np.isfinite(f).all() and ((0.0 <= f) & (f <= 1.0)).all(), f"{label}: probabilities outside [0, 1]"


def subset_evset(ev: dict, keep: np.ndarray) -> dict:
    """The rows ``keep`` (bool mask) of an evaluation set."""
    assert keep.dtype == bool and keep.shape == ev["y"].shape, (keep.dtype, keep.shape)
    return {"y": ev["y"][keep], "docs": ev["docs"][keep], "error_type": ev["error_type"][keep],
            "probs": {k: v[keep] for k, v in ev["probs"].items()}}


def split_report(ev: dict, error_types: list[str]) -> dict:
    """Counts per error type, and the check that the data split cleanly: exactly the expected error types, each
    with valid and invalid rows and more than one document (otherwise no document bootstrap).

    Raises:
        AssertionError: an unexpected / missing error type, or a type without both classes.
    """
    found = sorted(set(ev["error_type"].tolist()))
    assert found == sorted(error_types), f"error types {found} != expected {sorted(error_types)}"
    out: dict[str, float] = {}
    for et in error_types:
        sel = ev["error_type"] == et
        y = ev["y"][sel]
        assert y.any() and (~y).any(), f"error type {et!r} lacks valid or invalid rows ({int(y.sum())} valid / {int((~y).sum())} invalid)"
        n_docs = len(set(ev["docs"][sel].tolist()))
        assert n_docs > 1, f"error type {et!r}: {n_docs} document(s)"
        out[f"n_{et}"], out[f"n_valid_{et}"], out[f"n_invalid_{et}"], out[f"n_docs_{et}"] = int(sel.sum()), int(y.sum()), int((~y).sum()), n_docs
    return out


# ------------------------------------------------------------------------------ metrics and bootstrap
def score_sample(y: np.ndarray, f: np.ndarray) -> tuple[dict, float]:
    """``({smece, bss, auroc}, smECE bandwidth)`` of one (re)sample; BSS reference = the sample's own valid rate."""
    assert y.dtype == bool and f.shape == y.shape and f.dtype == np.float64
    assert y.any() and (~y).any(), "a sample lacks a class"
    yf = y.astype(np.float64)
    ce, sigma = relplot.smECE(f, yf, return_width=True)
    ref = float(yf.mean())
    brier_ref = float(np.mean((ref - yf) ** 2))
    assert brier_ref > 0
    return {"smece": float(ce), "bss": 1.0 - float(np.mean((f - yf) ** 2)) / brier_ref, "auroc": float(roc_auc_score(y, f))}, float(sigma)


def doc_draws(docs: np.ndarray, n_boot: int, seed: int | list[int]) -> tuple[list[np.ndarray], np.ndarray]:
    """``(rows of each document, draws)``: ``draws[b]`` lists the documents of resample ``b`` (with replacement,
    as many as there are documents)."""
    assert isinstance(n_boot, int) and n_boot > 0, n_boot
    uniq, inv = np.unique(docs, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    assert len(members) == len(uniq) and sum(len(m) for m in members) == len(docs)
    return members, np.random.default_rng(seed).integers(len(uniq), size=(n_boot, len(uniq)))


def interval(reps: np.ndarray, ci_level: float) -> tuple[float, float]:
    assert 0.0 < ci_level < 1.0 and np.isfinite(reps).all()
    lo, hi = np.quantile(reps, [(1 - ci_level) / 2, (1 + ci_level) / 2])
    return float(lo), float(hi)


def analyse_type(y: np.ndarray, probs: dict[str, np.ndarray], docs: np.ndarray, rep: dict, ci_level: float,
                 seed: int, type_index: int) -> dict:
    """All methods' metrics, curves and the shuffled baseline on one error type's rows (see the module docstring).

    Returns ``{"n", "n_docs", "valid_rate", "mesh", "methods": {label: {metric: (point, lo, hi), "curve": ...}},
    SHUFFLED: {metric: (point, lo, hi)}}``.
    """
    n_boot, mesh = rep["n_boot"], np.linspace(0.0, 1.0, rep["mesh_pts"])
    members, draws = doc_draws(docs, n_boot, [seed, type_index])
    y_perm = np.random.default_rng([seed, type_index, 1]).permutation(y)
    yf = y.astype(np.float64)
    full, sigma, point_curve, density = {}, {}, {}, {}
    shuf_full = {}
    for label, f in probs.items():
        full[label], sigma[label] = score_sample(y, f)
        kern = ReflectedGaussianKernel(sigma[label])
        point_curve[label] = kern.smooth(f, yf, mesh)[0]
        dens = kern.kde(f, mesh)
        density[label] = dens / dens.max()
        shuf_full[label], _ = score_sample(y_perm, f)
    reps = {l: {m: np.empty(n_boot) for m in METRICS} for l in probs}
    shuf_reps = {l: {m: np.empty(n_boot) for m in METRICS} for l in probs}
    curves = {l: np.empty((n_boot, len(mesh))) for l in probs}
    for b in range(n_boot):
        idx = np.concatenate([members[j] for j in draws[b]])
        for label, f in probs.items():
            sc, _ = score_sample(y[idx], f[idx])
            ssc, _ = score_sample(y_perm[idx], f[idx])
            for m in METRICS:
                reps[label][m][b], shuf_reps[label][m][b] = sc[m], ssc[m]
            curves[label][b] = ReflectedGaussianKernel(sigma[label]).smooth(f[idx], yf[idx], mesh)[0]
    out = {"n": int(len(y)), "n_docs": len(members), "valid_rate": float(y.mean()), "mesh": mesh, "methods": {}}
    for label in probs:
        res = {m: (full[label][m], *interval(reps[label][m], ci_level)) for m in METRICS}
        lo, hi = np.quantile(curves[label], [(1 - ci_level) / 2, (1 + ci_level) / 2], axis=0)
        res["curve"] = {"mu": point_curve[label], "lo": lo, "hi": hi, "density": density[label], "sigma": sigma[label]}
        out["methods"][label] = res
    out[SHUFFLED] = {m: (float(np.mean([shuf_full[l][m] for l in probs])),
                         *interval(np.mean([shuf_reps[l][m] for l in probs], axis=0), ci_level)) for m in METRICS}
    out[f"{SHUFFLED}_by_method"] = {l: {m: (shuf_full[l][m], *interval(shuf_reps[l][m], ci_level)) for m in METRICS} for l in probs}
    assert abs(out[SHUFFLED]["auroc"][0] - 0.5) <= rep["shuffled_auroc_tolerance"], (
        f"shuffled-label AUROC {out[SHUFFLED]['auroc'][0]:.4f} is not ~0.5 -- something leaks")
    return out


def analyse(ev: dict, rep: dict, error_types: list[str], ci_level: float, seed: int) -> dict:
    """``{error type: analyse_type(...)}`` for ``error_types`` (in that order), checked for a clean split first."""
    check_evset(ev, list(rep["methods"]))
    split_report(ev, error_types)
    res = {}
    for i, et in enumerate(error_types):
        sel = ev["error_type"] == et
        res[et] = analyse_type(ev["y"][sel], {l: ev["probs"][l][sel] for l in rep["methods"]}, ev["docs"][sel], rep, ci_level, seed, i)
    return res


def assert_matches_existing(ev: dict, existing: dict, prefix: str = "test") -> None:
    """The pooled smECE / BSS / AUROC computed here equal the analysis's own ``metrics`` (``existing``, keys
    ``<label>_test_smece`` ...), so this module cannot disagree with the numbers already reported.

    Raises:
        AssertionError: a disagreement above 1e-9.
    """
    for label, f in ev["probs"].items():
        sc, _ = score_sample(ev["y"], f)
        for m in METRICS:
            want = existing[f"{label}_{prefix}_{m}"]
            assert abs(sc[m] - want) <= 1e-9, f"{label}: pooled {m} {sc[m]!r} != the analysis's {want!r}"


# ------------------------------------------------------------------------------ outputs
def pastel(color, white_frac: float) -> tuple:
    """``color`` mixed with white (``white_frac`` of white)."""
    assert 0.0 <= white_frac < 1.0
    return tuple((1 - white_frac) * c + white_frac * 1.0 for c in to_rgb(color))


def set_style(font: str) -> None:
    """ACL-style serif text; the font must be installed (no silent fallback); TrueType (Type 42) fonts in PDFs."""
    font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({"font.family": "serif", "font.serif": [font], "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "stix", "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})


def method_colors(rep: dict) -> dict[str, tuple]:
    assert len(rep["methods"]) <= len(TAB10), "more methods than tab10 colours"
    return {label: pastel(TAB10[i], rep["white_frac"]) for i, label in enumerate(rep["methods"])}


def draw_faded(ax, mesh: np.ndarray, c: dict, color: tuple, alpha_floor: float, band_alpha: float, alpha_levels: int) -> None:
    """The curve ``c["mu"]`` and its band ``c["lo"]``..``c["hi"]``, faded by density (``c["density"]``, max 1): a mesh
    interval's opacity is ``alpha_floor + (1 - alpha_floor) * density`` (mean of its ends) rounded to one of
    ``alpha_levels`` evenly spaced levels; the band is further scaled by ``band_alpha``. Each run of intervals at one level
    is a single polyline / fill (semi-transparent segments drawn one by one darken where they overlap)."""
    assert 0.0 <= alpha_floor < 1.0 and 0.0 <= band_alpha <= 1.0 and c["density"].max() == 1.0
    assert isinstance(alpha_levels, int) and alpha_levels >= 2, alpha_levels
    w = alpha_floor + (1 - alpha_floor) * c["density"]
    level = np.rint(((w[:-1] + w[1:]) / 2 - alpha_floor) / (1 - alpha_floor) * (alpha_levels - 1)).astype(int)
    start = 0
    for i in range(1, len(level) + 1):
        if i == len(level) or level[i] != level[start]:
            sl = slice(start, i + 1)   # the run's mesh points: its intervals' ends, shared with the neighbouring runs
            a = alpha_floor + (1 - alpha_floor) * level[start] / (alpha_levels - 1)
            ax.fill_between(mesh[sl], c["lo"][sl], c["hi"][sl], color=color, alpha=band_alpha * a, lw=0, zorder=2)
            ax.plot(mesh[sl], c["mu"][sl], color=color, alpha=a, lw=1.8, solid_capstyle="butt", zorder=3)
            start = i


def plot_calibration(res: dict, rep: dict, out_dir: Path, stem: str) -> list[Path]:
    """One ``<stem>_<error type>.pdf`` per error type (all methods, shaded document-bootstrap bands) and
    ``<stem>_legend.pdf`` (the legend alone)."""
    set_style(rep["font"])
    colors, paths = method_colors(rep), []
    for et, r in res.items():
        fig, ax = plt.subplots(figsize=tuple(rep["figsize"]), constrained_layout=True)
        ax.plot([0, 1], [0, 1], ":", color="0.45", lw=1.0, zorder=1)
        for label, mr in r["methods"].items():
            draw_faded(ax, r["mesh"], mr["curve"], colors[label], rep["alpha_floor"], rep["band_alpha"], rep["alpha_levels"])
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Predicted probability (valid)", ylabel="Observed frequency (valid)",
               title=rep["error_type_names"][et])
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        paths.append(out_dir / f"{stem}_{et}.pdf")
        fig.savefig(paths[-1])
        plt.close(fig)
    handles = [Line2D([0], [0], color=colors[l], lw=2.4, label=name) for l, name in rep["methods"].items()]
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="center", handlelength=1.6, columnspacing=1.4)
    paths.append(out_dir / f"{stem}_legend.pdf")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    return paths


def fmt_cell(t: tuple) -> str:
    return f"{t[0]:.3f} {{\\scriptsize [{t[1]:.3f}, {t[2]:.3f}]}}"


def latex_table(res: dict, rep: dict, ci_level: float, label: str, caption_extra: str) -> str:
    """booktabs ``table*`` (needs ``\\usepackage{booktabs}``): a row per method + the shuffled-label baseline, a
    column group per error type holding smECE / BSS / AUROC with arrows and bracketed document-bootstrap intervals."""
    ets, k = list(res), len(METRICS)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{l" + ("c" * k) * len(ets) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{rep['error_type_names'][et]}}}" for et in ets) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(len(ets))),
             "Method & " + " & ".join(HEADERS[m] for _ in ets for m in METRICS) + r" \\", r"\midrule"]
    for name_label, name in rep["methods"].items():
        lines.append(name + " & " + " & ".join(fmt_cell(res[et]["methods"][name_label][m]) for et in ets for m in METRICS) + r" \\")
    lines += [r"\midrule", "Shuffled labels & " + " & ".join(fmt_cell(res[et][SHUFFLED][m]) for et in ets for m in METRICS) + r" \\",
              r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Calibration (smECE), Brier skill (BSS) and AUROC by error type; brackets: {ci_level:.0%} intervals over "
              rf"{rep['n_boot']} bootstrap resamples of documents. Shuffled labels: the mean over methods of the same metrics with the "
              rf"labels randomly permuted. {caption_extra}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, res: dict, rep: dict, set_name: str) -> None:
    """Every number of the report: (set, error type, method, metric, point, lo, hi, n rows, n docs, n boot)."""
    rows = []
    for et, r in res.items():
        entries = [(name, r["methods"][l]) for l, name in rep["methods"].items()]
        entries += [(SHUFFLED, r[SHUFFLED])] + [(f"{SHUFFLED}:{rep['methods'][l]}", v) for l, v in r[f"{SHUFFLED}_by_method"].items()]
        for name, vals in entries:
            for m in METRICS:
                rows.append([set_name, et, name, m, *vals[m], r["n"], r["n_docs"], rep["n_boot"]])
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "error_type", "method", "metric", "point", "ci_lo", "ci_hi", "n_rows", "n_docs", "n_boot"])
        w.writerows(rows)


def write_report(ev: dict, rep: dict, error_types: list[str], ci_level: float, seed: int, fig_dir: Path, out_dir: Path,
                 stem: str, caption_extra: str) -> dict:
    """Analyse ``ev`` and write ``<stem>_<error type>.pdf``, ``<stem>_legend.pdf`` (``fig_dir``), ``<stem>_results.csv``
    and ``<stem>_table.tex`` (``out_dir`` and ``fig_dir``). Returns the split counts (flat, for ``metrics.json``)."""
    assert sorted(rep) == sorted(REPORT_KEYS), f"report keys {sorted(rep)} != {sorted(REPORT_KEYS)}"
    counts = split_report(ev, error_types)
    res = analyse(ev, rep, error_types, ci_level, seed)
    plot_calibration(res, rep, fig_dir, stem)
    tex = latex_table(res, rep, ci_level, f"tab:{stem}", caption_extra)
    for d in (out_dir, fig_dir):
        write_csv(d / f"{stem}_results.csv", res, rep, stem)
        (d / f"{stem}_table.tex").write_text(tex)
    return {f"{stem}_{k}": v for k, v in counts.items()}


# ------------------------------------------------------------------------------ saved predictions
def save_evset(path: Path, ev: dict, **extra: np.ndarray) -> None:
    """``ev`` (+ per-row ``extra`` arrays) as one npz, so the report can be redrawn without refitting any probe."""
    np.savez_compressed(path, y=ev["y"], docs=ev["docs"], error_type=ev["error_type"],
                        **{f"prob__{k}": v for k, v in ev["probs"].items()}, **extra)


def load_evset(path: Path) -> tuple[dict, dict]:
    """``(evset, extra arrays)`` of ``save_evset``."""
    with np.load(path, allow_pickle=False) as z:
        ev = {"y": z["y"], "docs": z["docs"], "error_type": z["error_type"],
              "probs": {k[len("prob__"):]: z[k] for k in z.files if k.startswith("prob__")}}
        extra = {k: z[k] for k in z.files if k not in ("y", "docs", "error_type") and not k.startswith("prob__")}
    return ev, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("predictions")
    ap.add_argument("out_dir")
    ap.add_argument("stem")
    ap.add_argument("--control", action="store_true", help="the npz holds a `control` row mask: also write control_<stem>")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    ev, extra = load_evset(Path(a.predictions))
    p = cfg["params"]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ets = list(p["report"]["error_type_names"])
    write_report(ev, p["report"], ets, p["ci_level"], cfg["seed"], out_dir, out_dir, a.stem, "")
    if a.control:
        write_report(subset_evset(ev, extra["control"]), p["report"], ets, p["ci_level"], cfg["seed"], out_dir, out_dir, f"control_{a.stem}", "")


if __name__ == "__main__":
    main()
