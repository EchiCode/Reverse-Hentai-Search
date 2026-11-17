"""Plot measured posting-vector recall and gzip costs from a saved comparison."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "balanced-l2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter


FAMILIES = ("reference", "lvq", "sq", "pq", "opq", "rq", "lsq", "gguf", "fp8")
COLORS = ("#666666", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F3C8D", "#17BECF")
MARKERS = ("s", "o", "^", "D", "P", "X", "v", "<", ">")
EXCLUDED_METHOD_PREFIXES = ("sq", "rq", "lsq", "gguf_", "opq", "pq192x", "pq96x", "pq64x", "pq32x", "fp8_e4m3")


def _validate(saved):
    config, points, cases = saved["manifest"], saved["points"], saved["cases"]
    methods, probes = config["methods"], config["probes"]
    if (saved["status"] not in ("running", "complete") or not methods
            or len(set(methods)) != len(methods) or not points
            or config["query_count"] <= 0 or not probes or len(set(probes)) != len(probes)
            or any(isinstance(probe, bool) or not isinstance(probe, int) or probe <= 0 for probe in probes)
            or not math.isfinite(config["noise_std"]) or config["noise_std"] < 0):
        raise ValueError("Missing or invalid posting quantization settings")
    seen = [(point["method"], point["probes"]) for point in points]
    if len(set(seen)) != len(seen) or not set(seen).issubset((method, probe) for method in methods for probe in probes):
        raise ValueError("Duplicate or unknown quantization method/probe setting")
    if set(cases) - set(methods):
        raise ValueError("Unknown comparison case")
    completed = {name for name, case in cases.items() if case["status"] == "complete"}
    rejected = {name for name, case in cases.items() if case["status"] == "rejected"}
    expected = {(method, probe) for method in completed for probe in probes}
    if set(seen) != expected or completed & rejected:
        raise ValueError("Completed cases do not match plotted measurements")
    if saved["status"] == "complete" and completed | rejected != set(methods):
        raise ValueError("Completed comparison is missing method outcomes")
    for point in points:
        if (point["queries"] != config["query_count"] or point["probes"] not in probes
                or point["noise_std"] != config["noise_std"] or point["family"] not in FAMILIES):
            raise ValueError("Measurement differs from shared comparison settings")
        for key in ("recall_percent", "recall_vs_fp16_percent", "routing_coverage_percent"):
            if not math.isfinite(point[key]) or not 0 <= point[key] <= 100:
                raise ValueError("Invalid recall measurement")
        for key in ("posting_gzip_MB_mean", "posting_gzip_MB_p95", "cold_gzip_MB_mean", "total_posting_gzip_GB"):
            if not math.isfinite(point[key]) or point[key] <= 0:
                raise ValueError("Invalid gzip measurement")
        for key in ("model_gzip_MB", "centroid_gzip_MB", "encode_seconds", "decode_seconds", "score_seconds"):
            if not math.isfinite(point[key]) or point[key] < 0:
                raise ValueError("Invalid codec size or timing")
        expected = point["posting_gzip_MB_mean"] + point["model_gzip_MB"] + point["centroid_gzip_MB"]
        if not math.isclose(point["cold_gzip_MB_mean"], expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Cold download size does not include measured postings and shared state")
    return config, points, rejected


def _frontier(points, cost):
    best = -1
    frontier = []
    for point in sorted(points, key=lambda row: (row[cost], -row["recall_percent"])):
        if point["recall_percent"] > best:
            frontier.append(point)
            best = point["recall_percent"]
    return frontier


def _scatter(ax, points, cost):
    for family, color, marker in zip(FAMILIES, COLORS, MARKERS):
        subset = [point for point in points if point["family"] == family]
        if subset:
            label = " / ".join(dict.fromkeys(point["method"].upper() for point in subset)) if family == "reference" else family.upper()
            ax.scatter([point[cost] for point in subset], [point["recall_percent"] for point in subset],
                       s=35, marker=marker, c=color, alpha=.8, label=label)
    frontier = _frontier(points, cost)
    ax.plot([point[cost] for point in frontier], [point["recall_percent"] for point in frontier],
            color="#666666", linewidth=.8, linestyle="--", alpha=.6)
    # A bounded set of frontier labels keeps the full 114-method plot readable.
    labels = frontier if len(frontier) <= 12 else [frontier[i * (len(frontier) - 1) // 11] for i in range(12)]
    baseline = [point for point in points if point["method"] == "fp16"]
    if baseline:
        point = max(baseline, key=lambda row: row["probes"])
        if point not in labels:
            labels.append(point)
    for index, point in enumerate(labels):
        label = f"{point['method'].replace('x', '×')} p{point['probes']}"
        ax.annotate(label, (point[cost], point["recall_percent"]),
                    fontsize=6, xytext=(4, 5 if index % 2 else -10), textcoords="offset points")
    _style(ax, points)


def _style(ax, points):
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    low = max(0, min(point["recall_percent"] for point in points) - 1)
    ax.set_ylim(low, min(101, max(point["recall_percent"] for point in points) + 1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=6))


def _groups(points):
    groups = {}
    for point in points:
        family, method = point["family"], point["method"]
        if family == "pq":
            stem = "pq_m" + re.fullmatch(r"pq(\d+)x\d+", method).group(1)
        elif family == "fp8":
            stem = "_".join(method.split("_")[:2])
        else:
            stem = family
        groups.setdefault(stem, []).append(point)
    return groups


def _curves(ax, points, cost):
    methods = list(dict.fromkeys(point["method"] for point in points))
    for index, method in enumerate(methods):
        curve = sorted((point for point in points if point["method"] == method), key=lambda point: point["probes"])
        color, marker = COLORS[index % len(COLORS)], MARKERS[index % len(MARKERS)]
        ax.plot([point[cost] for point in curve], [point["recall_percent"] for point in curve],
                color=color, marker=marker, markersize=4, linewidth=1.5 if method == "fp16" else 1,
                linestyle="--" if method == "fp16" else "-",
                label="FP16" if method == "fp16" else method.replace("x", "×"))
        for point in curve:
            ax.annotate(str(point["probes"]), (point[cost], point["recall_percent"]), fontsize=6,
                        color=color, xytext=(3, 4 + index % 3 * 3), textcoords="offset points")
    _style(ax, points)


def render(directory):
    directory = Path(directory)
    saved = json.loads((directory / "results.json").read_text())
    config, points, rejected = _validate(saved)
    points = [point for point in points if not point["method"].startswith(EXCLUDED_METHOD_PREFIXES) and point["probes"] >= 4]
    rejected = {name for name in rejected if not name.startswith(EXCLUDED_METHOD_PREFIXES)}
    if not points:
        raise ValueError("No measurements remain after the plot filters")
    baseline = [point for point in points if point["method"] == "fp16"]
    output = directory / "plots"
    if output.is_symlink():
        raise ValueError("Plot output must not be a symlink")
    output.mkdir(exist_ok=True)
    prefix = "PARTIAL · " if saved["status"] != "complete" else ""
    probes = [probe for probe in config["probes"] if probe >= 4]
    probe_label = ", ".join(map(str, probes))
    subtitle = (f"{config['head_count']:,} fixed clusters · noise {config['noise_std']:g} · "
                f"probes {probe_label} · {config['query_count']:,} shared replay queries")
    footnote = "MB = 1,000,000 bytes. Posting downloads exclude centroids, shared codec state, image model, UI, metadata and HTTP overhead."
    limitation = "Image-ID top-1 recall after reconstructed-vector scoring; tie-aware recall is in CSV/JSON. Noisy replay is not real-image recall; timings exclude browser/network latency."
    figures, titles = [], {}
    download_columns = (("posting_gzip_MB_mean", "Mean gzip posting download (MB / query)"),)
    specifications = [("quantization", "All methods: recall versus per-query downloads", points, download_columns, False),
                      ("storage_recall", "All methods: recall versus full posting storage", points,
                       (("total_posting_gzip_GB", "Full posting corpus gzip GB"),), False)]
    lvq = [point for point in points if point["family"] == "lvq"]
    if lvq:
        specifications.append(("comparison_lvq", "All LVQ formats vs FP16", baseline + lvq, download_columns, True))
    for stem, title, subset, columns, curves in specifications:
        fig, axes = plt.subplots(1, len(columns), figsize=(13, 7), squeeze=False, sharey=True)
        for ax, (cost, label) in zip(axes[0], columns):
            (_curves if curves else _scatter)(ax, subset, cost)
            ax.set_xlabel(label)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(4, 6, 8, 9)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[0, 0].set_ylabel("Image-ID recall@1 (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .13), ncol=3 if curves else 5, frameon=False, fontsize=8)
        fig.suptitle(f"{prefix}Posting vector quantization · {title}", fontweight="bold")
        fig.text(.5, .91, subtitle, ha="center")
        fig.text(.055, .065, footnote, fontsize=8)
        fig.text(.055, .035, limitation, fontsize=8)
        fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.3, wspace=.12)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=170)
        plt.close(fig)
        figures.append(stem)
        titles[stem] = title
    obsolete_groups = {"sq4", "sq6", "sq8", "reference", "lvq", "sq"}
    for group, subset in _groups(saved["points"]).items():
        if all(point["method"].startswith(EXCLUDED_METHOD_PREFIXES) for point in subset):
            obsolete_groups.add(group)
    obsolete_stems = {"comparison_4bit", "comparison_5_6bit", "comparison_8bit"}
    obsolete_stems.update("family_" + group for group in obsolete_groups)
    for stem in obsolete_stems:
        for extension in ("png", "svg", "pdf"):
            (output / f"{stem}.{extension}").unlink(missing_ok=True)
    rejected_html = "".join(f"<li><strong>{html.escape(name)}</strong>: {html.escape(str(saved['cases'][name].get('reason', 'Rejected')))}</li>"
                            for name in sorted(rejected))
    (output / "index.html").write_text("<!doctype html><meta charset=utf-8><title>Posting vector quantization</title>"
        "<style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:0 16px}img{width:100%}</style>"
        f"<h1>{prefix}Posting vector quantization</h1><p>{html.escape(subtitle)}</p>"
        "<p><a href='../results.csv'>CSV</a> · <a href='../results.json'>JSON</a></p>"
        f"<p>{len(points) // len(probes)} measured methods across {len(probes)} probe counts; {len(rejected)} rejected. "
        "Curve labels show the number of posting probes. In comparison plots, the dashed FP16 curve is the reference. "
        "All LVQ formats share one comparison plot with FP16. At a fixed probe count FP16 has the highest measured recall; "
        "compare different codec/probe pairs at similar download sizes to assess the bandwidth tradeoff. "
        "Plots omit SQ, RQ, LSQ, GGUF, OPQ, FP8 E4M3, "
        "PQ with 32/64/96/192 subquantizers, and probe counts below 4.</p>"
        "<p>Mean gzip is the average sum of compressed posting files fetched for a query; posting cache hits are not simulated. "
        "Cold gzip adds the compressed centroid index and shared codec model once. Cold measurements remain in CSV/JSON. "
        "The saved benchmark results are unchanged.</p>"
        + "<nav>" + " · ".join(f"<a href='#{stem}'>{html.escape(titles[stem])}</a>" for stem in figures) + "</nav>"
        + "".join(f"<section id='{stem}'><h2>{html.escape(title)}</h2><img src='{stem}.png' alt='{html.escape(title)}'>"
                    f"<p><a href='{stem}.pdf'>PDF</a> · <a href='{stem}.svg'>SVG</a></p></section>"
                    for stem, title in titles.items())
        + f"<p>{html.escape(footnote)}</p><p>{html.escape(limitation)}</p>"
        + (f"<h2>Rejected methods</h2><ul>{rejected_html}</ul>" if rejected else ""))
    return {"figures": figures, "points": len(points), "rejected": len(rejected)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(render(parser.parse_args().directory), indent=2))
