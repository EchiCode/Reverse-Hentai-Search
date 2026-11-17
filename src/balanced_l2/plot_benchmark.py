"""Plot saved recall and gzip-download frontiers without rebuilding datasets."""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "balanced-l2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(directory):
    directory = Path(directory)
    saved = json.loads((directory / "points.json").read_text())
    config, points = saved["manifest"], saved["points"]
    expected = {(head, noise, probe) for head in config["head_targets"] for noise in config["noise"] for probe in config["probes"]}
    seen = [(point["target_heads"], point["noise_std"], point["probes"]) for point in points]
    if not points or len(set(seen)) != len(seen) or not set(seen).issubset(expected):
        raise ValueError("Missing, duplicate or unknown plot settings")
    if saved["status"] == "complete" and set(seen) != expected:
        raise ValueError("Completed benchmark does not contain the full probe grid")
    for point in points:
        if (point["queries"] != config["query_count"] or not 0 <= point["recall_percent"] <= 100
                or abs(point["recall_percent"] - 100 * point["recovered_exact_nearest_rows"] / point["queries"]) > 1e-9
                or point["posting_gzip_MB_mean"] <= 0 or point["cold_gzip_MB_excluding_metadata"] <= 0):
            raise ValueError("Plot recall or gzip cost differs from measured query counts")
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    generated = []
    for noise in config["noise"]:
        subset = [row for row in points if row["noise_std"] == noise]
        if not subset:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
        for index, target in enumerate(config["head_targets"]):
            curve = sorted([row for row in subset if row["target_heads"] == target], key=lambda row: row["probes"])
            if not curve:
                continue
            label = f"{target:,} requested ({curve[0]['actual_heads']:,} actual)"
            for ax, key in zip(axes, ("posting_gzip_MB_mean", "cold_gzip_MB_excluding_metadata")):
                ax.plot([p[key] for p in curve], [p["recall_percent"] for p in curve], marker="o", markersize=4,
                        color=colors[index % len(colors)], label=label)
                for point in curve:
                    ax.annotate(str(point["probes"]), (point[key], point["recall_percent"]), fontsize=6,
                                xytext=(3, 4 + index * 3), textcoords="offset points", color=colors[index % len(colors)])
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Mean gzip posting MB / query (log scale)")
        axes[1].set_xlabel("Cold query: model/libraries + centroid + posting gzip MB")
        axes[0].set_ylabel("Recall@1 posting coverage (%)")
        for ax in axes:
            ax.set_ylim(0, 103)
            ax.grid(alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .13), ncol=2, frameon=False)
        prefix = "PARTIAL · " if saved["status"] != "complete" else ""
        fig.suptitle(f"{prefix}Balanced L2 | noise {noise:g} | LVQ6", fontweight="bold")
        fig.text(.5, .91, f"{config['query_count']:,} shared replay queries; labels show posting probes", ha="center")
        fig.text(.055, .065, "Actual gzip MB = 1,000,000 bytes. Cold cost excludes metadata, UI and HTTP overhead; browser/network latency unmeasured.", fontsize=8)
        fig.text(.055, .035, "Source: saved points.json; normalized full-corpus exact nearest-neighbor truth. One posting assignment per vector.", fontsize=8)
        fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.3, wspace=.12)
        stem = f"noise_{noise:g}"
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=170)
        plt.close(fig)
        generated.append(stem)
    (output / "index.html").write_text("<!doctype html><meta charset=utf-8><title>Balanced L2 calibration</title>"
        "<style>body{font:16px system-ui;max-width:1300px;margin:24px auto}img{width:100%}</style>"
        "<h1>Balanced L2 calibration</h1><p><a href='../points.csv'>CSV</a> · <a href='../points.json'>JSON</a></p>"
        + "".join(f"<section><h2>{name.replace('_', ' ')}</h2><img src='{name}.png' alt='Recall and gzip downloads'>"
                    f"<p><a href='{name}.pdf'>PDF</a> · <a href='{name}.svg'>SVG</a></p></section>" for name in generated))
    return {"figures": generated, "points": len(points)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(render(parser.parse_args().directory), indent=2))
