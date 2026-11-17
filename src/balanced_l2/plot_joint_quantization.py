"""Plot the independent centroid/posting codec grid from its saved measurements."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'balanced-l2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


METHODS = ('fp16', 'lvq8', 'lvq6', 'lvq5')
COLORS = ('#666666', '#0072B2', '#D55E00', '#009E73')
MARKERS = ('s', 'o', '^', 'D')


def _validate(saved):
    config, points = saved['manifest'], saved['points']
    if saved['status'] not in ('running', 'complete', 'pending') or not points:
        raise ValueError('No completed joint measurements to plot')
    heads, centroids, postings, probes = (config[key] for key in
        ('head_targets', 'centroid_methods', 'posting_methods', 'probes'))
    if (not heads or len(set(heads)) != len(heads) or any(not isinstance(h, int) or h < 1 for h in heads)
            or not centroids or not postings or len(set(centroids)) != len(centroids)
            or len(set(postings)) != len(postings) or set(centroids + postings) - set(METHODS)
            or not probes or sorted(set(probes)) != probes or any(not isinstance(p, int) or p < 1 for p in probes)
            or config['query_count'] < 1 or not math.isfinite(config['noise_std']) or config['noise_std'] < 0):
        raise ValueError('Invalid joint comparison settings')
    seen, pairs = set(), {}
    for point in points:
        key = point['target_heads'], point['centroid_method'], point['posting_method']
        setting = (*key, point['probes'])
        if (setting in seen or key[0] not in heads or key[1] not in centroids or key[2] not in postings
                or point['probes'] not in probes or point['actual_heads'] != key[0]
                or point['queries'] != config['query_count'] or point['noise_std'] != config['noise_std']):
            raise ValueError('Duplicate, unknown, or inconsistent joint measurement')
        seen.add(setting)
        pairs.setdefault(key, set()).add(point['probes'])
        for field in ('recall_percent', 'tie_aware_recall_percent', 'routing_coverage_percent'):
            if not math.isfinite(point[field]) or not 0 <= point[field] <= 100:
                raise ValueError('Invalid recall measurement')
        for field in ('posting_gzip_MB_mean', 'posting_gzip_MB_p95', 'centroid_gzip_MB',
                      'cold_gzip_MB_mean', 'total_posting_gzip_GB'):
            if not math.isfinite(point[field]) or point[field] <= 0:
                raise ValueError('Invalid measured gzip size')
        if not math.isclose(point['cold_gzip_MB_mean'], point['posting_gzip_MB_mean'] + point['centroid_gzip_MB'],
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('Cold bytes must include the centroid index and postings')
    if any(values != set(probes) for values in pairs.values()):
        raise ValueError('A measured codec pair is missing probe counts')
    if saved['status'] == 'complete' and len(pairs) != len(heads) * len(centroids) * len(postings):
        raise ValueError('Complete joint comparison is missing codec pairs')
    return config, points, len(pairs)


def _style(axis):
    axis.set_xscale('log')
    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(4, 6, 8, 9)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.25)
    axis.set_xlabel('Mean gzip posting download (MB / query)')
    axis.set_ylabel('Image-ID recall@1 (%)')


def _save(figure, directory, stem):
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(directory / f'{stem}.{extension}', dpi=170)
    plt.close(figure)


def _selection_plots(config, points, output, heads, centroid, prefix, subtitle):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    axis = axes[0]
    for index, count in enumerate(config['head_targets']):
        subset = sorted((p for p in points if p['target_heads'] == count and p['centroid_method'] == centroid),
                        key=lambda p: (p['posting_gzip_MB_mean'], -p['recall_percent']))
        best, frontier = -1, []
        for point in subset:
            if point['recall_percent'] > best:
                frontier.append(point)
                best = point['recall_percent']
        if frontier:
            axis.plot([p['posting_gzip_MB_mean'] for p in frontier], [p['recall_percent'] for p in frontier],
                      color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)], markersize=4,
                      label=f"{count:,} clusters · {frontier[0]['centroid_gzip_MB']:.1f} MB initial")
    _style(axis)
    axis.set_title(f'Cluster count with fixed {centroid.upper()} centroids')
    axis.legend(frameon=False, fontsize=9, loc='lower right')

    axis = axes[1]
    baseline = {p['probes']: p['recall_percent'] for p in points if p['target_heads'] == heads
                and p['centroid_method'] == 'fp16' and p['posting_method'] == 'fp16'}
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        curve = sorted((p for p in points if p['target_heads'] == heads and p['centroid_method'] == method
                        and p['posting_method'] == 'fp16' and p['probes'] in baseline), key=lambda p: p['probes'])
        if curve:
            axis.plot([p['probes'] for p in curve], [p['recall_percent'] - baseline[p['probes']] for p in curve],
                      color=color, marker=marker, markersize=4,
                      label=f"{method.upper()} · {curve[0]['centroid_gzip_MB']:.1f} MB initial")
    axis.set_xscale('log', base=2)
    axis.set_xticks(config['probes'], [str(p) for p in config['probes']])
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.25)
    axis.set_xlabel('Probes (FP16 posting vectors held fixed)')
    axis.set_ylabel('Recall change vs FP16 centroids (percentage points)')
    axis.set_title(f'Centroid compression at {heads:,} clusters')
    axis.legend(frameon=False, fontsize=9, loc='lower right')
    figure.suptitle(f'{prefix}Choosing one persistent centroid index', fontweight='bold')
    figure.text(.5, .91, subtitle, ha='center')
    figure.text(.06, .055, 'Left: each line fixes the centroid codec and varies posting codecs/probes. Segments connect measured points only.', fontsize=9)
    figure.text(.06, .025, 'MB = 1,000,000 bytes. Initial centroid download is separate from per-query posting bytes.', fontsize=9)
    figure.subplots_adjust(top=.8, bottom=.21, left=.065, right=.98, wspace=.3)
    counts_stem = f'fixed_{centroid}_counts'
    _save(figure, output, counts_stem)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for axis, metric, title in zip(axes, ('recall_percent', 'tie_aware_recall_percent'),
                                   ('Exact image-ID recall@1', 'Tie-aware recall@1 (cosine tolerance 2e-6)')):
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            curve = sorted((p for p in points if p['target_heads'] == heads and p['centroid_method'] == centroid
                            and p['posting_method'] == method), key=lambda p: p['probes'])
            if not curve:
                continue
            axis.plot([p['posting_gzip_MB_mean'] for p in curve], [p[metric] for p in curve],
                      color=color, marker=marker, markersize=4, linestyle='--' if method == 'fp16' else '-',
                      label=f'Vectors: {method.upper()}')
            for point in curve:
                axis.annotate(str(point['probes']), (point['posting_gzip_MB_mean'], point[metric]),
                              fontsize=8, color=color,
                              xytext=(4, -12 if method == 'lvq6' or (method == 'lvq8' and point['probes'] == 64) else 5),
                              textcoords='offset points')
        _style(axis)
        axis.set_title(title)
        axis.set_ylabel('Recall@1 (%)')
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    size = next(p['centroid_gzip_MB'] for p in points if p['target_heads'] == heads and p['centroid_method'] == centroid)
    figure.suptitle(f'{prefix}{heads:,} clusters · {centroid.upper()} centroids · {size:.2f} MB initial index', fontweight='bold')
    figure.text(.5, .91, subtitle, ha='center')
    figure.text(.06, .055, 'Point labels are probes. Tie-aware recall accepts a different image ID only if its original cosine score is within tolerance.', fontsize=9)
    figure.text(.06, .025, 'Actual gzip posting bytes, counted afresh per query. Real-image recall and browser/network latency are not measured.', fontsize=9)
    figure.subplots_adjust(top=.8, bottom=.21, left=.065, right=.98, wspace=.13)
    selected_stem = f'selected_{heads}_{centroid}'
    _save(figure, output, selected_stem)
    return {counts_stem: 'Fixed-index cluster count and centroid compression tradeoffs',
            selected_stem: f'{heads:,} clusters / {centroid.upper()}: switchable posting vectors'}


def render(directory, *, selected_heads=None, selected_centroid='lvq6'):
    directory = Path(directory)
    saved = json.loads((directory / 'results.json').read_text())
    config, points, pair_count = _validate(saved)
    if selected_heads is not None and not any(p['target_heads'] == selected_heads
            and p['centroid_method'] == selected_centroid for p in points):
        raise ValueError('Selected centroid index has no measurements')
    output = directory / 'plots'
    if output.is_symlink():
        raise ValueError('Plot output must not be a symlink')
    output.mkdir(exist_ok=True)
    prefix = 'PARTIAL · ' if saved['status'] != 'complete' else ''
    subtitle = (f"noise {config['noise_std']:g} · {config['query_count']:,} shared queries · probes "
                + ', '.join(map(str, config['probes'])))
    footnote = 'Actual gzip sizes; MB = 1,000,000 bytes. Centroid index is an initial download; postings are counted afresh per query.'
    limitation = 'Noisy vector replay; real-image recall and browser/network latency are not measured. Full details: results.csv / results.json.'
    figures = {}
    for heads in config['head_targets']:
        subset = [point for point in points if point['target_heads'] == heads]
        if not subset:
            continue
        figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True, squeeze=False)
        for axis in axes.flat:
            axis.set_visible(False)
        for axis, method in zip(axes.flat, config['centroid_methods']):
            panel = [point for point in subset if point['centroid_method'] == method]
            if not panel:
                axis.set_visible(False)
                continue
            axis.set_visible(True)
            centroid_sizes = {point['centroid_gzip_MB'] for point in panel}
            if len(centroid_sizes) != 1:
                raise ValueError('One centroid index has inconsistent download sizes')
            axis.set_title(f"Centroids: {method.upper()} · {next(iter(centroid_sizes)):.2f} MB initial", fontsize=11)
            for posting, color, marker in zip(METHODS, COLORS, MARKERS):
                curve = sorted((point for point in panel if point['posting_method'] == posting), key=lambda p: p['probes'])
                if not curve:
                    continue
                axis.plot([p['posting_gzip_MB_mean'] for p in curve], [p['recall_percent'] for p in curve],
                          color=color, marker=marker, markersize=4, linewidth=1.2,
                          linestyle='--' if posting == 'fp16' else '-', label=f'Postings: {posting.upper()}')
                for point in curve:
                    axis.annotate(str(point['probes']), (point['posting_gzip_MB_mean'], point['recall_percent']),
                                  fontsize=7, color=color, xytext=(3, 4), textcoords='offset points')
            _style(axis)
        handles = {}
        for axis in axes.flat:
            for handle, label in zip(*axis.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        figure.legend(handles.values(), handles.keys(), loc='lower center', bbox_to_anchor=(.5, .075), ncol=4, frameon=False)
        figure.suptitle(f'{prefix}{heads:,} clusters · centroid × posting quantization', fontweight='bold')
        figure.text(.5, .93, subtitle, ha='center')
        figure.text(.055, .035, footnote, fontsize=8)
        figure.text(.055, .014, limitation, fontsize=8)
        figure.subplots_adjust(left=.075, right=.98, top=.86, bottom=.19, hspace=.35, wspace=.15)
        stem = f'heads_{heads}'
        _save(figure, output, stem)
        figures[stem] = f'{heads:,} clusters: all centroid/posting pairs'

    figure, axis = plt.subplots(figsize=(13, 7))
    for index, heads in enumerate(config['head_targets']):
        subset = sorted((point for point in points if point['target_heads'] == heads),
                        key=lambda p: (p['posting_gzip_MB_mean'], -p['recall_percent']))
        best, frontier = -1, []
        for point in subset:
            if point['recall_percent'] > best:
                frontier.append(point)
                best = point['recall_percent']
        if frontier:
            axis.plot([p['posting_gzip_MB_mean'] for p in frontier], [p['recall_percent'] for p in frontier],
                      color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)], markersize=4,
                      label=f'{heads:,} clusters')
    _style(axis)
    axis.legend(frameon=False)
    figure.suptitle(f'{prefix}Best measured recall versus posting download', fontweight='bold')
    figure.text(.5, .92, subtitle, ha='center')
    figure.text(.055, .06, 'Each frontier combines all centroid/posting pairs. Exact settings are in the per-count plots and CSV.', fontsize=9)
    figure.text(.055, .03, limitation, fontsize=8)
    figure.subplots_adjust(top=.86, bottom=.2)
    _save(figure, output, 'frontier')
    figures['frontier'] = 'Combined accuracy/download frontier'

    if selected_heads is not None:
        figures.update(_selection_plots(config, points, output, selected_heads, selected_centroid, prefix, subtitle))

    links = ' · '.join(f"<a href='#{stem}'>{html.escape(title)}</a>" for stem, title in figures.items())
    sections = ''.join(f"<section id='{stem}'><h2>{html.escape(title)}</h2><img src='{stem}.png' alt='{html.escape(title)}'>"
                       f"<p><a href='{stem}.pdf'>PDF</a> · <a href='{stem}.svg'>SVG</a></p></section>" for stem, title in figures.items())
    (output / 'index.html').write_text('<!doctype html><meta charset=utf-8><title>Joint quantization comparison</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 16px}img{width:100%}</style>'
        f"<h1>{prefix}Cluster count × centroid codec × posting codec</h1><p>{html.escape(subtitle)}</p>"
        f"<p>{pair_count} measured configurations; {len(points)} method/probe measurements. Curve labels are probe counts.</p>"
        "<p><a href='../results.csv'>CSV</a> · <a href='../results.json'>JSON</a></p>"
        f'<nav>{links}</nav>{sections}<p>{html.escape(footnote)}</p><p>{html.escape(limitation)}</p>')
    return {'figures': list(figures), 'points': len(points), 'configurations': pair_count}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--selected-heads', type=int, help='Also plot one fixed index and its count/codec tradeoffs')
    parser.add_argument('--selected-centroid', choices=METHODS, default='lvq6')
    args = parser.parse_args()
    print(json.dumps(render(args.directory, selected_heads=args.selected_heads,
                            selected_centroid=args.selected_centroid), indent=2))
