from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from balanced_l2.plot_joint_quantization import METHODS, render


class JointPlotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.saved = {'status': 'complete', 'manifest': {'head_targets': [64, 80],
            'centroid_methods': list(METHODS), 'posting_methods': list(METHODS),
            'probes': [4, 8], 'query_count': 100, 'noise_std': .025}, 'points': []}
        for heads in (64, 80):
            for c, centroid in enumerate(METHODS):
                for p, posting in enumerate(METHODS):
                    for probe in (4, 8):
                        size = probe * (4 - p) / 2
                        self.saved['points'].append({'target_heads': heads, 'actual_heads': heads,
                            'centroid_method': centroid, 'posting_method': posting, 'probes': probe,
                            'queries': 100, 'noise_std': .025, 'recall_percent': 90 + probe / 2 - c - p,
                            'tie_aware_recall_percent': 95, 'routing_coverage_percent': 99,
                            'posting_gzip_MB_mean': size, 'posting_gzip_MB_p95': size * 2,
                            'centroid_gzip_MB': 4 - c / 2, 'cold_gzip_MB_mean': size + 4 - c / 2,
                            'total_posting_gzip_GB': 4 - p / 2})

    def plot(self, saved=None):
        (self.root / 'results.json').write_text(json.dumps(saved or self.saved))
        return render(self.root)

    def test_all_pairs_and_formats_are_present_and_results_preserved(self):
        result = self.plot()
        self.assertEqual(result, {'figures': ['heads_64', 'heads_80', 'frontier'], 'points': 64, 'configurations': 32})
        self.assertEqual(json.loads((self.root / 'results.json').read_text()), self.saved)
        for stem in result['figures']:
            for extension in ('png', 'svg', 'pdf'):
                self.assertGreater((self.root / 'plots' / f'{stem}.{extension}').stat().st_size, 100)
        page = (self.root / 'plots/index.html').read_text()
        self.assertIn('32 measured configurations; 64 method/probe measurements', page)
        self.assertIn('probes 4, 8', page)
        self.assertIn('heads_80.png', page)
        self.assertIn('centroid', page)

    def test_partial_results_are_labeled_and_invalid_measurements_rejected(self):
        saved = deepcopy(self.saved)
        saved['status'] = 'running'
        saved['points'] = [point for point in saved['points'] if point['target_heads'] == 64]
        self.assertEqual(self.plot(saved)['points'], 32)
        self.assertIn('PARTIAL', (self.root / 'plots/index.html').read_text())
        for change in (lambda r: r['points'].pop(),
                       lambda r: r['points'].append(r['points'][0]),
                       lambda r: r['points'][0].update(cold_gzip_MB_mean=1),
                       lambda r: r['points'][0].update(recall_percent=float('nan'))):
            broken = deepcopy(self.saved)
            change(broken)
            with self.assertRaises(ValueError):
                self.plot(broken)

    def test_fixed_index_plots_and_missing_selection(self):
        (self.root / 'results.json').write_text(json.dumps(self.saved))
        result = render(self.root, selected_heads=80, selected_centroid='lvq6')
        for stem in ('fixed_lvq6_counts', 'selected_80_lvq6'):
            self.assertIn(stem, result['figures'])
            for extension in ('png', 'svg', 'pdf'):
                self.assertGreater((self.root / 'plots' / f'{stem}.{extension}').stat().st_size, 100)
        self.assertIn('Tie-aware recall', (self.root / 'plots/selected_80_lvq6.svg').read_text())
        self.assertEqual(json.loads((self.root / 'results.json').read_text()), self.saved)
        with self.assertRaisesRegex(ValueError, 'no measurements'):
            render(self.root, selected_heads=128)


if __name__ == '__main__':
    unittest.main()
