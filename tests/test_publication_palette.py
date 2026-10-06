"""Publication identity and language invariance of the primary FULL figures."""
from copy import deepcopy
import unittest

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

import full_plot_style as style
from reporting import figures
from tests.test_generic_figures import report_fixture


def data_artists(fig):
    """Include inset artists, excluding localized text and layout positions."""
    def axes_artists(ax):
        return (
            [(line.get_color(), line.get_linewidth(), line.get_linestyle(),
              line.get_marker(), line.get_zorder(), np.asarray(line.get_xdata()).tolist(),
              np.asarray(line.get_ydata()).tolist()) for line in ax.lines],
            [(p.get_facecolor(), p.get_edgecolor(), p.get_linewidth(),
              p.get_alpha(), p.get_zorder(), p.get_path().vertices.tolist()) for p in ax.patches],
            [(c.get_facecolors().tolist(), c.get_edgecolors().tolist(),
              c.get_linewidths().tolist(), c.get_alpha(), c.get_zorder()) for c in ax.collections],
            [(im.get_array().tolist(), im.get_clim(), im.get_cmap().name) for im in ax.images],
            [axes_artists(child) for child in ax.child_axes],
        )
    return [axes_artists(ax) for ax in fig.axes]


class PublicationPaletteTests(unittest.TestCase):
    def test_fixed_unique_colors_resolve_canonical_names_and_order(self):
        expected = style.PUBLICATION_OPTIMIZER_COLORS
        self.assertEqual(len(expected), 14)
        self.assertEqual(len(set(expected.values())), 14)
        self.assertEqual(expected['MaCRO-DE'], '#19365F')
        names = ['MaCRO-DE', 'OriginalGWO', 'DevDMOA', 'OriginalFLA', 'BaseDE']
        self.assertEqual(list(style.palette(names).values()),
                         [expected[a] for a in ('MaCRO-DE', 'GWO', 'DMOA', 'FLA', 'DE')])
        self.assertEqual(style.palette(names), style.palette(names[::-1]))

    def test_primary_figures_change_only_text_with_language(self):
        report = report_fixture(('A', 'B'), tuple(style.PUBLICATION_OPTIMIZER_COLORS), ('knn', 'svm'))
        original = deepcopy(report.indexed)
        snapshots, texts = [], []
        for language in ('en', 'es'):
            report.args.figure_language = language
            observed, titles = {}, []
            for stem, fig in figures.base_publication_figures(report, []):
                try:
                    observed[stem] = data_artists(fig)
                    titles.append(fig.axes[0].get_title())
                    ax = fig.axes[0]
                    if stem.startswith('01'):
                        self.assertEqual(to_hex(ax.patches[0].get_facecolor()), '#19365f')
                        self.assertEqual(ax.patches[0].get_linewidth(), 2.8)
                    if stem.startswith(('02', '05')):
                        primary = ax.lines[-1]
                        self.assertEqual(to_hex(primary.get_color()), '#19365f')
                        self.assertGreater(primary.get_linewidth(), ax.lines[0].get_linewidth())
                        self.assertGreater(primary.get_zorder(), ax.lines[0].get_zorder())
                        if stem.startswith('05'):
                            self.assertEqual(to_hex(ax.child_axes[0].lines[-1].get_color()), '#19365f')
                    if stem.startswith('06'):
                        self.assertEqual(ax.patches[0].get_linewidth(), 2.8)
                        self.assertFalse(ax.patches[0].get_clip_on())
                finally:
                    plt.close(fig)
            snapshots.append(observed)
            texts.append(titles)
        self.assertEqual(len(snapshots[0]), 9)
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertNotEqual(texts[0], texts[1])
        np.testing.assert_equal(report.indexed, original)


if __name__ == '__main__':
    unittest.main()
