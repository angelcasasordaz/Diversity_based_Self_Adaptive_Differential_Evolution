"""Shared visual primitives from the numbered FULL figures (no data processing)."""
import hashlib
import math

import matplotlib.pyplot as plt
import numpy as np
from plot_labels import plot_display_label


STYLE_ID = 'numbered-full-v1-neutral-text'
PUBLICATION_OPTIMIZER_COLORS = {
    'MaCRO-DE': '#19365F',
    'BRO': '#1B9E77', 'DBO': '#D95F02', 'DE': '#E66101',
    'DMOA': '#7570B3', 'GWO': '#E7298A', 'HHO': '#C51B7D',
    'MFO': '#66A61E', 'MGO': '#4DAF4A', 'PSO': '#E6AB02',
    'SHADE': '#A6761D', 'WOA': '#8C510A', 'JADE': '#666666',
    'FLA': '#999999',
}
OPTIMIZER_COLORS = {
    'DSADE': '#0072B2', 'DE': '#009E73', 'JADE': '#E69F00',
    'SHADE': '#CC79A7', 'PSO': '#56B4E9', 'WOA': '#D55E00',
    'HHO': '#6A3D9A', 'GOA': '#8DAA00', 'SA': '#F0C808',
    'BRO': '#A65628', 'RUN': '#4D4D4D', 'FOX': '#999999',
}
STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'figure.dpi': 100, 'savefig.dpi': 600,
    'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'grid.linewidth': .8,
}
HEADER_STYLES = (('#d8e8f3', '#b8d3e6'), ('#d2efee', '#abd9d7'),
                 ('#f7efd8', '#ead9ad'), ('#f9d5d9', '#edaeb8'))


def method_key(name):
    name = str(name).strip().upper()
    return 'DSADE' if name in {'DSADE', 'DSA-DE', 'DSA_DE', 'MACRO-DE-T'} else name


def display_label(name, present_methods=()):
    if str(name).strip().upper() == 'MACRO-DE-T':
        return plot_display_label(name, present_methods)
    return 'DSA-DE' if method_key(name) == 'DSADE' else plot_display_label(name, present_methods)


def method_index(name):
    key = method_key(name)
    if key in OPTIMIZER_COLORS:
        return list(OPTIMIZER_COLORS).index(key)
    # Unknown algorithms keep a stable style when the selected order changes.
    return len(OPTIMIZER_COLORS) + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def palette(algorithms):
    """Fixed publication colors by identity, independent of order and language."""
    colors = {}
    for name in algorithms:
        key = publication_key(name)
        if key in PUBLICATION_OPTIMIZER_COLORS:
            colors[name] = PUBLICATION_OPTIMIZER_COLORS[key]
        else:
            # Generic reports may contain other methods; never cycle known colors.
            colors[name] = '#' + hashlib.sha256(key.encode()).hexdigest()[:6]
    return colors


def publication_key(name):
    key = str(name).strip().upper()
    for prefix in ('ORIGINAL', 'BASE', 'DEV'):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key in {'MACRO-DE', 'MACRO-DE-T', 'DSADE', 'DSA-DE', 'DSA_DE'}:
        return 'MaCRO-DE'
    return key


def is_primary(name):
    return publication_key(name) == 'MaCRO-DE'


def line_style(name):
    index = method_index(name)
    return {'linestyle': ('-', '--', ':', '-.')[index % 4],
            'marker': ('o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'p')[index % 12]}


def grid_shape(count):
    columns = min(4, max(1, math.ceil(math.sqrt(max(1, count)))))
    return math.ceil(max(1, count) / columns), columns


def figure_size(kind, algorithms, datasets=0):
    """Dimensions taken from FULL figures 06, 07, 08 and 09."""
    if kind == 'heatmap':
        return max(10, .9*datasets+4), max(5, .45*algorithms+2)
    if kind == 'violin':
        return max(12, .85*algorithms+5), 6.5
    if kind == 'boxplot':
        return max(12, .8*algorithms+5), 6
    if kind == 'features_runtime':
        return 12, 6
    raise ValueError(f'Unknown FULL figure geometry: {kind}')


def style_axes(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', alpha=.25, color=STYLE['grid.color'],
            linestyle=STYLE['grid.linestyle'], linewidth=STYLE['grid.linewidth'])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    ax.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')


def highlight_patch(patch, method, linewidth=2.8):
    if is_primary(method):
        patch.set_edgecolor('black')
        patch.set_linewidth(linewidth)


def metric_header(ax, label, index):
    face, edge = HEADER_STYLES[index % len(HEADER_STYLES)]
    ax.set_title(label, fontsize=12, fontweight='bold', color='black', pad=12,
                 bbox=dict(boxstyle='round,pad=0.22', facecolor=face, edgecolor=edge))


def metric_bars(ax, values, algorithms, colors):
    """The numbered FULL summary's bars, mean guide, and value labels."""
    bars = ax.bar(np.arange(len(algorithms)), values, width=.68,
                  color=[colors[a] for a in algorithms], edgecolor='none')
    for bar, method, value in zip(bars, algorithms, values):
        highlight_patch(bar, method)
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width()/2, value + .006, f'{value:.3f}',
                    ha='center', va='bottom', fontsize=6.5, rotation=90, color='black')
    if np.isfinite(values).any():
        ax.axhline(np.nanmean(values), color='#d76c6c', linestyle='--', linewidth=.9, alpha=.8)
    return bars


def neutral_text(fig):
    """Black non-data text; white heatmap labels retain their contrast exception."""
    from matplotlib.text import Text
    from matplotlib.colors import to_rgba
    for text in fig.findobj(match=Text):
        if to_rgba(text.get_color())[:3] != (1., 1., 1.):
            text.set_color('black')
