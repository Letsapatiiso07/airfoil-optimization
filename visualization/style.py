"""Shared matplotlib look for the README figures."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
BLUE = '#2a78d6'      # our model
ORANGE = '#eb6834'
AQUA = '#1baf7a'
VIOLET = '#6250d6'


def apply():
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': INK_2,
        'axes.titlecolor': INK,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.labelsize': 9.5,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'xtick.color': INK_2,
        'ytick.color': INK_2,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'lines.linewidth': 2,
        'lines.solid_capstyle': 'round',
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.fontsize': 8.5,
        'legend.labelcolor': INK_2,
        'font.family': 'DejaVu Sans',
    })


def marker(color, filled=True):
    """Experimental data points: round markers with a surface-colour ring."""
    return dict(linestyle='none', marker='o', markersize=6,
                markerfacecolor=color if filled else SURFACE,
                markeredgecolor=SURFACE if filled else color,
                markeredgewidth=1.2 if filled else 1.6)
