import pandas            as pd
import matplotlib.pyplot as plt
import re

from matplotlib import rcParams, rc_context
from typing     import Optional


# %%  Global Setting

DEFAULT_RC = {
    # --- fonts / text ---
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "text.usetex": False,

    # --- axes / lines ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "lines.linewidth": 1.5,

    # --- ticks ---
    "xtick.direction": "in",
    "ytick.direction": "in",

    # --- figure / saving ---
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

def configure(rc_updates: Optional[dict] = None) -> None:
    """Apply global plotting styles for the whole paper."""
    plt.style.use('default')
    rcParams.update(DEFAULT_RC if rc_updates is None else DEFAULT_RC | rc_updates)

# style_name:
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
# print(plt.style.available)
#
# import matplotlib as mpl
# def configure(rc_updates: Optional[dict] = None, 
#                style_name: str = "ggplot"
#               ) -> None:
#     """Apply global plotting styles for the whole paper."""
#     style_colors = mpl.style.library[style_name]['axes.prop_cycle']
#     DEFAULT_RC['axes.prop_cycle'] = style_colors
#     rcParams.update(DEFAULT_RC if rc_updates is None else DEFAULT_RC | rc_updates)

def reset() -> None:
    """Reset matplotlib to defaults (optional)."""
    plt.rcdefaults()

def fig_ax(figsize=(6, 4)):
    """Convenience helper to create a figure/axes with our defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

class paper_style:
    """
    Context manager for temporary overrides:
        with paper_style({"font.size": 12}):
            ...
    """
    def __init__(self, overrides=None):
        self._ctx = rc_context(overrides or {})
    def __enter__(self): return self._ctx.__enter__()
    def __exit__(self, *args): return self._ctx.__exit__(*args)


# %%  Fix Variable Name

STRATEGY_NAMES = {
    'TB_BT':       'TB / BT',
    'BuyT_BuyB':   'BuyT / BuyB',
    'BuyT_SellB':  'BuyT / SellB',
    'SellB_SellT': 'SellB / SellT',
    'BuyT':        'BuyT',
    'BuyB':        'BuyB',
    'SellT':       'SellT',
    'SellB':       'SellB',
    'BuyTSellB':   'BuyTSellB',
    'SP500':       'S&P 500',
}

def rename_strategy_index(index):
    out = []
    for idx in index:
        s = str(idx)

        if s == 'SP500':
            out.append(STRATEGY_NAMES.get('SP500', s))
            continue

        m = re.search(r'_(\d+)%$', s)
        if m:
            pct = m.group(1) + '%'
            strat_code = s[:m.start()]
            name = STRATEGY_NAMES.get(strat_code, strat_code)
            out.append(f"{name} ({pct})")
        else:
            out.append(STRATEGY_NAMES.get(s, s))
    return out


# %%  Plot Function

# Cumulative Return

def plot_cumulative_return(df: pd.DataFrame,
                           xlabel="",
                           ylabel="",
                           legend_loc="best",
                           figsize=(6, 4)):
    df = df.copy()
    df.index = rename_strategy_index(df.index)
    
    fig, ax = fig_ax(figsize=figsize)
    df.T.plot(ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")
    ax.legend(loc=legend_loc, frameon=False)
    ax.tick_params(axis="x", rotation=0)
    
    fig.tight_layout()    
    return fig, ax








