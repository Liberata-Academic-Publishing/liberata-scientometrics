from __future__ import annotations
from typing import Optional, Iterable, List, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from liberata_metrics.utils import matrix_to_plot_array

def matrix_heatmap(
    mat: sparse.spmatrix,
    out_path: str | Path,
    title: Optional[str] = None,
    cmap: str = "viridis",
    max_side: int = 500,
    agg_fn: Callable = np.sum,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    dpi: int = 150,
) -> None:
    '''plot matrix heatmap (note: this is all ChatGPT)'''
    arr = matrix_to_plot_array(mat, max_side=max_side, agg_fn=agg_fn)

    fig, ax = plt.subplots(figsize=(8, 8 * arr.shape[0] / max(1, arr.shape[1])))
    im = ax.imshow(arr, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_xlabel("columns")
    ax.set_ylabel("rows")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("value", rotation=270, labelpad=12)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi)
    plt.close(fig)