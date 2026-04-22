from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse


def _select_ids_from_df(
    df: pd.DataFrame,
    requested_ids: Optional[Sequence[str]],
    c: int,
    rng_seed: Optional[int] = None
) -> List[str]:
    '''helper function to select a subset of contributors/manuscripts to plot'''

    cols = list(df.columns)
    if requested_ids:
        # preserve requested order but only keep those present in df
        selected = [rid for rid in requested_ids if rid in cols]
        if len(selected) < len(requested_ids):
            missing = [rid for rid in requested_ids if rid not in cols]
            raise ValueError(f'The following IDs are not valid: {missing}')
        return selected
    
    # return all if c exceeds available
    if c >= len(cols):
        return cols.copy()
    
    # randomly select if requested IDs is empty
    rng = np.random.default_rng(rng_seed)
    return list(rng.choice(cols, size=c, replace=False))



def plot_contributor_time_series(
    contributor_df: pd.DataFrame,
    contributor_ids: Optional[Sequence[str]],
    c: int,
    output_path: Union[str, Path],
    rng_seed: Optional[int] = None
) -> Path:

    x = pd.to_datetime(contributor_df.index)
    # choose which contributor IDs to plot
    plotting_ids = _select_ids_from_df(contributor_df, contributor_ids, c, rng_seed=rng_seed)

    plt.figure(figsize=(10, 5))
    for cid in plotting_ids:
        y = contributor_df[cid].values
        plt.plot(x, y, label=str(cid)[:8], linewidth=1.5)

    plt.title("Contributor Academic Capital Over Time")
    plt.xlabel("Date")
    plt.ylabel("Academic Capital")
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=1)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    return out


def plot_manuscript_time_series(
    manuscript_df: pd.DataFrame,
    manuscript_ids: Optional[Sequence[str]],
    c: int,
    output_path: Union[str, Path],
    rng_seed: Optional[int] = None
) -> Path:

    x = pd.to_datetime(manuscript_df.index)
    ids_to_plot = _select_ids_from_df(manuscript_df, manuscript_ids, c, rng_seed=rng_seed)

    plt.figure(figsize=(10, 5))
    for mid in ids_to_plot:
        y = manuscript_df[mid].values
        plt.plot(x, y, label=str(mid)[:8], linewidth=1.5)

    plt.title("Manuscript Academic Capital Over Time")
    plt.xlabel("Date")
    plt.ylabel("Academic Capital")
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=1)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    return out
