from __future__ import annotations
from typing import Dict, Callable, List
from pathlib import Path
import numpy as np
from scipy import sparse
from datetime import date, timedelta


# ==================== for plotting ====================

def coo_to_binned_array(
    coo: sparse.coo_matrix, 
    max_side: int, 
    agg_fn: Callable = np.sum
) -> np.ndarray:
    '''utility function to convert COO to binned array'''

    coo = coo.tocoo()
    n_rows, n_cols = coo.shape

    if n_rows <= max_side and n_cols <= max_side:
        return coo.toarray().astype(float)

    # create bin edges
    row_bins = np.linspace(0, n_rows, num=max_side + 1, dtype=int)
    col_bins = np.linspace(0, n_cols, num=max_side + 1, dtype=int)

    # map each nonzero to a bin index
    bin_rows = np.searchsorted(row_bins, coo.row, side="right")-1
    bin_cols = np.searchsorted(col_bins, coo.col, side="right")-1
    bin_rows = np.clip(bin_rows, 0, max_side-1)
    bin_cols = np.clip(bin_cols, 0, max_side-1)

    flat_idx = bin_rows * max_side + bin_cols
    sums = np.bincount(flat_idx, weights=coo.data.astype(float), minlength=max_side * max_side)
    agg = sums.reshape((max_side, max_side))
    return agg.astype(float)



def matrix_to_plot_array(mat: sparse.spmatrix, max_side: int = 500, agg_fn: Callable = np.sum) -> np.ndarray:
    '''convert any COO matrix into appropriate array form for plotting (note: this is all ChatGPT)'''

    if sparse.issparse(mat):
        coo = mat.tocoo()
        return coo_to_binned_array(coo, max_side=max_side, agg_fn=agg_fn)
    
    arr = np.asarray(mat, dtype=float)
    n_rows, n_cols = arr.shape
    if n_rows <= max_side and n_cols <= max_side:
        return arr
    
    # aggregate dense matrices to bins
    row_bins = np.linspace(0, n_rows, num=max_side + 1, dtype=int)
    col_bins = np.linspace(0, n_cols, num=max_side + 1, dtype=int)
    agg = np.zeros((max_side, max_side), dtype=float)
    for i in range(max_side):
        r0, r1 = row_bins[i], row_bins[i + 1]
        for j in range(max_side):
            c0, c1 = col_bins[j], col_bins[j + 1]
            block = arr[r0:r1, c0:c1]
            if block.size:
                agg[i, j] = agg_fn(block)
    return agg



# ==================== for encoding upload dates ====================

def serialize_upload_dates(upload_dates: Dict[str, date]) -> Dict[str, str]:
    '''convert date objects to ISO strings YYYY-MM-DD'''
    return {id: dt.isoformat() for id, dt in upload_dates.items()}


def deserialize_upload_dates(upload_dates_json: Dict[str, str]) -> Dict[str, date]:
    '''convert ISO strings YYYY-MM-DD to date objects'''
    return {id: date.fromisoformat(dt_str) for id, dt_str in upload_dates_json.items()}


def make_date_grid(start_date: date, end_date: date, time_step: timedelta) -> List[date]:
    '''build a grid of dates from start_date to end_date with given time_step increments for time series data extraction'''

    # validation
    if start_date > end_date:
        raise ValueError('start_date must be before end_date')
    if not isinstance(time_step, timedelta):
        raise TypeError('time_step must be a datetime.timedelta object')
    if time_step.days < 1:
        # enforce smallest unit of 1 day corresponding to database refresh rate
        raise ValueError('time_step must be at least one day')

    grid: List[date] = []
    current = start_date
    while current <= end_date:
        grid.append(current)
        current = current + time_step
    return grid