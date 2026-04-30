import sys
from pathlib import Path
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics.distribution_metrics import hhi_discrepancy, share_splits_inequality


def main():
    print('Testing distribution metrics from distribution_metrics.py')

    # synthetic portfolio: 3 manuscripts x 4 authors; rows sum to 1
    portfolio_dense = np.array([
        [0.5, 0.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    portfolio = sparse.csr_matrix(portfolio_dense)
    print(portfolio_dense)
    print('Portfolio shape:', portfolio.shape, 'nnz:', portfolio.nnz)

    mean_hhi = share_splits_inequality(portfolio)
    print('share_splits_inequality:', mean_hhi)

    field_hhi = 0.4
    discrepancy = hhi_discrepancy(portfolio, [1])
    print(f'hhi_discrepancy(manuscript={1}):', discrepancy)

    # Ensure function enforces sparse input
    try:
        share_splits_inequality(portfolio_dense)
    except TypeError as e:
        print('Expected TypeError for dense input:', e)
    else:
        print('ERROR: expected TypeError for dense input but none raised')


if __name__ == '__main__':
    main()