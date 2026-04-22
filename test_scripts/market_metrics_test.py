import sys
from pathlib import Path
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt

from data_loading import load_data, select_contributor_subset, select_manuscript_subset, liberata_algo_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics import compute_fair_marketprice, compute_risk_premiums

# BASE_DIR = 'test_scripts/output/m50_c100_20251126_224706'
# BASE_DIR = 'test_scripts/output/m50_c100_20251201_180948'
BASE_DIR = 'test_scripts/output/m5000_c20000_20251204_002212'

# select subset of contributor and manuscript IDs
SELECTED_CONTRIBUTORS = {}
SELECTED_MANUSCRIPTS = None

import sys
from pathlib import Path
import numpy as np
from scipy import sparse

from data_loading import load_data, select_contributor_subset, select_manuscript_subset, liberata_algo_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liberata_metrics.metrics import academic_capital, allocation_weights, portfolio_hhi, portfolio_gini, \
                                        portfolio_normalized_entropy, query_portfolio_mix, get_per_manuscript_cap

# BASE_DIR = 'test_scripts/output/m50_c100_20251126_224706'
# BASE_DIR = 'test_scripts/output/m50_c100_20251201_180948'
BASE_DIR = 'test_scripts/output/m5000_c8000_20251204_005806'

# select subset of contributor and manuscript IDs
SELECTED_CONTRIBUTORS = {}
SELECTED_MANUSCRIPTS = None



def main():
    print('Project root:', PROJECT_ROOT)
    print('Using src dir:', SRC_DIR)
    print('Base data dir:', BASE_DIR)

    figs_dir = Path(BASE_DIR) / 'figs'

    # load data
    alg_data  = load_data(BASE_DIR)
    capital = alg_data.capital
    contributor_index_map = alg_data.contributor_index_map
    manuscript_index_map = alg_data.manuscript_index_map

    print('\nLoaded current capital matrix shape:', capital.shape, 'nnz:', capital.nnz)
    M = int(capital.shape[0])
    C = int(capital.shape[1])-M
    size_zero_block = M if C % 3 == 0 else 0
    num_contributors = C // 3 if C % 3 == 0 else C
    print('Total manuscripts:', M, '\nTotal contributors:', num_contributors)

    # select subset of contributor IDs
    if SELECTED_CONTRIBUTORS:
        subset_con_map = select_contributor_subset(contributor_index_map, selected_ids=SELECTED_CONTRIBUTORS)
    else:
        subset_con_map = select_contributor_subset(contributor_index_map, n_first=1000)
    
    subset_con_single = select_contributor_subset(contributor_index_map, n_first=1)

    print('\nSelected contributors:')
    for con_id, idx in subset_con_map.items():
            print(f"  {con_id}: {idx}")
    
    print('\nSelected single contributor:')
    for con_id, idx in subset_con_single.items():
            print(f"  {con_id}: {idx}")

    # select subset of manuscript IDs (NOT REALLY USED RIGHT NOW)
    if SELECTED_MANUSCRIPTS:
        subset_man_map = select_manuscript_subset(manuscript_index_map, selected_ids=SELECTED_MANUSCRIPTS)
        print('\nSelected manuscripts:')
        for man_id, idx in subset_man_map.items():
                print(f"  {man_id}: {idx}")
    else: 
        subset_man_map = select_manuscript_subset(manuscript_index_map, n_first=5)

    # test metric functions
    try:
        reviewer_fmp, replicator_fmp = compute_fair_marketprice(
            capital=capital,
            mask_reviewers=alg_data.reviewers_slice,
            mask_replicators=alg_data.replicators_slice,
            manuscript_memberships=alg_data.primary_tag_memberships,
            contributor_index_map_subset=contributor_index_map,
            num_contributors=num_contributors,
            # size_zero_block=size_zero_block,
        )
        review_risk_premiums, replicator_risk_premiums = compute_risk_premiums(
            capital=capital,
            mask_authors=alg_data.authors_slice,
            mask_reviewers=alg_data.reviewers_slice,
            mask_replicators=alg_data.replicators_slice,
            manuscript_memberships=alg_data.primary_tag_memberships,
            contributor_index_map_subset=subset_con_map,
            reviewer_fmp=reviewer_fmp,
            replicator_fmp=replicator_fmp,
            # num_contributors=num_contributors,
            # size_zero_block=size_zero_block,
        )
        
    except Exception as e:
        print('Exception raised:', e)
        raise

    # print('\nResult from academic_capital():', academic_cap)
    # print('Result from portfolio_hhi()', hhi)
    # print('Result from portfolio_gini()', gini)
    # print('Result from portfolio_normalized_entropy()', normalized_entropy)
    # print('Result from portfolio_mix() by role:')
    # print('  mix by role:', mix_by_role)
    # print('  mix by tag per role:', [i.toarray().squeeze() for i in mix_by_tag_per_role])

    print('\nResult from compute_fair_marketprice():')
    print('  reviewer fair market prices:', reviewer_fmp.squeeze())
    print('  replicator fair market prices:', replicator_fmp.squeeze())
    print('\nResult from compute_risk_premiums():')
    print('  reviewer risk premiums:', review_risk_premiums.squeeze())
    print('  replicator risk premiums:', replicator_risk_premiums.squeeze())

    fig,ax = plt.subplots(1,2, figsize=(12,5))
    x_data = np.arange(capital.shape[0])
    capital = capital.tocsc()
    total_capital = capital[alg_data.authors_slice] + capital[alg_data.reviewers_slice] + capital[alg_data.replicators_slice]
    ax[0].stem(
            x_data, 
            total_capital.getcol(list(subset_con_single.values())[0]).toarray().squeeze(), 
            linefmt='b-', # Line color and style
            markerfmt='bo', # Marker color and style (the lollipop head)
            basefmt=' '
            )
    ax[0].set_title('Capital Distribution \n for Contributor #: ' + str(list(subset_con_single.values())[0]))
    ax[0].set_ylabel('Capital')
    ax[0].set_xlabel('Manuscript #')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1].stem(
            x_data,
            capital[alg_data.reviewers_slice].getcol(list(subset_con_single.values())[0]).toarray().squeeze(), 
            linefmt='C1--',         # Dashed line for reviewer
            markerfmt='D',          # Diamond marker for clarity
            basefmt=' ',
            label='Reviewer Capital',
            # use_line_collection=True
            )
    ax[1].stem(
            x_data,
            capital[alg_data.replicators_slice].getcol(list(subset_con_single.values())[0]).toarray().squeeze(), 
            linefmt='C2-',          # Solid line for replicator
            markerfmt='o',          # Circle marker
            basefmt=' ',
            label='Replicator Capital',
            # use_line_collection=True
            )
    ax[1].stem(
            x_data,
            capital[alg_data.authors_slice].getcol(list(subset_con_single.values())[0]).toarray().squeeze(), 
            linefmt='r:',           # Dotted line for combined (Red)
            markerfmt='x',          # 'x' marker for combined
            basefmt=' ',
            label='Author Capital', # New label for the legend
            # use_line_collection=True
            )
    ax[1].set_title('Capital split by roles \n for Contributor #: ' + str(list(subset_con_single.values())[0]))
    ax[1].set_ylabel('Capital')
    ax[1].set_xlabel('Manuscript #')
    ax[1].legend()
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    fig.savefig(figs_dir / 'single_contributor_capital_distribution.png')
    plt.close(fig)

    

if __name__ == "__main__":
    main()