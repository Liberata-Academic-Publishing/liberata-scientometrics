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
                                        portfolio_normalized_entropy, query_portfolio_mix, get_per_manuscript_cap, \
                                        role_based_proportional_loss

# BASE_DIR = 'test_scripts/output/m50_c100_20251126_224706'
# BASE_DIR = 'test_scripts/output/m50_c100_20251201_180948'
BASE_DIR = 'test_scripts/output/m800_c1200_20260120_205627'

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
    print('Total manuscripts:', M, '\nTotal contributors:', C)

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
        academic_cap = academic_capital(capital, subset_con_map)
        allocation_w = allocation_weights(capital, subset_con_map)
        hhi = portfolio_hhi(capital, subset_con_map)
        gini = portfolio_gini(capital, subset_con_map)
        normalized_entropy = portfolio_normalized_entropy(capital, subset_con_map)
        portfolio_mix = query_portfolio_mix(capital = capital, 
                                            mask_authors = alg_data.authors_slice, 
                                            mask_reviewers = alg_data.reviewers_slice, 
                                            mask_replicators = alg_data.replicators_slice,
                                            manuscript_memberships= alg_data.primary_tag_memberships,
                                            contributor_index_map_subset = subset_con_single,
                                            by= 'role'
                                            )
        mix_by_role = portfolio_mix[0]
        mix_by_tag_per_role = portfolio_mix[1]
        rewviewer_proportional_loss = role_based_proportional_loss(
                                            capital=capital,
                                            retractions_capital = alg_data.retractions_capital,
                                            mask_by_role = alg_data.reviewers_slice,
                                            contributor_index_map_subset = subset_con_map,
                                        )
        replicator_proportional_loss = role_based_proportional_loss(
                                            capital=capital,
                                            retractions_capital = alg_data.retractions_capital,
                                            mask_by_role = alg_data.replicators_slice,
                                            contributor_index_map_subset = subset_con_map,
                                        )
        
    except Exception as e:
        print('Exception raised:', e)
        raise

    print('\nResult from academic_capital():', academic_cap)
    print('Result from portfolio_hhi()', hhi)
    print('Result from portfolio_gini()', gini)
    print('Result from portfolio_normalized_entropy()', normalized_entropy)
    print('Result from portfolio_mix() by role:')
    print('  mix by role:', mix_by_role)
    print('  mix by tag per role:', [i.toarray().squeeze() for i in mix_by_tag_per_role])
    print('\nResult from role_based_proportional_loss():')
    print('  reviewer proportional loss:', rewviewer_proportional_loss)
    print('  replicator proportional loss:', replicator_proportional_loss)


if __name__ == "__main__":
    main()