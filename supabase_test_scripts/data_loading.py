from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from scipy import sparse
from datetime import date
from dataclasses import dataclass, field

@dataclass(frozen=True)
class liberata_algo_data:
    capital:  sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    contributor_index_map: Dict[str, int] = field(default_factory=dict)
    manuscript_index_map: Dict[str, int] = field(default_factory=dict)
    all_tag_memberships: sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    primary_tag_memberships: sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    authors_mask: sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    reviewers_mask: sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    replicators_mask: sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    retractions: sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    retractions_capital:  sparse.spmatrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    topic_index_map: Dict[str, int] = field(default_factory=dict)
    authors_slice: Any = None
    reviewers_slice: Any = None
    replicators_slice: Any = None

    def __getitem__(self, key:int):
        field_values = (self.capital, self.contributor_index_map, self.manuscript_index_map,
                        self.all_tag_memberships, self.primary_tag_memberships,
                        self.authors_mask, self.reviewers_mask, self.replicators_mask,
                        self.retractions, self.retractions_capital, self.topic_index_map,
                        self.authors_slice, self.reviewers_slice, self.replicators_slice)
        return field_values[key]


def load_data(base_dir: str) -> Tuple[
    sparse.spmatrix, Dict[str, int], Dict[str, int],
    sparse.spmatrix, Dict[str, int], Dict[str, int],
]:
    '''data loader for local testing'''

    base = Path(base_dir)

    # current data
    capital_path = base / "capital_coo.npz"
    contributor_map_path = base / "contributor_index_map.json"
    manuscript_map_path = base / "manuscript_index_map.json"
    all_tag_memberships_path = base / "all_memberships_tags.npz"
    primary_tag_memberships_path = base / "primary_memberships_tags.npz"
    # authors_mask_path = base / "mask_authors_coo.npz"
    # reviewers_mask_path = base / "mask_reviewers_coo.npz"
    # replicators_mask_path = base / "mask_replicators_coo.npz"
    retractions_path = base / "retractions_coo.npz"
    retractions_capital_path = base / "retractions_capital_coo.npz"
    topic_index_map_path = base / "topic_index_map.json"

    if not capital_path.exists():
        raise FileNotFoundError(f"Capital file not found: {capital_path}")
    if not contributor_map_path.exists():
        raise FileNotFoundError(f"Contributor index map not found: {contributor_map_path}")
    if not manuscript_map_path.exists():
        raise FileNotFoundError(f"Manuscript index map not found: {manuscript_map_path}")
    if not all_tag_memberships_path.exists():
        raise FileNotFoundError(f"All tag memberships file not found: {all_tag_memberships_path}")
    if not primary_tag_memberships_path.exists():
        raise FileNotFoundError(f"Primary tag memberships file not found: {primary_tag_memberships_path}")
    # if not authors_mask_path.exists():
    #     raise FileNotFoundError(f"Authors mask file not found: {authors_mask_path}")
    # if not reviewers_mask_path.exists():
    #     raise FileNotFoundError(f"Reviewers mask file not found: {reviewers_mask_path}")
    # if not replicators_mask_path.exists():
    #     raise FileNotFoundError(f"Replicators mask file not found: {replicators_mask_path}")
    if not retractions_path.exists():
        raise FileNotFoundError(f"Retractions file not found: {retractions_path}")
    if not retractions_capital_path.exists():
        raise FileNotFoundError(f"Retractions capital file not found: {retractions_capital_path}")
    if not topic_index_map_path.exists():
        raise FileNotFoundError(f"Topic index map not found: {topic_index_map_path}")

    # load current capital matrix and index mappings
    capital = sparse.load_npz(str(capital_path))
    all_tag_memberships = sparse.load_npz(str(all_tag_memberships_path))
    primary_tag_memberships = sparse.load_npz(str(primary_tag_memberships_path))
    # authors_mask = sparse.load_npz(str(authors_mask_path))
    # reviewers_mask = sparse.load_npz(str(reviewers_mask_path))
    # replicators_mask = sparse.load_npz(str(replicators_mask_path))
    retractions = sparse.load_npz(str(retractions_path))
    retractions_capital = sparse.load_npz(str(retractions_capital_path))
    with open(contributor_map_path, "r", encoding="utf-8") as fh:
        contributor_index_map = json.load(fh)
    with open(manuscript_map_path, "r", encoding="utf-8") as fh:
        manuscript_index_map = json.load(fh)
    with open(topic_index_map_path, "r", encoding="utf-8") as fh:
        topic_index_map = json.load(fh)


    # cast to ints
    contributor_index_map = {k: int(v) for k, v in contributor_index_map.items()}
    manuscript_index_map = {k: int(v) for k, v in manuscript_index_map.items()}
    num_contributors = len(contributor_index_map)

    M = int(capital.shape[0])
    C = int(capital.shape[1])-M
    size_zero_block = M if C > num_contributors else 0
    authors_slice = np.s_[:,size_zero_block : size_zero_block + num_contributors]
    reviewers_slice = np.s_[:, size_zero_block + num_contributors: size_zero_block + 2*num_contributors]
    replicators_slice = np.s_[:, size_zero_block + 2*num_contributors:] 


    # return capital, contributor_index_map, manuscript_index_map, all_tag_memberships, primary_tag_memberships, \
    #         authors_mask, reviewers_mask, replicators_mask, retractions, retractions_capital
    return liberata_algo_data(
        capital=capital,
        contributor_index_map=contributor_index_map,
        manuscript_index_map=manuscript_index_map,
        all_tag_memberships=all_tag_memberships,
        primary_tag_memberships=primary_tag_memberships,
        # authors_mask=authors_mask,
        # reviewers_mask=reviewers_mask,
        # replicators_mask=replicators_mask,
        retractions=retractions,
        retractions_capital=retractions_capital,
        topic_index_map=topic_index_map,
        authors_slice=authors_slice,
        reviewers_slice=reviewers_slice,
        replicators_slice=replicators_slice,
    )



def select_contributor_subset(
    contributor_index_map: Dict[str, int],
    selected_ids: Optional[List[str]] = None,
    n_first: Optional[int] = 5,
) -> Dict[str, int]:
    '''selects some subset of contributor indices to act as a portfolio'''
    
    # choose indices from argument
    if selected_ids:
        subset = {}
        for con_id in selected_ids:
            if con_id not in contributor_index_map:
                raise KeyError(f'Contributor id <{con_id}> not found in contributor_index_map')
            subset[con_id] = int(contributor_index_map[con_id])
        return subset

    keys = list(contributor_index_map.keys())
    if len(keys) == 0:
        return {}

    # choose first n by default if no specific IDs provided
    chosen = keys[: max(1, int(n_first))]
    return {con_id: int(contributor_index_map[con_id]) for con_id in chosen}



def select_manuscript_subset(
    manuscript_index_map: Dict[str, int],
    selected_ids: Optional[List[str]] = None,
    n_first: Optional[int] = 5,
) -> Dict[str, int]:
    '''selects some subset of manuscript indices to act as a portfolio'''

    # choose indices from argument
    if selected_ids:
        subset = {}
        for man_id in selected_ids:
            if man_id not in manuscript_index_map:
                raise KeyError(f'Manuscript id <{man_id}> not found in manuscript_index_map')
            subset[man_id] = int(manuscript_index_map[man_id])
        return subset

    keys = list(manuscript_index_map.keys())
    if len(keys) == 0:
        return {}
    
    # choose first n by default if no specific IDs provided
    chosen = keys[: max(1, int(n_first))]
    return {man_id: int(manuscript_index_map[man_id]) for man_id in chosen}




    