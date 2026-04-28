from __future__ import annotations
import uuid
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional, Iterable, Union
import numpy as np
from scipy import sparse
import pandas as pd
from warnings import warn
from dataclasses import dataclass
from pathlib import Path

from liberata_metrics.utils import _rng, random_date, make_date_grid

OPEN_ALEX_TOPICS_CSV_PATH = Path(__file__).resolve().parents[3]/'data'/'OpenAlex_topic_mapping_table.csv'
MAX_RETRACTION_CUTOFF = 2**63 - 1

@dataclass(frozen=True)
class Manuscript:
    id: str
    upload_date: date
    primary_topic: str
    topics: List[str]
    retraction_cutoff: int

# TODO - Move to an appropriate utils file and import
def build_COO(num_manuscripts: int,
            num_contributors: int,
            data: Union[List, None] = None, 
            rows: Union[List, None] = [],
            cols: Union[List, None] = [],
            dtype: type = float,
            ) -> sparse.spmatrix :
        
    if not data:
        data_block = sparse.coo_matrix(
            (np.array([], dtype=dtype), (np.array([], dtype=int), np.array([], dtype=int))),
            (num_manuscripts, num_contributors)
        )
    else:
        data_block = sparse.coo_matrix(
            (np.array(data, dtype=type(data[0]) ), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
            shape=(num_manuscripts, num_contributors),
        )
    
    return data_block


# ==================== generating matrices from scratch ====================

def generate_references_matrix(
    num_manuscripts: int,
    citation_density: float = 0.05,
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2024, 1, 1),
    seed: Optional[int] = None,
) -> Tuple[sparse.coo_matrix, List[str], Dict[str, int], Dict[str, date], \
           pd.DataFrame, sparse.coo_matrix, sparse.coo_matrix]:
    '''build toy references matrix'''

    rng = _rng(seed)

    # Load OpenAlex topics document
    if OPEN_ALEX_TOPICS_CSV_PATH.exists():
        openalex_topics = pd.read_csv(OPEN_ALEX_TOPICS_CSV_PATH)
    else:
        openalex_topics = pd.DataFrame({
            'topic_name': [f'Topic_{i}' for i in range(100)]
        })
    num_topics = num_manuscripts // 10 if num_manuscripts >= 10 else 1
    # choose num_topics unique topics randomly
    unique_topics = openalex_topics['topic_name'].unique()
    chosen_topics = rng.choice(unique_topics, size=num_topics, replace=False)    

    # randomly generate manuscripts with IDs and upload dates
    # manuscript_meta: List[Tuple[str, date, str, List[str], int]] = []
    manuscript_meta: List[Manuscript] = []
    for _ in range(num_manuscripts):
        id = str(uuid.uuid4())
        upload = random_date(rng, start_date, end_date)
        num_topics_on_manuscript = rng.randint(1, num_topics+1)
        all_topics = rng.choice(chosen_topics, size=num_topics_on_manuscript, replace=False)
        primary_topic = all_topics[0]
        retraction_cutoff = MAX_RETRACTION_CUTOFF  # Using a large number instead of float inf
        manuscript_meta.append(Manuscript(id, upload, primary_topic, all_topics, retraction_cutoff))

    # sort manuscripts by upload date
    manuscript_meta.sort(key = lambda m: (m.upload_date.toordinal(), m.id))
    manuscript_ids = [m.id for m in manuscript_meta]
    upload_dates = {m.id: m.upload_date for m in manuscript_meta}
    manuscript_meta_df = pd.DataFrame([m.__dict__ for m in manuscript_meta])

    # extract manuscript to index mapping
    manuscript_index_map = {id: idx for idx, id in enumerate(manuscript_ids)}

    # topics index mapping
    topic_index_map = {topic: idx for idx, topic in enumerate(chosen_topics)}

    # COO format
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i, citing_id in enumerate(manuscript_ids):
        # choose available indices according to upload time
        idx_avail = list(range(0, i))
        if not idx_avail: continue

        max_citations = int(len(idx_avail)*citation_density)
        num_citations = rng.randint(0, max_citations+1) if max_citations > 0 else 0
        if num_citations == 0: continue

        # choose cited manuscripts and weight
        idx_cited = rng.choice(idx_avail, size=min(num_citations, len(idx_avail)), replace=False)
        w = 1.0/float(num_citations)

        # insert values into matrix
        for idx in np.atleast_1d(idx_cited):
            rows.append(idx)
            cols.append(i)
            data.append(w)
    
    # Construct COO for manuscript memberships
    rows_membership: List[int] = []
    cols_membership: List[int] = []
    data_membership: List[bool] = []

    rows_all_membership: List[int] = []
    cols_all_membership: List[int] = []
    data_all_membership: List[bool] = []

    for m in manuscript_meta:
        primary_topic_idx = topic_index_map[m.primary_topic]
        manuscript_idx = manuscript_index_map[m.id]
        rows_membership.append(primary_topic_idx)
        cols_membership.append(manuscript_idx)
        data_membership.append(True)

        for topic in m.topics:
            topic_idx = topic_index_map[topic]
            rows_all_membership.append(topic_idx)
            cols_all_membership.append(manuscript_idx)
            data_all_membership.append(True)
    
    primary_memberships = build_COO(num_manuscripts=num_topics,
                                  num_contributors=num_manuscripts,
                                  rows=rows_membership,
                                  cols=cols_membership,
                                  data=data_membership)
    
    
    all_memberships = build_COO(num_manuscripts=num_topics,
                                num_contributors=num_manuscripts,
                                rows=rows_all_membership,
                                cols=cols_all_membership,
                                data=data_all_membership,
                                )

    
    # output matrix in COO
    if not data: 
        references = sparse.coo_matrix(
            (np.array([], dtype=float), (np.array([], dtype=int), np.array([], dtype=int))),
            shape=(num_manuscripts, num_manuscripts)
        )
    else: 
        references = sparse.coo_matrix(
            (np.array(data, dtype=float), ((np.array(rows, dtype=int)), np.array(cols, dtype=int))), 
            shape=(num_manuscripts, num_manuscripts)
        )
    
    return references.tocoo(), manuscript_ids, manuscript_index_map, upload_dates, \
        manuscript_meta_df, primary_memberships.tocoo(), all_memberships.tocoo(), topic_index_map



def generate_shares_matrix(
    manuscript_ids: Iterable[str],
    manuscript_index_map: Dict[str, int],
    num_contributors: int,
    avg_contributors_per_man: int,
    std_contributors_per_man: int,
    contributor_shares_dist: str = 'dirichlet',
    pareto_alpha: int = 2,
    seed: Optional[int] = None,
) -> Tuple[sparse.coo_matrix, List[str], Dict[str, int]]:
    '''build toy shares matrix'''
    
    rng = _rng(seed)
    manuscript_ids = list(manuscript_ids)
    num_manuscripts = len(manuscript_ids)

    # create contributor IDs and mapping
    contributor_ids = [str(uuid.uuid4()) for _ in range(num_contributors)]
    contributor_index_map = {id: idx for idx, id in enumerate(contributor_ids)}

    # COO format
    rows:               List[int] = []
    cols:               List[int] = []
    data:               List[float] = []
    mask_author:        List[bool] = []
    mask_reviewer:      List[bool] = []
    mask_replicator:    List[bool] = []
    rows_reviewer:      List[int] = []
    cols_reviewer:      List[int] = [] 
    rows_rep:           List[int] = []
    cols_rep:           List[int] = []
    rows_auth:          List[int] = []
    cols_auth:          List[int] = []


    # pick number of authors for a manuscript
    def pick_num_contributors() -> int:
        # pick number of contributors normally
        n = rng.normal(loc=float(avg_contributors_per_man), scale=float(std_contributors_per_man))
        n = int(round(n))

        # truncate
        if n < 1: n = 1
        if n > num_contributors: n = num_contributors
        return n

    for man in manuscript_ids:
        man_idx = manuscript_index_map[man]

        # pick number of contributors
        n_contributors = pick_num_contributors()
        n_contributors = max(1, min(n_contributors, num_contributors))

        # choose contributor indices
        idx_chosen = rng.choice(num_contributors, size=n_contributors, replace=False)

        # compute shares
        if contributor_shares_dist == 'uniform':
            shares = np.full(n_contributors, 1.0/n_contributors, dtype=float)
        elif contributor_shares_dist == 'dirichlet':
            shares = rng.dirichlet(np.ones(n_contributors)).astype(float)
        elif contributor_shares_dist == 'pareto':
            raw = rng.pareto(a=float(pareto_alpha), size=n_contributors).astype(float) + 1e-12
            shares = raw/raw.sum()
        else:
            raise ValueError('author_shares_dist must be <uniform>, <dirichlet> or <pareto>')
        
        # set the lowest 2 contributor shares as reviewers and no replicators
        k=2
        sorted_shares_idxs = np.argsort(np.where(shares == 0, np.inf, shares))
        peerrev_idxs = sorted_shares_idxs[:k]
        

        repl_shares_idxs = sorted_shares_idxs[k:2*k]
        


        # insert values in matrix
        for con_idx, share in zip(idx_chosen, shares):
            # rows.append(int(man_idx))
            # cols.append(int(con_idx))
            # data.append(float(share))

            # Build the peer reviwer mask in COO
            if con_idx in peerrev_idxs:
                rows.append(int(man_idx))
                cols.append(num_contributors + int(con_idx))
                data.append(float(share))
                
            
            # Build the replicators mask in COO
            if con_idx in repl_shares_idxs:
                rows.append(int(man_idx))
                cols.append(2*num_contributors + int(con_idx))
                data.append(float(share))
            
            # Build the authors mask in COO
            if con_idx not in peerrev_idxs and con_idx not in repl_shares_idxs:
                rows.append(int(man_idx))
                cols.append(int(con_idx))
                data.append(float(share))
    
    # build COO 
    
    shares_block = build_COO(num_manuscripts=num_manuscripts,
                             num_contributors=3*num_contributors,
                             rows=rows,
                             cols=cols,
                             data=data,
                             )
    # mask_authors_block = build_COO(num_manuscripts=num_manuscripts,
    #                                num_contributors=num_contributors,
    #                                rows=rows_auth,
    #                                cols=cols_auth,
    #                                data=mask_author,
    #                                dtype=bool)
    # mask_reviewers_block = build_COO(num_manuscripts=num_manuscripts,
    #                                num_contributors=num_contributors,
    #                                rows=rows_reviewer,
    #                                cols=cols_reviewer,
    #                                data=mask_reviewer,
    #                                dtype=bool)
    # mask_replicators_block = build_COO(num_manuscripts=num_manuscripts,
    #                                num_contributors=num_contributors,
    #                                rows=rows_rep,
    #                                cols=cols_rep,
    #                                data=mask_replicator,
    #                                dtype=bool)
    
    
    # match orientation and index shifts
    zero_block = sparse.coo_matrix(
        (np.array([], dtype=float), (np.array([], dtype=int), np.array([], dtype=int))), 
        (num_manuscripts, num_manuscripts)
    )

    # append M by M block of 0's and shift indices
    shares = sparse.hstack([zero_block, shares_block], format='coo')
    # mask_authors = sparse.hstack([zero_block, mask_authors_block], format='coo')
    # mask_reviewers = sparse.hstack([zero_block, mask_reviewers_block], format='coo')
    # mask_replicators = sparse.hstack([zero_block, mask_replicators_block], format='coo')
    

    # contributor_index_map_shifted = {id: (idx+num_manuscripts) for id, idx in contributor_index_map.items()}

    return shares.tocoo(), contributor_ids, contributor_index_map, \
        # contributor_index_map_shifted, \
        # mask_authors.tocoo(), mask_reviewers.tocoo(), mask_replicators.tocoo()



def build_capital_matrix(
    references: sparse.spmatrix,
    shares: sparse.spmatrix
) -> sparse.coo_matrix:
    '''computes capital matrix given shares and reference at current time'''
    ones_M = sparse.csr_matrix(np.ones((references.shape[1], 1))) # M by 1
    ones_MplusC = sparse.csr_matrix(np.ones((1, shares.shape[1]))) # 1 by M+C

    # convolve
    capital = sparse.coo_matrix(shares.multiply(references.dot(ones_M).dot(ones_MplusC))) # M by M+C
    return capital



# ==================== generating time series capital data ====================

def get_references_earlier(
    references: sparse.spmatrix,
    manuscript_index_map: Dict[str, int],
    upload_dates: Dict[str, date],
    time_cutoff: date
) -> sparse.spmatrix:
    '''extracts the earlier references matrix corresponding to the current manuscripts'''

    if not manuscript_index_map:
        return sparse.coo_matrix((0, 0))

    # extract indices of manuscripts uploaded before time cutoff
    indices = []
    for man_id, idx in manuscript_index_map.items():
        dt = upload_dates.get(man_id)
        if dt is None:
            continue
        if dt <= time_cutoff:
            indices.append(int(idx))

    # sort and return empty if no manuscript satisfies time cutoff
    indices.sort()
    m = len(indices)
    if m == 0:
        return sparse.coo_matrix((0, 0))

    references_csr = references.tocsr()
    # get submatrix
    if indices == list(range(0, m)):
        return references_csr[:m, :m].tocoo()
    
    # protects against non-time ordered indices, probably not needed but just in case
    return references_csr[indices, :][:, indices].tocoo()



def get_capital_earlier(
    references: sparse.spmatrix,
    shares: sparse.spmatrix,
    manuscript_index_map: Dict[str, int],
    upload_dates: Dict[str, date],
    time_cutoff: date
) -> sparse.spmatrix:
    '''builds the earlier capital matrix corresponding to given time cutoff'''

    M = int(references.shape[0])
    C = int(shares.shape[1]) - M

    references_earlier = get_references_earlier(
        references=references,
        manuscript_index_map=manuscript_index_map,
        upload_dates=upload_dates,
        time_cutoff=time_cutoff
    )
    m = int(references_earlier.shape[0])
    # return empty if no manuscript satisfies time cutoff
    if m == 0: return sparse.coo_matrix((0, 0))

    # get relevant subblock in shares matrix
    shares_csr = shares.tocsr()
    shares_block = shares_csr[:, M:(M + C)]

    # For m > 0
    shares_block_earlier = shares_block[:m, :].tocoo()  # m by C
    zero_block_earlier = sparse.coo_matrix((m, m)) # m by m
    shares_earlier = sparse.hstack([zero_block_earlier, shares_block_earlier], format='csr')  # m by m+C

    capital_earlier = build_capital_matrix(references=references_earlier, shares=shares_earlier) # m by m+C
    return capital_earlier.tocoo()



def generate_capital_time_series(
    references: sparse.spmatrix,
    shares: sparse.spmatrix,
    manuscript_index_map: Dict[str, int],
    upload_dates: Dict[str, date],
    start_date: date,
    end_date: date,
    time_step: timedelta,
) -> Dict[str, object]:
    '''generates time series data for capital matrices at different time snapshots'''

    timestamps = make_date_grid(start_date, end_date, time_step)
    T = len(timestamps)
    M = int(references.shape[0])
    C = int(shares.shape[1]) - M

    capital_snapshots: List[sparse.csr_matrix] = []
    manuscript_cutoffs: List[int] = [] # indices for manuscripts that correspond to the last upload before each time snapshot
    contributor_totals = np.zeros((T, C), dtype=float)
    manuscript_totals = np.full((T, M), -1.0, dtype=float) # setting unpulished manuscripts to -1

    for t_idx, cutoff_date in enumerate(timestamps):
        # build earlier capital matrix
        capital_earlier = get_capital_earlier(
            references=references,
            shares=shares,
            manuscript_index_map=manuscript_index_map,
            upload_dates=upload_dates,
            time_cutoff=cutoff_date
        ) # m times m+C

        m = int(capital_earlier.shape[0])
        manuscript_cutoffs.append(m)
        capital_snapshots.append(capital_earlier)

        if m == 0:
            # set contributor totals to 0 if no manuscript published yet
            contributor_totals[t_idx, :] = 0.0
            continue

        # get contributor capital
        col_sums = np.array(capital_earlier.sum(axis=0)).ravel()
        contributor_totals[t_idx, :] = col_sums[m:(m + C)]
        
        # get manuscrip capital
        row_sums = np.array(capital_earlier.sum(axis=1)).ravel()
        manuscript_totals[t_idx, :m] = row_sums

    return {
        'timestamps': timestamps,
        'manuscript_cutoff_idx': manuscript_cutoffs,
        'capital_snapshots': capital_snapshots,
        'contributor_totals': contributor_totals,
        'manuscript_totals': manuscript_totals
    }

def update_retractions_graph(
    references: sparse.spmatrix,
    retractions: Union[sparse.spmatrix, None],
    manuscript_index_map: Dict[str, int],
    newly_retracted_manuscript_ids: List[str],
    manuscripts_metadata: Union[pd.DataFrame, list],
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Union[pd.DataFrame, list]]:
    # Add complete docstring below

    
    """
    This function handles the movement of citation rows from the references graph to the 
    retractions graph when manuscripts are newly retracted. It also updates previously 
    retracted manuscripts to capture any new citations that occurred after retraction.
    
    Parameters
    ----------
    references : sparse.spmatrix
        Sparse matrix where columns represent manuscripts and rows represent cited works.
        Non-zero entries represent citation relationships.
    retractions : Union[sparse.spmatrix, None]
        Sparse matrix of the same shape as references containing previously retracted 
        manuscript citations. If None, an empty sparse matrix is initialized.
    manuscript_index_map : Dict[str, int]
        Mapping from manuscript IDs to their corresponding row indices in the matrices.
    newly_retracted_manuscript_ids : List[str]
        List of manuscript IDs that have been newly retracted in this update.
    manuscripts_metadata : Union[pd.DataFrame, list]
        Metadata associated with manuscripts. If a DataFrame, the retraction_cutoff 
        column is updated for newly retracted manuscripts.
    
    Returns
    -------
    Tuple[sparse.csr_matrix, sparse.csr_matrix, Union[pd.DataFrame, list]]
        A tuple containing:
        - retractions (sparse.coo_matrix): Updated retractions graph with newly retracted 
          and previously retracted manuscript citations.
        - references (sparse.coo_matrix): Updated references graph with retracted manuscript 
          rows zeroed out.
        - manuscripts_metadata (Union[pd.DataFrame, list]): Updated metadata with 
          retraction information for newly retracted manuscripts.
    Notes
    -----
    - Both newly retracted and previously retracted manuscript rows are removed from 
      the references graph and consolidated in the retractions graph.
    - If manuscripts_metadata is not a DataFrame, a warning is issued and metadata 
      is returned unchanged.
    - Output matrices are converted to COO format for sparse representation efficiency.

    
    """

    # Use CSR for fast row slicing and modification
    references = references.tocsr()

    if retractions is None:
        retractions = sparse.csr_matrix(references.shape, dtype=references.dtype)
    else:
        retractions = retractions.tocsr()

    # Identify indices of newly retracted manuscripts
    new_rows = np.fromiter(
        (manuscript_index_map[mid] for mid in newly_retracted_manuscript_ids),
        dtype=int
    )

    # Add new retractions in one batch
    # Combine into retractions; these rows become nonzero in retractions
    # and zeroed out in the reference graph.
    if new_rows.size:
        # Efficiently add into retractions
        retractions[new_rows, :] = references[new_rows, :]
        references[new_rows, :] = 0.0

    # Update previously retracted rows:
    prev_rows = np.unique(retractions.nonzero()[0])
    len_prev_retracted = len(prev_rows)
    if prev_rows.size:
        # find new citations (nonzeros in references that correspond to prev_retracted rows)
        # merge them
        retractions[prev_rows, :] = retractions[prev_rows, :] + references[prev_rows, :]
        references[prev_rows, :] = 0.0

    # ⬆ Update metadata if it's a DataFrame
    if isinstance(manuscripts_metadata, pd.DataFrame) and new_rows.size:
        manuscripts_metadata.loc[
            manuscripts_metadata["id"].isin(newly_retracted_manuscript_ids),
            "retraction_cutoff"
        ] = len_prev_retracted  # or another suitable counter
    else:
        warn(
            "manuscripts_metadata is not a DataFrame; skipping metadata update."
        )

    return retractions.tocoo(), references.tocoo(), manuscripts_metadata