"""
Supabase integration: fetch_supabase_json, helpers to build maps/matrices,
and build_supabase_matrices wrapper that saves artifacts expected by the
toy-dev and supabase-dev pipelines.

Public functions:
- fetch_supabase_json(output_path, tables=None, batch_size=1000, overwrite=False)
- build_supabase_matrices(raw_json, output_dir, save_raw_json=True)
"""
from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix

from dotenv import load_dotenv
from supabase import create_client, Client

from liberata_metrics.logging import get_logger
from liberata_metrics.utils import save_sparse_npz

supabase: Optional[Client] = None


# ==================== fetch data from Supabase ====================

def initialize_supabase_client(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    force_reinit: Optional[bool] = False,
    use_dotenv: Optional[bool] = True,
    log: Optional[logging.Logger] = None
) -> Client:
    """
    Initialize and return a Supabase client.

    Parameters
    ----------
    supabase_url, supabase_key : Optional[str]
        Overrides for env vars.
    force_reinit : bool
        If True, recreate the client even if one exists.
    use_dotenv : bool
        If True, call load_dotenv() to populate environment variables.
    logger : logging.Logger
        Pass in initialized logger.

    Returns
    -------
    supabase.Client
    """

    logger = log or get_logger(__name__)

    # return client if one exists already
    global supabase
    if not force_reinit and 'supabase' in globals() and supabase is not None:
        logger.info('Using pre-initialized Supabase client')
        return supabase

    if use_dotenv:
        load_dotenv()

    url = supabase_url or os.getenv('SUPABASE_URL')
    key = supabase_key or os.getenv('SUPABASE_KEY')
    if not url or not key:
        logger.error("Supabase URL/KEY not provided (env or args)")
        raise RuntimeError("Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY or pass overrides")

    try:
        supabase = create_client(url, key)
        logger.info("New Supabase client initialized")
        return supabase
    except Exception:
        logger.exception('Failed to initialized Supabase client')
        supabase = None
        raise



def fetch_supabase_json(
    output_path: Union[str, Path],
    batch_size: int = 1000,
    save_json: bool = False,
    overwrite: bool = False,
    log: Optional[logging.Logger] = None,
    supabase_client: Optional[Client] = None,
) -> Path:
    """
    Fetch the canonical tables from Supabase using the same queries as the
    existing fetch_data.py script and write a single supabase_data.json.

    Parameters
    ----------
    output_path : str|Path
        Directory or file path to write 'supabase_data.json'. If a directory is
        provided, file will be written as <output_path>/supabase_data.json.
    batch_size : int
        Page size for each table fetch.
    save_json : bool
        If True, save the resultant JSON file. 
    overwrite : bool
        If False and file exists, returns existing file.
    logger : Optional[logging.Logger]
        If provided, used for logging; otherwise a module logger is used.
    supabase_client : Optional[Client]
        If provided, used instead of initializing a client via initialize_supabase_client().

    Returns
    -------
    Path to the saved JSON file.
    """

    # setup logging and output
    logger = log or get_logger(__name__)
    outp = Path(output_path)
    if outp.is_dir():
        outp = outp / "supabase_data.json"
    outp.parent.mkdir(parents=True, exist_ok=True)

    if outp.exists() and not overwrite:
        logger.info(f"Supabase JSON already exists at {outp}")
        return outp

    # initialize supabase client
    supabase = supabase_client
    if supabase is None:
        supabase = initialize_supabase_client(log=logger)

    # fetch Supabase data
    try:
    
        # Fetch manuscripts
        logger.info("Fetching manuscripts...")
        manuscripts = []
        start = 0

        while True:
            end = start + batch_size - 1

            response = (
                supabase.table("manuscripts")
                .select("id, retracted, topics")
                .eq("identifier_type", "openalex")
                .order("id")
                .range(start, end)
                .execute()
            )
            
            data = response.data or []
            if not data:
                logger.error(f"Error fetching manuscripts")
                raise RuntimeError(f"Supabase fetch error for manuscripts")

            manuscripts.extend(data)
            if len(data) < batch_size:
                break

            start += batch_size
        
        logger.info(f"Total manuscripts fetched: {len(manuscripts)}")


        # Fetch users
        logger.info("Fetching users...")
        users = []
        start = 0

        while True:
            end = start + batch_size - 1
            response = (
                supabase.table("users")
                .select("id")
                .not_.is_("oa_int_id", "null")
                .order("id")
                .range(start, end)
                .execute()
            )
            
            data = response.data or []
            if not data:
                logger.error(f"Error fetching users")
                raise RuntimeError(f"Supabase fetch error for users")

            users.extend(data)
            if len(data) < batch_size:
                break

            start += batch_size

        logger.info(f"Total users fetched: {len(users)}")


        # Fetch manuscript contributors
        logger.info("Fetching manuscript contributors...")
        shares = []
        start = 0

        while True:
            end = start + batch_size - 1
            response = (
                supabase.table("manuscript_contributors")
                .select("manuscript_id, user_id, contributor_type, weight, manuscripts!inner(identifier_type), users!inner(oa_int_id)")
                .eq("manuscripts.identifier_type", "openalex")
                .not_.is_("users.oa_int_id", "null")
                .order("manuscript_id, user_id, contributor_type")
                .range(start, end)
                .execute()
            )
            
            data = response.data or []
            if not data:
                logger.error(f"Error fetching manuscript_contributors")
                raise RuntimeError(f"Supabase fetch error for manuscript_contributors")

            shares.extend(data)
            if len(data) < batch_size:
                break

            start += batch_size
        
        logger.info(f"Total contributors fetched: {len(shares)}")


        # Fetch citations
        logger.info("Fetching citations...")
        citations = []
        start = 0

        while True:
            end = start + batch_size - 1
            response = (
                supabase.table("citations")
                .select("citing_manuscript_id, cited_manuscript_id, weight, citing_manuscript:manuscripts!citing_manuscript_id!inner(identifier_type), cited_manuscript:manuscripts!cited_manuscript_id!inner(identifier_type)")
                .eq("citing_manuscript.identifier_type", "openalex")
                .eq("cited_manuscript.identifier_type", "openalex")
                .order("citing_manuscript_id, cited_manuscript_id")
                .range(start, end)
                .execute()
            )
            
            data = response.data or []
            if not data:
                logger.error(f"Error fetching citations")
                raise RuntimeError(f"Supabase fetch error for citations")

            citations.extend(data)
            if len(data) < batch_size:
                break

            start += batch_size

        logger.info(f"Total citations fetched: {len(citations)}")

        # Save to JSON file
        output_data = {
            "manuscripts": manuscripts,
            "users": users,
            "contributors": shares,
            "citations": citations,
        }

        if save_json:
            with open(outp, "w", encoding="utf-8") as fh:
                json.dump(output_data, fh, indent=2)
            logger.info(f"Saved supabase JSON to {outp}")

        return output_data, outp

    except Exception:
        logger.exception("Failed to fetch supabase data")
        raise



# ==================== matrix creation helpers ====================

def create_manuscript_map(
    manuscripts: List[Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mapping from manuscript UUID to index and reverse mapping"""

    manuscript_map: Dict[str, int] = {}
    manuscript_reverse_map: Dict[int, str] = {}
    for idx, manuscript in enumerate(manuscripts):
        manuscript_map[manuscript['id']] = idx
        manuscript_reverse_map[idx] = manuscript['id']
    return manuscript_map, manuscript_reverse_map



def create_user_map(
    users: List[Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mapping from user UUID to index and reverse mapping"""

    user_map = {}
    user_reverse_map = {}
    for idx, user in enumerate(users):
        user_map[user['id']] = idx
        user_reverse_map[idx] = user['id']
    return user_map, user_reverse_map



def create_references_matrix(
    citations: List[Dict[str, Any]], 
    manuscript_map: Dict[str, int],
    log: Optional[logging.Logger] = None,
) -> coo_matrix:
    """Create references matrix from citations data"""

    logger = log or get_logger(__name__)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    skipped = 0

    for record in citations:
        citing_id = record["citing_manuscript_id"]
        cited_id = record["cited_manuscript_id"]
        weight = record["weight"]

        # skip manuscripts if their index does not exist
        if citing_id not in manuscript_map or cited_id not in manuscript_map:
            skipped += 1
            continue

        citing_idx = manuscript_map[citing_id]
        cited_idx = manuscript_map[cited_id]

        # insert manuscripts into matrix
        rows.append(cited_idx)
        cols.append(citing_idx)
        data.append(weight)
    
    if skipped:
        logger.warning(f"Skipped {skipped} citations referencing unknown manuscripts")

    num_manuscripts = len(manuscript_map)
    sparse_matrix = coo_matrix(
        (np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))), 
        shape=(num_manuscripts, num_manuscripts)
    )

    return sparse_matrix



def create_shares_matrix(
    contributors: List[Dict[str, Any]],
    manuscript_map: Dict[str, int],
    user_map: Dict[str, int],
    log: Optional[logging.Logger] = None,
) -> coo_matrix:
    """Create shares matrix from manuscript_contributors data with separate roles"""
    
    logger = log or get_logger(__name__)

    rows = []
    cols = []
    data = []
    
    skipped = 0

    # role offsets: authors at M, reviewers at M+|C|, replicators at M+2|C|
    num_users = len(user_map)
    num_manuscripts = len(manuscript_map)
    role_offsets = {
        'author': 0,
        'reviewer': num_users,
        'replicator': 2 * num_users
    }

    for record in contributors:
        manuscript_id = record["manuscript_id"]
        user_id = record["user_id"]
        weight = record["weight"]
        contributor_type = record["contributor_type"]
        
        # skip if manuscript or user doesn't exist in maps
        if manuscript_id not in manuscript_map or user_id not in user_map or contributor_type not in role_offsets:
            skipped += 1
            continue
        
        manuscript_idx = manuscript_map[manuscript_id]
        user_base_idx = user_map[user_id]
        role_offset = role_offsets[contributor_type]
        user_idx = num_manuscripts + user_base_idx + role_offset

        # insert manuscripts into matrix
        rows.append(manuscript_idx)
        cols.append(user_idx)
        data.append(weight)

    if skipped:
        logger.warning(f"Skipped {skipped} contributor records with invalid IDs or roles")

    shares_matrix_size = num_manuscripts + (3 * num_users)
    sparse_matrix = coo_matrix(
        (np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))), 
        shape=(shares_matrix_size, shares_matrix_size)
    )

    return sparse_matrix



def create_capital_matrix(
    references_matrix: coo_matrix, 
    shares_matrix: coo_matrix
) -> coo_matrix:
    """
    Construct capital matrix from shares and references matrices
    """
    M = references_matrix.shape[0]

    # Extract only manuscript rows from shares matrix
    shares_csr = shares_matrix.tocsr()
    shares_manuscript_rows = shares_csr[:M, :]

    # Create ones vectors for convolution
    ones_M = csr_matrix(np.ones((references_matrix.shape[1], 1)))
    ones_MplusC = csr_matrix(np.ones((1, shares_manuscript_rows.shape[1])))

    # convolve
    capital = coo_matrix(shares_manuscript_rows.multiply(references_matrix.dot(ones_M).dot(ones_MplusC)))
    return capital



def build_topic_memberships(
    manuscripts: List[Dict[str, Any]],
    log: Optional[logging.Logger] = None,
) -> Tuple[coo_matrix, coo_matrix, Dict[str, int]]:
    """Build (T x M) all_memberships and primary_memberships matrices and topic_index_map"""

    logger = log or get_logger(__name__)

    topic_to_idx: Dict[str, int] = {}
    all_rows: List[int] = []
    all_cols: List[int] = []
    all_data: List[float] = []
    primary_rows: List[int] = []
    primary_cols: List[int] = []
    primary_data: List[float] = []
    skipped = 0

    def get_topic_id(t: Any) -> Optional[str]:
        '''Small helper function to extract topic IDs'''
        # take last 5 characters of 'id' field
        if isinstance(t, dict):
            whole_id = str(t.get("id"))
            topic_id = whole_id[-5:] if len(whole_id) >= 5 else whole_id
            return topic_id
        return None

    # iterate through manuscripts to extract topics
    for midx, m in enumerate(manuscripts):
        topics = m.get('topics') or []
        if len(topics) == 0:
            skipped += 1
            continue

        # iterate through all topics and add to matrix
        for t in topics:
            topic_id = get_topic_id(t)
            if not topic_id: 
                continue

            if topic_id not in topic_to_idx:
                topic_to_idx[topic_id] = len(topic_to_idx)
            
            tidx = topic_to_idx[topic_id]
            all_rows.append(tidx)
            all_cols.append(midx)
            all_data.append(True)

        # add first topic as primary topic to primary topics matrix
        primary_topic_id = get_topic_id(topics[0])
        if primary_topic_id not in topic_to_idx:
            topic_to_idx[primary_topic_id] = len(topic_to_idx)
        
        ptidx = topic_to_idx[primary_topic_id]
        primary_rows.append(ptidx)
        primary_cols.append(midx)
        primary_data.append(True)
    
    if skipped:
        logger.warning(f"Skipped {skipped} manuscripts with no associated topic tags")

    T = len(topic_to_idx)
    M = len(manuscripts)

    # build indicator matrices for topics per manuscript
    if T > 0:
        all_memberships = coo_matrix(
            (np.array(all_data, dtype=float), (np.array(all_rows, dtype=int), np.array(all_cols, dtype=int))),
            shape=(T, M),
        )
        primary_memberships = coo_matrix(
            (np.array(primary_data, dtype=float), (np.array(primary_rows, dtype=int), np.array(primary_cols, dtype=int))),
            shape=(T, M),
        )
    else:
        all_memberships = coo_matrix((0, M))
        primary_memberships = coo_matrix((0, M))
    
    return all_memberships, primary_memberships, topic_to_idx



# ==================== build matrices from Supabase data ====================

def build_supabase_matrices(
    raw_json: Union[str, Path, Dict[str, Any]],
    output_dir: Union[str, Path],
    save_raw_json: bool = False,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Wrapper that builds canonical artifacts from supabase_data (path or dict)
    and saves the files expected by data_loading and the supabase dev scripts.

    Parameters
    ----------
    raw_json : str|Path|dict
        Path to supabase_data.json or in-memory dict.
    output_dir : str|Path
        Directory to write artifact files.
    save_raw_json : bool
        If True and raw_json was dict, save a copy to output_dir/supabase_data.json.

    Returns
    -------
    summary dict with counts and saved file paths.
    """
    
    # setup logging and output dir
    logger = log or get_logger(__name__)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load raw JSON if a path is provided
    if isinstance(raw_json, (str, Path)):
        raw_path = Path(raw_json)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw Supabase JSON not found: {raw_path}")
        
        with open(raw_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = raw_json

    # build matrices
    try:
        manuscripts = data.get("manuscripts", [])
        users = data.get("users", [])
        contributors = data.get("contributors", [])
        citations = data.get("citations", [])

        # build maps
        manuscript_map, manuscript_reverse = create_manuscript_map(manuscripts)
        user_map, user_reverse = create_user_map(users)
        num_manuscripts = len(manuscript_map)
        num_users = len(user_map)

        logger.info(f"Building matrices for {num_manuscripts} manuscripts, {num_users} users")

        # references and retractions split
        references = create_references_matrix(citations, manuscript_map, logger).tocoo()
        retracted_ids = [str(m.get("id")) for m in manuscripts if m.get("retracted")]
        
        if retracted_ids:
            logger.info(f"Detected {len(retracted_ids)} retracted manuscripts")

            retracted_set = {manuscript_map[s] for s in retracted_ids if s in manuscript_map}
            keep_mask = np.array([r not in retracted_set for r in references.row])
            retractions_mask = ~keep_mask
            references_clean = coo_matrix(
                (references.data[keep_mask], (references.row[keep_mask], references.col[keep_mask])), 
                shape=references.shape
            )

            retractions = coo_matrix(
                (references.data[retractions_mask], (references.row[retractions_mask], references.col[retractions_mask])), 
                shape=references.shape
            )
        
        else:
            retractions = coo_matrix(references.shape)

        # shares and capital matrices
        shares = create_shares_matrix(contributors, manuscript_map, user_map, logger)
        capital = create_capital_matrix(references_clean, shares)
        retractions_capital = create_capital_matrix(retractions, shares)

        # topic memberships
        all_memberships, primary_memberships, topic_to_idx = build_topic_memberships(manuscripts, logger)

        # save results
        save_sparse_npz(outdir / "references_coo.npz", references_clean.tocoo(), log=logger)
        save_sparse_npz(outdir / "references_all_coo.npz", references.tocoo(), log=logger)
        save_sparse_npz(outdir / "shares_coo.npz", shares.tocoo(), log=logger)
        save_sparse_npz(outdir / "capital_coo.npz", capital.tocoo(), log=logger)

        save_sparse_npz(outdir / "retractions_coo.npz", retractions.tocoo(), log=logger)
        save_sparse_npz(outdir / "retractions_capital_coo.npz", retractions_capital.tocoo(), log=logger)
        save_sparse_npz(outdir / "primary_memberships_tags.npz", primary_memberships.tocoo(), log=logger)
        save_sparse_npz(outdir / "all_memberships_tags.npz", all_memberships.tocoo(), log=logger)

        # save maps and pickles
        with open(outdir / "manuscript_index_map.json", "w", encoding="utf-8") as fh:
            json.dump(manuscript_map, fh, indent=2)
        with open(outdir / "contributor_index_map.json", "w", encoding="utf-8") as fh:
            json.dump(user_map, fh, indent=2)
        with open(outdir / "topic_index_map.json", "w", encoding="utf-8") as fh:
            json.dump(topic_to_idx, fh, indent=2)

        # save raw JSON copy if JSON file was given
        if save_raw_json:
            raw_copy = data if isinstance(raw_json, dict) else json.load(open(raw_json, "r", encoding="utf-8"))
            with open(outdir / "supabase_data.json", "w", encoding="utf-8") as fh:
                json.dump(raw_copy, fh)

        # create summary JSON
        summary = {
            "output_dir": str(outdir),
            "num_manuscripts": num_manuscripts,
            "num_users": num_users,
            "files": {
                "references": str(outdir / "references_coo.npz"),
                "references_all": str(outdir / "references_all_coo.npz"),
                "shares": str(outdir / "shares_coo.npz"),
                "capital": str(outdir / "capital_coo.npz"),
                "retractions": str(outdir / "retractions_coo.npz"),
                "retractions_capital": str(outdir / "retractions_capital_coo.npz"),
                "manuscript_map": str(outdir / "manuscript_index_map.json"),
                "contributor_map": str(outdir / "contributor_index_map.json"),
                "topic_map": str(outdir / "topic_index_map.json"),
                "supabase_json": str(outdir / "supabase_data.json") if save_raw_json else None,
            },
        }

        logger.info(f"Saved Supabase data to {outdir}")
        return summary

    except Exception:
        logger.exception("Failed to process and save Supabase data")
        raise