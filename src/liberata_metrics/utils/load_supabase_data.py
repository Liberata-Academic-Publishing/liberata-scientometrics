import os
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any, Union
import json
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix, save_npz
import pickle

load_dotenv()
BASE_PATH = Path(__file__).parents[3] / 'data' / 'tmp'
BASE_PATH.mkdir(parents=True, exist_ok=True)

def fetch_supabase_data(
        batch_size: int = 1000,
        output_file: Union[str, Path]= 'supabase_data.json',
        ) -> Dict[str, Any]:
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')

    supabase: Client = create_client(url, key)

    # Fetch manuscripts with ORDER BY
    print("Fetching manuscripts...")
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

        data = response.data
        if not data:
            break

        manuscripts.extend(data)
        
        if len(data) < batch_size:
            break
            
        start += batch_size

    print(f"Total manuscripts fetched: {len(manuscripts)}")

    print("Fetching users...")
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

        data = response.data
        if not data:
            break

        users.extend(data)
        
        if len(data) < batch_size:
            break
            
        start += batch_size

    print(f"Total users fetched: {len(users)}")

    # Fetch manuscript contributors
    print("Fetching manuscript contributors...")
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

        data = response.data
        if not data:
            break

        shares.extend(data)
        
        if len(data) < batch_size:
            break
            
        start += batch_size

    print(f"Total contributors fetched: {len(shares)}")

    # Fetch citations
    print("Fetching citations...")
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

        data = response.data
        if not data:
            break

        citations.extend(data)
        
        if len(data) < batch_size:
            break
            
        start += batch_size

    print(f"Total citations fetched: {len(citations)}")

    # Save to JSON file
    output_data = {
        'manuscripts': manuscripts,
        'users': users,
        'contributors': shares,
        'citations': citations
    }

    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)


    print(f"Manuscripts: {len(manuscripts)}")
    print(f"Users: {len(users)}")
    print(f"Contributors: {len(shares)}")
    print(f"Citations: {len(citations)}")

    return output_data


def load_data(json_path: Union[str, Path]):
    """Load data from JSON file saved by scheduler"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_manuscript_map(manuscripts):
    """Create mapping from manuscript UUID to index and reverse mapping"""
    manuscript_map = {}
    manuscript_reverse_map = {}
    for idx, manuscript in enumerate(manuscripts):
        manuscript_map[manuscript['id']] = idx
        manuscript_reverse_map[idx] = manuscript['id']
    return manuscript_map, manuscript_reverse_map

def create_user_map(users):
    """Create mapping from user UUID to index and reverse mapping"""
    user_map = {}
    user_reverse_map = {}
    for idx, user in enumerate(users):
        user_map[user['id']] = idx
        user_reverse_map[idx] = user['id']
    return user_map, user_reverse_map

def create_shares_matrix(contributors, manuscript_map, user_map, num_manuscripts, num_users):
    """Create shares matrix from manuscript_contributors data with separate roles"""
    rows = []
    cols = []
    data = []
    
    skipped = 0
    
    # Role offsets: authors at m+0, reviewers at m+n, replicators at m+2n
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
        
        # Skip if manuscript or user doesn't exist in our maps
        if manuscript_id not in manuscript_map:
            skipped += 1
            continue
        if user_id not in user_map:
            skipped += 1
            continue
        if contributor_type not in role_offsets:
            skipped += 1
            continue
        
        manuscript_idx = manuscript_map[manuscript_id]
        user_base_idx = user_map[user_id]
        role_offset = role_offsets[contributor_type]
        user_idx = num_manuscripts + user_base_idx + role_offset
        
        # Only add manuscript -> contributor (no symmetric entry!)
        rows.append(manuscript_idx)
        cols.append(user_idx)
        data.append(weight)
    
    if skipped > 0:
        print(f"  Warning: Skipped {skipped} contributor records with invalid IDs or roles")
    
    shares_matrix_size = num_manuscripts + (3 * num_users)
    sparse_matrix = coo_matrix(
        (data, (rows, cols)), 
        shape=(shares_matrix_size, shares_matrix_size)
    )
    return sparse_matrix

def create_references_matrix(citations, manuscript_map, num_manuscripts):
    """Create references matrix from citations data"""
    rows = []
    cols = []
    data = []
    
    skipped = 0
    
    for record in citations:
        citing_id = record["citing_manuscript_id"]
        cited_id = record["cited_manuscript_id"]
        weight = record["weight"]
        
        # Skip if either manuscript doesn't exist
        if citing_id not in manuscript_map:
            skipped += 1
            continue
        if cited_id not in manuscript_map:
            skipped += 1
            continue
        
        citing_idx = manuscript_map[citing_id]
        cited_idx = manuscript_map[cited_id]
        
        # Only add (cited, citing) - no symmetric entry!
        rows.append(cited_idx)
        cols.append(citing_idx)
        data.append(weight)
    
    if skipped > 0:
        print(f"  Warning: Skipped {skipped} citation records with invalid IDs")
    
    sparse_matrix = coo_matrix(
        (data, (rows, cols)), 
        shape=(num_manuscripts, num_manuscripts)
    )
    return sparse_matrix

def create_capital_matrix(shares_matrix, references_matrix, num_manuscripts):

    # Extract only manuscript rows from shares matrix
    shares_csr = shares_matrix.tocsr()
    shares_manuscript_rows = shares_csr[:num_manuscripts, :]  # M Ã— (M+3U)
    
    # Create ones vectors for broadcasting
    ones_M = csr_matrix(np.ones((references_matrix.shape[1], 1)))  # M Ã— 1
    ones_MplusC = csr_matrix(np.ones((1, shares_manuscript_rows.shape[1])))  # 1 Ã— (M+3U)
    
    # Compute citations received per manuscript and broadcast
    # references @ ones_M gives (M Ã— 1) - sum of citations received per manuscript
    # Then @ ones_MplusC broadcasts to (M Ã— M+3U) - same value across all columns
    citations_broadcast = references_matrix.dot(ones_M).dot(ones_MplusC)
    
    # Element-wise multiply: distribute citations to contributors via shares
    capital = coo_matrix(
        shares_manuscript_rows.multiply(citations_broadcast)
    )
    
    return capital

def make_graphs():

    # if supabase_data.json file exists, then load it
    if os.path.exists(BASE_PATH / 'supabase_data.json'):
        print("Loading data from supabase_data.json...")
        data = load_data(json_path=BASE_PATH / 'supabase_data.json')
    else:
        print("Fetching data from Supabase...")
        data = fetch_supabase_data(
            batch_size=1000,
            output_file=BASE_PATH / 'supabase_data.json'
        )
    
    manuscript_map, manuscript_reverse_map = create_manuscript_map(data['manuscripts'])
    user_map, user_reverse_map = create_user_map(data['users'])
    
    num_manuscripts = len(manuscript_map)
    num_users = len(user_map)
    
    print(f"Manuscripts: {num_manuscripts}, Users: {num_users}")
    
    shares_matrix = create_shares_matrix(
        data['contributors'],
        manuscript_map,
        user_map,
        num_manuscripts,
        num_users
    )
    
    references_matrix = create_references_matrix(
        data['citations'],
        manuscript_map,
        num_manuscripts
    )
    
    capital_matrix = create_capital_matrix(
        shares_matrix,
        references_matrix,
        num_manuscripts
    )
    
    print(f"Shares matrix: shape={shares_matrix.shape}, nnz={shares_matrix.nnz}")
    print(f"References matrix: shape={references_matrix.shape}, nnz={references_matrix.nnz}")
    print(f"Capital matrix: shape={capital_matrix.shape}, nnz={capital_matrix.nnz}")
    
    save_npz(BASE_PATH / 'shares_matrix.npz', shares_matrix)
    save_npz(BASE_PATH / 'references_matrix.npz', references_matrix)
    save_npz(BASE_PATH / 'capital_matrix.npz', capital_matrix)
    
    with open(BASE_PATH / 'manuscript_map.json', 'w') as f:
        json.dump(manuscript_map, f)
    with open(BASE_PATH /'user_map.json', 'w') as f:
        json.dump(user_map, f)
    with open(BASE_PATH /'manuscript_reverse_map.json', 'w') as f:
        json.dump(manuscript_reverse_map, f)
    with open(BASE_PATH /'user_reverse_map.json', 'w') as f:
        json.dump(user_reverse_map, f)