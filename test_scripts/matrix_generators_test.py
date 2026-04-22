from __future__ import annotations
import json

from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np
from scipy import sparse
import pandas as pd

from liberata_metrics.generators import generate_references_matrix, generate_shares_matrix, build_capital_matrix, generate_capital_time_series, update_retractions_graph
from liberata_metrics.visualizations import matrix_heatmap, plot_contributor_time_series, plot_manuscript_time_series
from liberata_metrics.utils import read_yaml_config, save_sparse_npz, serialize_upload_dates, _rng

from liberata_metrics.logging import configure_logging, get_logger


def main(config):
    '''testing with config'''
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rng = _rng(config['seed'])

    # setup logging
    configure_logging(level='INFO', log_file='logs/matrix_generating.log')
    logger = get_logger(__name__)
    logger.info('Starting logging...')

    # read from config file
    matrix_config = config.get('matrices', {})
    num_manuscripts = matrix_config.get('num_manuscripts', 100)
    num_contributors = matrix_config.get('num_contributors', 100)

    # create output directory
    output_dir = Path(config.get('output_dir', 'output')) / f'm{num_manuscripts}_c{num_contributors}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)
    time_dir = output_dir / 'time_series'
    time_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Beginning matrix generation to output directory: {output_dir}')

    # get references matrix    
    logger.info('Generating references matrix...')
    references, manuscript_ids, manuscript_index_map, upload_dates, \
    manuscript_meta, primary_memberships, all_memberships, topic_index_map = generate_references_matrix(
        num_manuscripts=num_manuscripts,
        citation_density=matrix_config['references']['citation_density'],
        start_date=date(matrix_config['start_date']['year'], matrix_config['start_date']['month'], matrix_config['start_date']['day']),
        end_date=date(matrix_config['end_date']['year'], matrix_config['end_date']['month'], matrix_config['end_date']['day']),
        seed = config['seed']
    )
    matrix_heatmap(references, figs_dir / 'references.png', title="References Matrix")
    logger.info(f'References matrix shape: {references.shape}  nnz: {references.nnz}')
    M = len(manuscript_ids)
    logger.info(f'Manuscript IDs length: {M}')

    # get shares matrix
    logger.info('Generating shares matrix...')
    shares, contributor_ids, contributor_index_map, = \
    generate_shares_matrix(
        manuscript_ids=manuscript_ids,
        manuscript_index_map=manuscript_index_map,
        num_contributors=num_contributors,
        avg_contributors_per_man=matrix_config['shares']['avg_contributors_per_man'],
        std_contributors_per_man=matrix_config['shares']['std_contributors_per_man'],
        contributor_shares_dist=matrix_config['shares']['contributor_shares_dist'],
        pareto_alpha=matrix_config['shares']['pareto_alpha']
    )
    matrix_heatmap(shares, figs_dir / 'shares.png', title="Shares Matrix")
    # matrix_heatmap(mask_authors, figs_dir / 'authors.png', title="Authors Mask Matrix")
    # matrix_heatmap(mask_reviewers, figs_dir / 'reviewers.png', title="Reviewers Mask Matrix")
    # matrix_heatmap(mask_replicators, figs_dir / 'replicators.png', title="Replicators Mask Matrix")
    logger.info(f'Shares matrix shape: {shares.shape}  nnz: {shares.nnz}')
    # print('Authors Mask matrix nnz:', mask_authors.nnz)
    # print('Reviewers Mask matrix nnz:', mask_reviewers.nnz)
    # print('Replicators Mask matrix nnz:', mask_replicators.nnz)
    C = len(contributor_ids)
    logger.info(f'Contributor IDs length: {C}')


    logger.info('Generating retractions matrix...')
    retractions, references, manuscript_meta = update_retractions_graph(
                                                                        references=references,
                                                                        retractions=None,
                                                                        manuscript_index_map=manuscript_index_map,
                                                                        newly_retracted_manuscript_ids=rng.choice(manuscript_ids, 
                                                                                                                  size=int(0.1 * len(manuscript_ids)), 
                                                                                                                  replace=False).tolist(),
                                                                        manuscripts_metadata=manuscript_meta,
                                                                    )


    # compute capital matrix at time t + 1
    logger.info('Building capital matrix...')
    capital = build_capital_matrix(references=references, shares=shares)
    retractions_capital = build_capital_matrix(references=retractions, shares=shares)
    matrix_heatmap(capital, figs_dir / 'capital.png', title='Capital Matrix')
    matrix_heatmap(retractions_capital, figs_dir / 'retractions_capital.png', title='Retractions Capital Matrix')   
    
    logger.info(f'Capital matrix shape: {capital.shape}  nnz: {capital.nnz}')
    logger.info(f'Retractions Capital matrix shape: {retractions_capital.shape}  nnz: {retractions_capital.nnz}')

    # save matrices
    logger.info('Saving matrices...')
    save_sparse_npz(output_dir / f'references_coo.npz', references, logger)
    save_sparse_npz(output_dir / f'shares_coo.npz', shares, logger)
    # save_sparse_npz(output_dir / f'mask_authors_coo.npz', mask_authors)
    # save_sparse_npz(output_dir / f'mask_reviewers_coo.npz', mask_reviewers)
    # save_sparse_npz(output_dir / f'mask_replicators_coo.npz', mask_replicators)
    save_sparse_npz(output_dir / f'capital_coo.npz', capital, logger)
    save_sparse_npz(output_dir / f'retractions_coo.npz', retractions, logger)
    save_sparse_npz(output_dir / f'primary_memberships_tags.npz', primary_memberships, logger)
    save_sparse_npz(output_dir / f'all_memberships_tags.npz', all_memberships, logger)
    save_sparse_npz(output_dir / f'retractions_coo.npz', retractions, logger)
    save_sparse_npz(output_dir / f'retractions_capital_coo.npz', retractions_capital, logger)

    # save manuscript ID mappings
    with open(output_dir / 'manuscript_index_map.json', 'w', encoding='utf-8') as f:
        json.dump(manuscript_index_map, f, indent=2)

    # save contributor ID mappings
    with open(output_dir / 'contributor_index_map.json', 'w', encoding='utf-8') as f:
        json.dump(contributor_index_map, f, indent=2)
    
    # save topic ID mappings
    with open(output_dir / 'topic_index_map.json', 'w', encoding='utf-8') as f:
        json.dump(topic_index_map, f, indent=2)

    # save upload dates
    with open(output_dir / 'upload_dates.json', 'w', encoding='utf-8') as f:
        json.dump(serialize_upload_dates(upload_dates), f, indent=2)
    
    # manuscript_meta.to_parquet(output_dir / "manuscripts_meta.parquet")
    manuscript_meta.to_csv(output_dir / "manuscripts_meta.csv")

    # extract time series capital data for contributors + manuscripts
    logger.info('Extracting time-series data...')
    capital_time_series = generate_capital_time_series(
        references=references,
        shares=shares,
        manuscript_index_map=manuscript_index_map,
        upload_dates=upload_dates,
        start_date=date(matrix_config['start_date']['year'], matrix_config['start_date']['month'], matrix_config['start_date']['day']),
        end_date=date(matrix_config['end_date']['year'], matrix_config['end_date']['month'], matrix_config['end_date']['day']),
        time_step=timedelta(days=config['time_series']['time_step'])
    )
    timestamps = capital_time_series['timestamps']
    contributor_time_series = capital_time_series['contributor_totals']
    manuscript_time_series = capital_time_series['manuscript_totals']
    logger.info(f'Time series length: {len(timestamps)}')

    # save time series
    contributor_df = pd.DataFrame(
        contributor_time_series,
        index=[d.isoformat() for d in timestamps],
        columns=[f"{id}" for id in contributor_ids]*3
    )
    plot_contributor_time_series(
        contributor_df=contributor_df,
        contributor_ids=None,
        c=min(10, len(contributor_ids)),
        output_path=figs_dir / 'contributor_time_series.png',
        rng_seed=config['seed']
    )
    contributor_csv_path = time_dir / "contributor_time_series.csv"
    contributor_df.to_csv(contributor_csv_path)

    manuscript_df = pd.DataFrame(
        manuscript_time_series,
        index=[d.isoformat() for d in timestamps],
        columns=[f"{id}" for id in manuscript_ids]
    )
    plot_manuscript_time_series(
        manuscript_df=manuscript_df,
        manuscript_ids=None,
        c=min(10, len(manuscript_ids)),
        output_path=figs_dir / 'manuscript_time_series.png',
        rng_seed=config['seed']
    )
    manuscript_csv_path = time_dir / "manuscript_time_series.csv"
    manuscript_df.to_csv(manuscript_csv_path)
    



if __name__ == '__main__':
    
    config_path = 'test_scripts/config/matrix_config.yaml'
    config = read_yaml_config(config_path)
    main(config)