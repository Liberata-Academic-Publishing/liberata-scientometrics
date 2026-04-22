from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Union
import numpy as np
import yaml
import os
from scipy import sparse
import logging

from liberata_metrics.logging import get_logger

def save_sparse_npz(path: Union[str, Path], matrix: sparse.spmatrix, log: Optional[logging.Logger] = None) -> None:
    '''save scipy.sparse matrix to .npz'''

    logger = log or get_logger(__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(str(path), matrix)
    logger.info(f'Matrix saved: {str(path)}')


def read_yaml_config(config_path: str):
    '''reads YAML config file and returns dictionary'''
    try: 
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f'YAML config file not found at {config_path}')
    except yaml.YAMLError as e:
        raise ValueError(f'Failed to parse YAML file {config_path}:\n{e}')