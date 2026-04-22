from pathlib import Path
import os
import json
from typing import Any, Dict, Tuple
from datetime import datetime
from scipy.sparse import load_npz

from liberata_metrics.logging import configure_logging, get_logger
from liberata_metrics.integrations.supabase import (
    fetch_supabase_json,
    build_supabase_matrices,
)

from liberata_metrics.visualizations import matrix_heatmap

# configure logging
configure_logging(level="INFO", log_file="logs/data_processing_test.log")
logger = get_logger(__name__)

def main():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path("supabase_test_scripts/output") / f'supabase_{timestamp}'
    outdir.mkdir(parents=True, exist_ok=True)
    figs_dir = outdir / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Supabase fetch...")
    raw_data = fetch_supabase_json(output_path=outdir, batch_size=1000, save_json=False, overwrite=True, log=logger)

    if isinstance(raw_data, tuple) and len(raw_data) == 2:
        supabase_data, json_path = raw_data
    else:
        json_path = Path(raw_data)
        with open(json_path, "r", encoding="utf-8") as fh:
            supabase_data = json.load(fh)

    logger.info("Fetched Supabase data")
    summary = build_supabase_matrices(raw_json=supabase_data, output_dir=outdir, save_raw_json=False, log=logger)

    logger.info(f"Data processing for Supabase data finished: {summary}")

    try:
        # prefer paths from summary if provided
        refs_path = Path(summary.get("files", {}).get("references") or outdir / "references_coo.npz")
        shares_path = Path(summary.get("files", {}).get("shares") or outdir / "shares_coo.npz")
        capital_path = Path(summary.get("files", {}).get("capital") or outdir / "capital_coo.npz")

        logger.info(f"Loading matrices for visualization: {refs_path}, {shares_path}, {capital_path}")

        references = load_npz(str(refs_path))
        shares = load_npz(str(shares_path))
        capital = load_npz(str(capital_path))

        logger.info("Generating heatmaps ...")
        matrix_heatmap(references, figs_dir / "references.png", title="References Matrix", cmap='Blues')
        matrix_heatmap(shares, figs_dir / "shares.png", title="Shares Matrix", cmap='Greens')
        matrix_heatmap(capital, figs_dir / "capital.png", title="Capital Matrix", cmap='Reds')
        logger.info(f"Saved matrix heatmaps to {figs_dir}")
    except Exception:
        logger.exception("Failed to generate matrix visualizations")

if __name__ == "__main__":
    main()