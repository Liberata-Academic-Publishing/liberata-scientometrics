from pathlib import Path
import os

from liberata_metrics.logging import configure_logging, get_logger
from liberata_metrics.integrations.supabase import fetch_supabase_json

# configure logging once
configure_logging(level="INFO", log_file="logs/supabase_fetch_test.log")
logger = get_logger(__name__)

def main():
    outdir = Path("supabase_test_scripts/data")
    outdir.mkdir(parents=True, exist_ok=True)
    save_json = False

    # fetch json 
    supabase_json, json_path = fetch_supabase_json(output_path=outdir, batch_size=1000, save_json=save_json, overwrite=True, log=logger)
    if save_json: logger.info(f"Supabase JSON saved to: {json_path}")

if __name__ == "__main__":
    main()