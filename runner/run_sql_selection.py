import sys
import argparse
import os
sys.path.append(".")

from app.config import get_config
from app.logger import configure_logger
from app.pipeline.sql_selection import SQLSelectionRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run L4 SQL selection, optionally for one ablation arm")
    parser.add_argument("--config", help="TOML configuration path")
    parser.add_argument("--arm", choices=list("ABCDEF"), help="Override [sql_selection].ablation_arm")
    parser.add_argument("--input_snapshot", help="Fixed L3 snapshot for this arm")
    parser.add_argument("--output_snapshot", help="Dedicated L4 snapshot for this arm")
    args = parser.parse_args()
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    app_config = get_config()
    if args.arm:
        app_config.sql_selection_config.ablation_arm = args.arm
    if args.output_snapshot:
        app_config.sql_selection_config.save_path = args.output_snapshot
    configure_logger(app_config.logger_config.print_level)
    sql_selection_runner = SQLSelectionRunner.from_config(app_config, input_save_path=args.input_snapshot)
    sql_selection_runner.run()
