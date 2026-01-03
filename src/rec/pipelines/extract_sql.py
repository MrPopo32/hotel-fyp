"""Extract hotel tables from MySQL to parquet files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine

from rec.util.io import INTERIM_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hotel/member/review tables to parquet.")
    parser.add_argument("--host", default=os.getenv("HOTELREC_DB_HOST", "localhost"))
    parser.add_argument("--user", default=os.getenv("HOTELREC_DB_USER", "root"))
    parser.add_argument("--password", default=os.getenv("HOTELREC_DB_PASSWORD"))
    parser.add_argument("--database", default=os.getenv("HOTELREC_DB_NAME", "hotelfyp"))
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.password:
        raise SystemExit(
            "Missing database password. Set HOTELREC_DB_PASSWORD or pass --password."
        )

    engine = create_engine(
        f"mysql+pymysql://{args.user}:{quote_plus(args.password)}@{args.host}/{args.database}"
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name in ("hotel", "member", "review"):
        frame = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        output_path = output_dir / f"{table_name}.parquet"
        frame.to_parquet(output_path, index=False)
        print(f"Saved {table_name} to {output_path} | shape={frame.shape}")


if __name__ == "__main__":
    main()
