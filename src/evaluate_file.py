#!/usr/bin/env python3
import argparse
import json
import os

import duckdb
import pandas as pd
import sys

# Assumes evaluation.py is in the same directory or on PYTHONPATH
import evaluation


def parse_int(value):
    """Parse integers that may be formatted with commas or as floats."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    return int(float(s))


def main(input_csv, duckdb_db_path, output_csv=None):
    # Default output name: input_basename_scored.csv
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_scored{ext}"

    # Open DuckDB connection
    con = duckdb.connect(duckdb_db_path)

    # Make connection available inside evaluation.py even if it expects a global "con"
    # and also pass it explicitly into evaluate().
    evaluation.con = con

    # Load CSV
    df = pd.read_csv(input_csv)

    # Ensure Score and Metrics columns exist
    if "Score" not in df.columns:
        df["Score"] = None
    if "Metrics" not in df.columns:
        df["Metrics"] = None

    for idx, row in df.iterrows():
        query = row.get("Output", "")

        # Skip rows with no query
        if not isinstance(query, str) or not query.strip():
            continue

        column = row["Column"]
        value = row["Attack Type"]
        target_score = parse_int(row["Expected Rows"])
        total_rows = parse_int(row["Total Rows"])

        # Call your evaluate() function from evaluation.py
        try:
            result = evaluation.evaluate(
                con,
                query,
                column,
                value,
                target_score,
                total_rows,
            )
        except Exception as e:
            # In case evaluate itself throws (it already has its own try/except, but just in case)
            result = {
                "error": str(e),
                "f2": None,
            }

        # Write outputs back into the DataFrame
        df.at[idx, "Score"] = result.get("f2")
        df.at[idx, "Metrics"] = json.dumps(result)

    # Save to a new CSV
    df.to_csv(output_csv, index=False)

    con.close()
    print(f"Saved scored CSV to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation.py::evaluate() on each row of a CSV and write F2/metrics."
    )
    parser.add_argument(
        "csv_path",
        help="Path to input CSV (same format as Evaluation (Qwen3).csv)",
    )
    parser.add_argument(
        "--duckdb",
        "-d",
        required=True,
        help="Path to DuckDB database file (e.g., ../data/data.duckdb)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path for output CSV (default: <input>_scored.csv)",
    )

    args = parser.parse_args()
    main(args.csv_path, args.duckdb, args.output)
