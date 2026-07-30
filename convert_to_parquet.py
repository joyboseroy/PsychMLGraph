"""
Convert the CSV tables into parquet, ready for a HuggingFace release
========================================================================
HuggingFace's dataset viewer and most dataset-loading code expect
parquet files, not CSV. This just converts each existing table over,
keeping the same column names and content, so this is a pure format
change, nothing about the data itself changes here (run
fix_metric_scale.py first if you have not already, to fix the
percentage-versus-decimal metric bug before this step).

Run:
    pip install pandas pyarrow
    python convert_to_parquet.py

Output (in a new hf_release/ folder):
    papers.parquet
    disorders.parquet
    models.parquet
    feature_types.parquet
    datasets.parquet
    edges.parquet
"""

import os
import pandas as pd

OUT_DIR = "hf_release"

FILES = {
    "nodes_papers.csv": "papers.parquet",
    "nodes_disorders.csv": "disorders.parquet",
    "nodes_models.csv": "models.parquet",
    "nodes_feature_types.csv": "feature_types.parquet",
    "nodes_datasets.csv": "datasets.parquet",
    "edges.csv": "edges.parquet",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for csv_name, parquet_name in FILES.items():
        if not os.path.exists(csv_name):
            print(f"Skipping {csv_name}, not found (did build_graph.py run?)")
            continue
        df = pd.read_csv(csv_name)
        out_path = os.path.join(OUT_DIR, parquet_name)
        df.to_parquet(out_path, index=False)
        print(f"{csv_name} -> {out_path} ({len(df)} rows)")

    print(f"\nAll files written to {OUT_DIR}/")
    print("This folder, plus a dataset card README.md, is what you upload")
    print("to the HuggingFace dataset repo.")


if __name__ == "__main__":
    main()
