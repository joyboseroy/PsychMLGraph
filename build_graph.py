"""
Stage 3: build the graph's node and edge tables
===================================================
Turns extracted.jsonl (from extract_kg.py) plus raw_papers.jsonl (from
fetch_corpus.py) into plain CSV node and edge tables, the same shape
used in the NCERT-KG-HI project (nodes.csv / edges.csv), so this graph
can be loaded, inspected, or extended the same way.

Run:
    python build_graph.py

Output:
    nodes_papers.csv
    nodes_disorders.csv
    nodes_datasets.csv
    nodes_models.csv
    nodes_feature_types.csv
    edges.csv
"""

import json
import csv
import re


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    papers = {p["paper_id"]: p for p in load_jsonl("raw_papers.jsonl")}
    extracted = load_jsonl("extracted.jsonl")

    disorder_nodes = {}     # slug -> display name
    model_nodes = {}
    feature_nodes = {}
    dataset_nodes = {}      # slug -> {name, sample_size, is_public} (first seen wins)
    edges = []

    for record in extracted:
        paper_id = record.get("paper_id")
        if paper_id not in papers:
            continue

        for d in record.get("disorders", []) or []:
            slug = "disorder_" + slugify(d)
            disorder_nodes[slug] = d
            edges.append({
                "source": paper_id, "target": slug, "relation": "STUDIES",
                "metric_name": "", "metric_value": "",
            })

        for m in record.get("models", []) or []:
            slug = "model_" + slugify(m)
            model_nodes[slug] = m
            edges.append({
                "source": paper_id, "target": slug, "relation": "USES_MODEL",
                "metric_name": record.get("best_metric_name") or "",
                "metric_value": record.get("best_metric_value")
                    if record.get("best_metric_value") is not None else "",
            })

        for ft in record.get("feature_types", []) or []:
            slug = "feature_" + slugify(ft)
            feature_nodes[slug] = ft
            edges.append({
                "source": paper_id, "target": slug, "relation": "USES_FEATURE_TYPE",
                "metric_name": "", "metric_value": "",
            })

        for ds in record.get("datasets", []) or []:
            name = ds.get("name")
            if not name:
                continue
            slug = "dataset_" + slugify(name)
            if slug not in dataset_nodes:
                dataset_nodes[slug] = {
                    "name": name,
                    "sample_size": ds.get("sample_size"),
                    "is_public": ds.get("is_public"),
                }
            edges.append({
                "source": paper_id, "target": slug, "relation": "USES_DATASET",
                "metric_name": "", "metric_value": "",
            })

    # ---- write node tables ----
    with open("nodes_papers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "title", "year", "venue"])
        for pid, p in papers.items():
            w.writerow([pid, p.get("title"), p.get("year"), p.get("venue")])

    with open("nodes_disorders.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "name"])
        for slug, name in disorder_nodes.items():
            w.writerow([slug, name])

    with open("nodes_models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "name"])
        for slug, name in model_nodes.items():
            w.writerow([slug, name])

    with open("nodes_feature_types.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "name"])
        for slug, name in feature_nodes.items():
            w.writerow([slug, name])

    with open("nodes_datasets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "name", "sample_size", "is_public"])
        for slug, d in dataset_nodes.items():
            w.writerow([slug, d["name"], d["sample_size"], d["is_public"]])

    # ---- write edges table ----
    with open("edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "relation", "metric_name", "metric_value"])
        for e in edges:
            w.writerow([e["source"], e["target"], e["relation"],
                        e["metric_name"], e["metric_value"]])

    print(f"Papers: {len(papers)}")
    print(f"Disorders: {len(disorder_nodes)}")
    print(f"Models: {len(model_nodes)}")
    print(f"Feature types: {len(feature_nodes)}")
    print(f"Datasets: {len(dataset_nodes)}")
    print(f"Edges: {len(edges)}")
    print("\nSaved: nodes_papers.csv, nodes_disorders.csv, nodes_models.csv,")
    print("       nodes_feature_types.csv, nodes_datasets.csv, edges.csv")


if __name__ == "__main__":
    main()
