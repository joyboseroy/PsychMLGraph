"""
Stage 4: a few quick summaries from the graph
=================================================
Loads the CSV tables from build_graph.py into NetworkX and prints a
handful of summaries. The main one this whole project is really about:
has the fraction of papers relying on imaging, versus cheaper data like
EEG or plain clinical/tabular records, shifted over the years covered
by the corpus? That is an empirical, literature-wide version of the
question M. Grassi's thesis argues for from a single research group's
work.

Run:
    pip install networkx pandas
    python analyze_graph.py
"""

import csv
from collections import defaultdict

import networkx as nx
import pandas as pd


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_networkx_graph():
    G = nx.DiGraph()
    papers = load_csv("nodes_papers.csv")
    for p in papers:
        G.add_node(p["paper_id"], node_type="Paper", **p)

    for fname, ntype in [
        ("nodes_disorders.csv", "Disorder"),
        ("nodes_models.csv", "Model"),
        ("nodes_feature_types.csv", "FeatureType"),
        ("nodes_datasets.csv", "Dataset"),
    ]:
        for row in load_csv(fname):
            G.add_node(row["node_id"], node_type=ntype, **row)

    edges = load_csv("edges.csv")
    for e in edges:
        G.add_edge(e["source"], e["target"], relation=e["relation"],
                   metric_name=e["metric_name"], metric_value=e["metric_value"])
    return G, papers, edges


def imaging_vs_clinical_trend(papers, edges):
    # A paper counts as "imaging" if it has a USES_FEATURE_TYPE edge to
    # Neuroimaging, "cheap data" if it uses EEG, Clinical/tabular,
    # Wearable/sensor, or Clinical notes/NLP instead (and no imaging).
    # Papers that use both, or neither, are left out of this specific
    # count rather than force-classified either way.
    paper_years = {p["paper_id"]: p["year"] for p in papers}
    paper_features = defaultdict(set)
    for e in edges:
        if e["relation"] == "USES_FEATURE_TYPE":
            paper_features[e["source"]].add(e["target"])

    cheap_types = {"feature_eeg", "feature_clinical_tabular",
                   "feature_wearable_sensor", "feature_clinical_notes_nlp"}

    rows = []
    for pid, feats in paper_features.items():
        year = paper_years.get(pid)
        if not year:
            continue
        has_imaging = "feature_neuroimaging" in feats
        has_cheap = bool(feats & cheap_types)
        if has_imaging and not has_cheap:
            category = "imaging"
        elif has_cheap and not has_imaging:
            category = "cheap_data"
        else:
            continue  # both or neither -- skip, do not force a label
        rows.append({"year": int(year), "category": category})

    if not rows:
        print("Not enough classified papers yet to show a trend.")
        return

    df = pd.DataFrame(rows)
    trend = df.groupby(["year", "category"]).size().unstack(fill_value=0)
    trend["total"] = trend.sum(axis=1)
    if "imaging" in trend.columns:
        trend["pct_imaging"] = (trend.get("imaging", 0) / trend["total"] * 100).round(1)
    print("\nImaging vs cheap-data papers by year (only papers using exactly one):")
    print(trend)


def most_common(G, node_type, top_n=10):
    counts = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == node_type:
            counts.append((data.get("name", node), G.in_degree(node)))
    counts.sort(key=lambda x: -x[1])
    return counts[:top_n]


def average_metric_by_model(edges):
    values = defaultdict(list)
    for e in edges:
        if e["relation"] == "USES_MODEL" and e["metric_value"]:
            try:
                values[e["target"]].append(float(e["metric_value"]))
            except ValueError:
                continue
    return {k: (sum(v) / len(v), len(v)) for k, v in values.items()}


def main():
    G, papers, edges = build_networkx_graph()

    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    print("\nMost studied disorders (by number of papers):")
    for name, count in most_common(G, "Disorder"):
        print(f"  {name}: {count}")

    print("\nMost used models (by number of papers):")
    for name, count in most_common(G, "Model"):
        print(f"  {name}: {count}")

    print("\nMost used feature types (by number of papers):")
    for name, count in most_common(G, "FeatureType"):
        print(f"  {name}: {count}")

    print("\nAverage reported metric value per model (mixing different")
    print("metric types together, so read this as a rough signal only):")
    avg_metrics = average_metric_by_model(edges)
    node_names = {row["node_id"]: row["name"] for row in load_csv("nodes_models.csv")}
    for slug, (avg, n) in sorted(avg_metrics.items(), key=lambda x: -x[1][0]):
        print(f"  {node_names.get(slug, slug)}: {avg:.3f} (n={n})")

    imaging_vs_clinical_trend(papers, edges)


if __name__ == "__main__":
    main()
