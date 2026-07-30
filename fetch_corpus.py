"""
Stage 1: fetch a small corpus of psychiatry ML papers
=========================================================
Pulls titles, abstracts, years, and venues from the Semantic Scholar
Graph API. This is free and does not need any application or account.

The free, unauthenticated tier is shared across everyone using it at
once, without a key of your own, so it throttles hard and unevenly, a
429 error means "rate limited," not that anything is wrong with your
setup. Getting a free API key from semanticscholar.org gives you your
own private rate limit instead of a shared pool, and is worth doing if
you keep hitting a lot of 429s. If you have one, set it as an
environment variable before running this script:

    export SEMANTIC_SCHOLAR_API_KEY=your_key_here

Run:
    python fetch_corpus.py

Output:
    raw_papers.jsonl -- one JSON object per line, one paper per line
    (appends to any existing file rather than overwriting it, so a
    partial run followed by a retry does not lose earlier progress)
"""

import json
import os
import time
import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,venue,externalIds"

# Each query pairs a disorder with a data modality, or is a broader
# catch-all term. This list is deliberately short for a fast first
# version -- add more queries here later to grow the corpus.
QUERIES = [
    "machine learning depression prediction",
    "machine learning schizophrenia classification",
    "machine learning Alzheimer's disease prediction",
    "machine learning OCD treatment outcome",
    "machine learning PTSD prediction",
    "machine learning bipolar disorder classification",
    "machine learning anxiety disorder prediction",
    "machine learning autism classification",
    "EEG deep learning psychiatric diagnosis",
    "neuroimaging deep learning psychiatric diagnosis",
    "electronic health records machine learning psychiatry",
    "natural language processing clinical notes psychiatry",
]

RESULTS_PER_QUERY = 20
SLEEP_BETWEEN_CALLS_SEC = 3.0  # be polite to the free public API
MAX_RETRIES = 5


def fetch_query(query, limit=RESULTS_PER_QUERY, api_key=None):
    params = {"query": query, "fields": FIELDS, "limit": limit}
    headers = {"x-api-key": api_key} if api_key else {}

    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            # Respect the server's own Retry-After header if it sends one,
            # otherwise back off with a growing delay each attempt.
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else (5 * attempt)
            print(f"    rate limited, waiting {wait:.0f}s before retry "
                  f"{attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json().get("data", [])

    print(f"    gave up after {MAX_RETRIES} retries, skipping this query")
    return []


def load_existing_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["paper_id"])
    return ids


def main():
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        print("Using your own Semantic Scholar API key (higher rate limit).")
    else:
        print("No SEMANTIC_SCHOLAR_API_KEY set, using the shared free tier, "
              "expect some 429s and retries.")

    out_path = "raw_papers.jsonl"
    seen_ids = load_existing_ids(out_path)
    if seen_ids:
        print(f"Found {len(seen_ids)} papers already saved from a previous "
              f"run, will not re-fetch or duplicate those.")

    new_papers = []
    for query in QUERIES:
        print(f"Searching: {query}")
        results = fetch_query(query, api_key=api_key)

        added = 0
        for paper in results:
            paper_id = paper.get("paperId")
            abstract = paper.get("abstract")
            if not paper_id or paper_id in seen_ids:
                continue
            if not abstract or len(abstract) < 200:
                # too short to reliably extract structured fields from
                continue
            seen_ids.add(paper_id)
            new_papers.append({
                "paper_id": paper_id,
                "title": paper.get("title"),
                "abstract": abstract,
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "external_ids": paper.get("externalIds"),
                "source_query": query,
            })
            added += 1
        print(f"  -> {added} new papers with usable abstracts")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print(f"\nNew papers collected this run: {len(new_papers)}")
    print(f"Total papers in {out_path} after this run: {len(seen_ids)}")
    with open(out_path, "a", encoding="utf-8") as f:
        for p in new_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
