"""
Stage 2: extract structured fields from each abstract with an LLM
======================================================================
Uses Groq's OpenAI-compatible endpoint, the same setup already used in
MarxGraph (model: openai/gpt-oss-120b). Needs a GROQ_API_KEY environment
variable, set beforehand, for example:

    export GROQ_API_KEY=your_key_here     (Mac or Linux)
    setx GROQ_API_KEY "your_key_here"      (Windows, then open a new terminal)

Run:
    pip install openai
    python extract_kg.py

Input:
    raw_papers.jsonl (from fetch_corpus.py)

Output:
    extracted.jsonl -- one JSON object per line, matched to paper_id
"""

import json
import os
import time

from openai import OpenAI

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

INPUT_FILE = "raw_papers.jsonl"
OUTPUT_FILE = "extracted.jsonl"

# Controlled vocabularies. The LLM is asked to pick from these where it
# can, and to use "Other" plus a short free-text note when nothing fits,
# so nothing gets forced into the wrong bucket.
DISORDERS = [
    "Depression", "Schizophrenia", "Alzheimer's/Dementia", "OCD", "PTSD",
    "Bipolar Disorder", "Anxiety", "Autism", "Other",
]
MODELS = [
    "SVM", "Random Forest", "Gradient Boosting", "Logistic Regression",
    "CNN", "RNN/LSTM", "Transformer", "Large Language Model", "Other",
]
FEATURE_TYPES = [
    "Neuroimaging", "EEG", "Genetic", "Clinical/tabular",
    "Wearable/sensor", "Clinical notes/NLP", "Speech/audio", "Other",
]

SYSTEM_PROMPT = f"""You are extracting structured facts from psychiatry
machine learning paper abstracts for a research knowledge graph.

Respond with ONLY a JSON object, no other text, matching exactly this
shape:

{{
  "disorders": [list of strings, choose only from: {DISORDERS}],
  "models": [list of strings, choose only from: {MODELS}],
  "feature_types": [list of strings, choose only from: {FEATURE_TYPES}],
  "datasets": [
    {{"name": string, "sample_size": integer or null, "is_public": true, false, or null}}
  ],
  "best_metric_name": string or null (for example "AUC", "accuracy", "sensitivity"),
  "best_metric_value": number or null (as a plain decimal, for example 0.87)
}}

Rules:
- Only include a disorder, model, or feature type if the abstract
  actually supports it. Do not guess.
- If the abstract does not clearly state a sample size, dataset name,
  or metric value, use null for that field rather than making one up.
- best_metric_name and best_metric_value should reflect the single
  headline result the abstract itself emphasizes most, if there is one.
- best_metric_value must always be a decimal fraction between 0 and 1,
  never a percentage number. If the abstract says "95% accuracy",
  best_metric_value should be 0.95, not 95.
- Use empty lists, not null, for list fields with nothing to report.
"""


def extract_one(client, paper):
    user_content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # Some models wrap JSON in code fences despite instructions;
            # strip those off if present before parsing.
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            if attempt == 2:
                print(f"  -> extraction failed for {paper['paper_id']}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. Get a free key "
            "from console.groq.com and set it before running this script."
        )
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    papers = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    print(f"Loaded {len(papers)} papers to extract from.")

    results = []
    for i, paper in enumerate(papers):
        print(f"[{i+1}/{len(papers)}] {paper['title'][:70]}")
        extracted = extract_one(client, paper)
        if extracted is not None:
            extracted["paper_id"] = paper["paper_id"]
            results.append(extracted)
        time.sleep(0.5)  # light rate limiting

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nExtracted fields for {len(results)}/{len(papers)} papers.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
