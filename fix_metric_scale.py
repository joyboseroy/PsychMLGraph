"""
Fix: some extracted metric values are on a 0-100 scale, others on 0-1
========================================================================
The LLM extraction step sometimes wrote a metric as "95" (copying a
percentage straight from the abstract, like "95% accuracy") and other
times correctly wrote "0.95" as a plain decimal fraction. Averaging
those together is meaningless, a value of 1.015 for an accuracy metric
is impossible, which is exactly the kind of thing that showed up in the
first real run of this pipeline.

This script rewrites edges.csv in place, converting any metric_value
greater than 1 (and no more than 100, since anything higher almost
certainly is not a valid percentage either and gets left alone rather
than guessed at) down to a 0-1 decimal fraction by dividing by 100.

Run this once, right after build_graph.py, before analyze_graph.py.

Run:
    python fix_metric_scale.py
"""

import csv

INPUT_FILE = "edges.csv"


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixed = 0
    left_alone = 0
    for row in rows:
        val = row.get("metric_value", "")
        if not val:
            continue
        try:
            num = float(val)
        except ValueError:
            continue
        if 1.0 < num <= 100.0:
            row["metric_value"] = str(round(num / 100.0, 4))
            fixed += 1
        elif num > 100.0:
            # something odder than a straightforward percentage slip,
            # leave it as-is rather than guess, but flag it for a look
            print(f"  -> odd value left as-is: {num} (source={row['source']}, "
                  f"target={row['target']})")
            left_alone += 1

    with open(INPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "target", "relation", "metric_name", "metric_value"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Rescaled {fixed} percentage-style values down to 0-1 decimals.")
    if left_alone:
        print(f"Left {left_alone} unusual values untouched, worth a manual look.")
    print(f"Saved back to {INPUT_FILE}")


if __name__ == "__main__":
    main()
