# PsychMLGraph: a small knowledge graph of psychiatry machine learning papers

A small personal project, not a paper, not a formal study. I read a 2022
PhD thesis on using machine learning in psychiatry (M. Grassi,
Maastricht University), which argues that models should use cheap,
easy-to-collect clinical information instead of expensive brain scans,
because that is what actually gets used in a real clinic. I wanted to
check something the thesis itself cannot answer on its own: across the
much wider published literature, has the field actually been moving
that way over time, or is imaging still the dominant approach? A single
thesis cannot show a trend like that. A small graph built from many
papers can at least gesture at one.

## What this actually is

A four-stage pipeline that searches for psychiatry machine learning
papers, pulls their titles and abstracts through a free public API,
asks a large language model to pull out a few structured facts from
each abstract (what disorder, what kind of data, what model, what
headline result), and turns all of that into a small graph. Then a
short analysis script asks a few questions of that graph, including the
imaging-versus-cheap-data trend question above.

This reuses the same basic shape as a few of my other repos
(MarxGraph, NCERT-KG-HI): download or fetch a corpus, extract
structured fields from it with an LLM, build a small graph from the
extracted fields, then look at it.

## Results from the first real run

This ran successfully end to end on 176 papers pulled from a dozen
search queries, spanning 2015 through 2026. Full data is on
[HuggingFace](https://huggingface.co/datasets/joyboseroy/PsychMLGraph);
here is the short version.

**Corpus snapshot:** 176 papers, 9 disorders, 9 model families, 8 data
modalities, 87 named datasets, 768 edges.

**Most studied disorders** (by number of papers):

| Disorder | Papers |
|---|---|
| Depression | 59 |
| Schizophrenia | 32 |
| Bipolar Disorder | 29 |
| PTSD | 26 |
| Anxiety | 26 |
| Other | 22 |
| Autism | 17 |
| OCD | 8 |
| Alzheimer's/Dementia | 2 |

This corpus is heavily skewed toward Depression and away from
Alzheimer's/Dementia, the disorder the original thesis actually studied
in most depth. That skew comes from which search queries happened to
return results before hitting Semantic Scholar's free-tier rate limit,
not a deliberate design choice, and should not be read as reflecting
the real balance of the wider literature. Anyone extending this corpus
should specifically target more Alzheimer's-related queries to correct
for this.

**Most used models** (by number of papers):

| Model | Papers |
|---|---|
| Other | 65 |
| Random Forest | 46 |
| SVM | 44 |
| Gradient Boosting | 32 |
| Logistic Regression | 25 |
| CNN | 18 |
| RNN/LSTM | 9 |
| Transformer | 5 |
| Large Language Model | 4 |

**Most used data types** (by number of papers):

| Data type | Papers |
|---|---|
| Clinical/tabular | 90 |
| EEG | 32 |
| Clinical notes/NLP | 30 |
| Neuroimaging | 26 |
| Other | 11 |
| Wearable/sensor | 10 |
| Genetic | 7 |
| Speech/audio | 4 |

**Average reported metric value by model** (accuracy, AUC, sensitivity,
and F1 all mixed together here, so read this as a rough signal, not a
precise comparison across different tasks):

| Model | Average metric | Papers |
|---|---|---|
| RNN/LSTM | 0.979 | 5 |
| CNN | 0.940 | 12 |
| SVM | 0.870 | 34 |
| Large Language Model | 0.845 | 2 |
| Random Forest | 0.844 | 35 |
| Other | 0.825 | 47 |
| Transformer | 0.817 | 4 |
| Gradient Boosting | 0.808 | 27 |
| Logistic Regression | 0.808 | 20 |

**The main question this project set out to ask:** has the literature
shifted away from imaging toward cheap, clinically-available data over
time, the way the source thesis argues it should?

| Year | Cheap-data papers | Imaging papers | Total | % imaging |
|---|---|---|---|---|
| 2015 | 1 | 0 | 1 | 0.0% |
| 2016 | 2 | 0 | 2 | 0.0% |
| 2017 | 2 | 1 | 3 | 33.3% |
| 2018 | 1 | 0 | 1 | 0.0% |
| 2019 | 2 | 1 | 3 | 33.3% |
| 2020 | 5 | 2 | 7 | 28.6% |
| 2021 | 7 | 1 | 8 | 12.5% |
| 2022 | 10 | 0 | 10 | 0.0% |
| 2023 | 15 | 0 | 15 | 0.0% |
| 2024 | 23 | 3 | 26 | 11.5% |
| 2025 | 50 | 6 | 56 | 10.7% |
| 2026 | 11 | 1 | 12 | 8.3% |

The years before 2020 have only 1 to 3 papers each, so those
percentages are noisy and should not be read as meaningful on their
own. From 2022 onward the sample sizes are large enough to say
something real: imaging's share of single-modality papers stays low,
mostly under 12%, while the raw count of cheap-data papers grows
substantially each year. That is a real, if modest, answer in favor of
the thesis's own argument, drawn from the wider literature rather than
one research group's work.

## Pipeline stages

1. `fetch_corpus.py`: searches the Semantic Scholar API (free, no
   application needed, no account required for light use) across a
   short list of search terms covering major psychiatric conditions and
   data types, and saves titles, abstracts, years, and venues to
   `raw_papers.jsonl`. The free tier without your own API key gets rate
   limited hard and unpredictably since it is shared across everyone
   using it at once, so this script retries with a growing delay when
   that happens, and appends to the output file rather than overwriting
   it, so a partial run followed by a retry never loses earlier
   progress. A free API key from semanticscholar.org gives you your own
   private rate limit and mostly avoids this, worth getting if you plan
   to pull a bigger corpus.
   **If the rate limiting is still too unreliable to be worth fighting,
   use `add_manual_papers.py` instead, see below.**
2. `extract_kg.py`: sends each abstract to an LLM and asks for a fixed
   set of structured fields back as JSON, which disorder, which kind of
   data, which model family, which dataset if named, and the headline
   result if the abstract states one clearly. Set up here to use Groq's
   free API, the same one already used in my MarxGraph project, with
   the model `openai/gpt-oss-120b`. Needs a `GROQ_API_KEY` environment
   variable set beforehand.
3. `build_graph.py`: turns the extracted fields into plain CSV node and
   edge tables (`nodes_papers.csv`, `nodes_disorders.csv`,
   `nodes_models.csv`, `nodes_feature_types.csv`, `nodes_datasets.csv`,
   `edges.csv`), the same shape used in my NCERT-KG-HI project.
4. `analyze_graph.py`: loads those tables into NetworkX and prints out
   the most studied disorders, the most used models and data types, a
   rough average of reported results per model, and the imaging versus
   cheap-data trend by year.

See `ontology.md` for the full node and edge type definitions and the
reasoning behind them.

## Adding papers by hand instead of the automated search

If the Semantic Scholar rate limiting is too unreliable, or you would
rather just search for relevant papers yourself on Google Scholar or
Semantic Scholar's own website and pick the ones that actually look
useful, `add_manual_papers.py` handles this. Copy
`manual_papers_template.csv`, fill in a row per paper (title, the full
abstract text, year, venue, and optionally a DOI or link), then run:

```
python add_manual_papers.py your_filename.csv
```

It writes to the exact same `raw_papers.jsonl` file `fetch_corpus.py`
uses, so from `extract_kg.py` onward the rest of the pipeline does not
know or care whether a paper came from the automated search, a manual
entry, or both mixed together. Duplicate entries (the same paper added
twice, or added both automatically and by hand) are detected and only
kept once.

## What I have actually tested, and what I have not

I want to be upfront about this rather than let it look more finished
than it is. I built and fully tested `build_graph.py`,
`analyze_graph.py`, `add_manual_papers.py`, `fix_metric_scale.py`, and
`convert_to_parquet.py` myself, using made-up sample data standing in
for real papers, and all of them work correctly, including correctly
merging the same dataset mentioned in two different papers into one
shared node, correctly computing the year-by-year trend, correctly
skipping duplicate or too-short manually-entered rows without silently
losing them, and correctly rescaling percentage-style metric values.

I was not able to run `fetch_corpus.py` or `extract_kg.py` end to end
myself against the real, live services, the environment I write code
in cannot reach the Semantic Scholar API or call the Groq API directly.
Both have since actually been run for real, successfully, end to end,
across 176 papers, and produced the results section above. The first
version of `fetch_corpus.py` did run into heavy rate limiting from
Semantic Scholar's free unauthenticated tier in practice (a lot of 429
errors), which makes sense since that tier is shared across everyone
using it at once without their own key. The current version retries
with a growing delay and appends progress rather than overwriting it,
and across three separate runs correctly built up from 38 to 127 to 176
papers with no duplicates and no lost progress. `extract_kg.py` also
turned up one genuine bug on that first real run, some metric values
came back on a 0-100 scale instead of the requested 0-1 decimal scale,
which `fix_metric_scale.py` now corrects, and the extraction prompt has
been tightened to ask for decimals explicitly going forward. If the
rate limiting is still too unreliable on a given connection,
`add_manual_papers.py` sidesteps the problem completely by letting you
add papers you find yourself.

## How to run this yourself

You do not need to apply for access to anything or recruit any
participants. You do need a free Groq API key for the extraction step
(console.groq.com).

```
pip install requests openai networkx pandas
python fetch_corpus.py
export GROQ_API_KEY=your_key_here
python extract_kg.py
python build_graph.py
python analyze_graph.py
```

The first version pulls a small corpus, around 100 to 200 papers
depending on how many abstracts pass the length filter, from a dozen
search queries. This is meant to be a fast first look, not a full
literature review. Add more queries to `fetch_corpus.py` to grow it
later.

## Honest limitations

An abstract alone often does not report every methodological detail a
full paper would, so sample sizes, dataset names, and exact metric
values will sometimes come back blank even when the extraction step
asks for them, because the abstract genuinely does not say. The LLM
extraction step can also get borderline cases wrong, for example
whether a lightly-tuned logistic regression on hand-picked features
counts as "clinical" data or something else. Metric values from
different papers are not directly comparable either, since different
papers report different metrics (accuracy, AUC, sensitivity, F1) on
different tasks and different data, so any averaging across papers in
the analysis script is a rough signal, not a rigorous comparison.

This is a quick, honest first look at a broad question, built from
abstracts and an LLM's reading of them, not a systematic review, and
should not be treated as one.

## Publishing this as a HuggingFace dataset

If you want to release the graph as a proper HuggingFace dataset rather
than just keeping the CSV files here, two more scripts handle that:

- `fix_metric_scale.py`: run this once, right after `build_graph.py` and
  before `analyze_graph.py`. The first real run of this pipeline turned
  up a genuine bug, some abstracts state a metric as a percentage (like
  "95% accuracy") and the extraction step sometimes wrote that down as
  95 instead of 0.95, which silently breaks any averaging across
  papers. This script rescales anything on a 0 to 100 scale back down
  to a 0 to 1 decimal, and leaves anything stranger untouched with a
  printed note rather than guessing.
- `convert_to_parquet.py`: turns the CSV tables into parquet files in a
  new `hf_release/` folder, the format HuggingFace's dataset viewer
  expects.

`HF_README.md` is a ready-to-use dataset card for the HuggingFace repo
itself (separate from this GitHub README, HuggingFace dataset cards use
their own format with a metadata header at the top). Before actually
calling a release finished, it is worth doing at least a small manual
check, pull 20 or so random papers, read the abstract yourself, and see
whether what the LLM extracted actually matches. No such check has been
done on this corpus yet, and `HF_README.md` says so plainly rather than
claiming an accuracy figure that was never actually measured.

## Data sources and credit

- Semantic Scholar Graph API: https://api.semanticscholar.org/graph/v1/paper/search
  Free, public, no application required.
- Groq API, used for the extraction step: https://console.groq.com
- The thesis that started all this: M. Grassi, "Supervised Machine
  Learning in Psychiatry: Towards Application in Clinical Practice",
  PhD thesis, Maastricht University, 2022.

## What this project is not

This is not a peer reviewed paper, not a systematic literature review,
and not a claim of any new scientific discovery. It is a small, honest
attempt to look at a broad question using a quick automated pipeline,
built and tested as far as I was able to test it myself before handing
it over. If you spot a mistake or have an idea for how to test any of
this better, feel free to open an issue.
