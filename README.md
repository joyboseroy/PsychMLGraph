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

This reuses the same basic shape as a few of my other repos: download or fetch a corpus, extract
structured fields from it with an LLM, build a small graph from the
extracted fields, then look at it.

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
`analyze_graph.py`, and `add_manual_papers.py` myself, using made-up
sample data standing in for real papers, and all three work correctly,
including correctly merging the same dataset mentioned in two different
papers into one shared node, correctly computing the year-by-year
trend, and correctly skipping duplicate or too-short manually-entered
rows without silently losing them.

I have not been able to run `fetch_corpus.py` or `extract_kg.py` end to
end myself against the real, live services, the environment I write
code in cannot reach the Semantic Scholar API or call the Groq API
directly. The first version of `fetch_corpus.py` ran into heavy rate
limiting from Semantic Scholar's free unauthenticated tier in practice
(a lot of 429 errors), which makes sense since that tier is shared
across everyone using it at once without their own key. The current
version retries with a growing delay and appends progress rather than
overwriting it, which should help, but if it is still too unreliable,
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
