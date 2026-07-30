# PsychMLGraph: ontology and scope (v1 plan)

A small knowledge graph built from psychiatry machine learning paper
abstracts, extracted with an LLM. The starting point was a 2022 PhD
thesis (M. Grassi, Maastricht University) on machine learning in
psychiatry. The question this graph is meant to help answer: across the
wider published literature, has the field actually moved toward the
thesis's own preferred approach, cheap clinical data over expensive
imaging, or is imaging still dominant? A graph over many papers can show
this as a trend over time, which a single thesis obviously cannot.

## Node types

- **Paper**: one node per paper. Attributes: paper_id (from Semantic
  Scholar), title, year, venue.
- **Disorder**: a normalized psychiatric or neurological condition, for
  example Depression, Schizophrenia, Alzheimer's/Dementia, OCD, PTSD,
  Bipolar Disorder, Anxiety, Autism. A small controlled list plus an
  "Other" bucket for anything that does not fit, so nothing gets thrown
  away, it just does not get its own dedicated node yet.
- **Dataset**: a named dataset mentioned in the abstract (for example
  ADNI, OASIS, a named clinical cohort), with sample size and whether it
  is public, when the abstract says so clearly enough to tell.
- **Model**: a normalized model family, for example SVM, Random Forest,
  Gradient Boosting, Logistic Regression, CNN, RNN or LSTM, Transformer,
  Large Language Model, Other.
- **FeatureType**: what kind of input data the model actually uses:
  Neuroimaging, EEG, Genetic, Clinical or tabular, Wearable or sensor,
  Clinical notes or NLP, Speech or audio, Other.

## Edge types

- Paper STUDIES Disorder
- Paper USES_DATASET Dataset
- Paper USES_MODEL Model (carries the best reported metric name and
  value as an edge attribute, when the abstract states one clearly)
- Paper USES_FEATURE_TYPE FeatureType

This keeps the graph a simple bipartite-ish structure around the Paper
node, similar in spirit to earlier projects like MarxGraph and
NCERT-KG-HI, just with a different node and edge vocabulary suited to
this literature.

## v1 scope

Just like the NCERT-KG-HI seed release, this is meant to be a small,
honestly-labeled first version, not a full literature review. The plan
is to pull maybe 60 to 100 paper abstracts across a handful of search
queries covering major disorders crossed with major data modalities,
extract structured fields from each abstract with an LLM, and build the
graph from that. Anything the LLM cannot confidently extract from an
abstract alone (a lot of detail is only in the full paper text) is left
blank rather than guessed at.

## Known limitations going in

Abstracts alone often do not report full methodological detail, so
sample sizes, dataset names, and exact metric values will sometimes be
missing even when the code asks for them. The LLM extraction step can
also make mistakes, especially on borderline cases (is a lightly-tuned
logistic regression a "classical" or "clinical" model, for instance).
This is a first pass meant to show a rough shape of the literature, not
a fully verified dataset.

## Pipeline stages

1. `fetch_corpus.py`: queries the Semantic Scholar API (free, no
   application needed) for paper titles and abstracts across a list of
   search terms, deduplicates, and saves a raw corpus file.
2. `extract_kg.py`: sends each abstract to an LLM (set up here for Groq,
   matching the setup already used in MarxGraph) with a fixed prompt
   asking for structured JSON fields, and saves the extracted fields.
3. `build_graph.py`: turns the extracted fields into node and edge
   tables (CSV files), the same nodes.csv / edges.csv shape used in
   NCERT-KG-HI.
4. `analyze_graph.py`: a few quick NetworkX-based summaries, including
   the main question this whole project is about, whether the fraction
   of papers relying on imaging versus cheaper clinical/EEG/tabular data
   has shifted over the years covered by the corpus.
