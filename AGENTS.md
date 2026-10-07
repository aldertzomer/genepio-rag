# AGENTS.md — GenEpiO RAG metadata normalizer

## Purpose

This repository is a prototype for converting heterogeneous, messy biological sample metadata into **ontology-grounded, machine-queryable annotations**.

The immediate input is ENA/SRA/BioSample-style metadata, but the design should remain generic enough for other microbiology and infectious-disease metadata.

The project is broader than MetaLyzer. MetaLyzer answers a specific classification question such as source attribution. This project should instead create a reusable semantic layer that supports later questions such as:

- Was the isolate associated with infection, colonisation/carriage, or neither?
- What anatomical site/specimen did it come from: urine, blood, respiratory tract, intestine, skin, etc.?
- What host species was involved?
- Was it clinical, environmental, food-associated, farm-associated, wastewater-associated, etc.?
- What disease, phenotype, treatment, collection process, geography, or other contextual concepts are represented?

Do not optimize the architecture around one downstream question. The output should preserve enough semantic information to support future queries that were not anticipated when the metadata was processed.

## Central design principle

The LLM is used for **semantic interpretation and decomposition**, while ontology identifiers come only from a local ontology retrieval layer.

The LLM must never be trusted to invent ontology IDs.

Pipeline:

```text
raw metadata record
      |
      v
Ministral-3-8B-Instruct-2512
concept extraction / normalization
      |
      v
local GenEpiO retrieval
(labels + synonyms + definitions + hierarchy)
      |
      v
candidate ontology terms
      |
      v
Ministral-3-8B-Instruct-2512
contextual candidate selection
      |
      v
validated ontology-grounded annotations
```

The model may generate a normalized search phrase such as `domestic cat` or `urinary tract infection`, but a final ontology ID is valid only if that ID was actually retrieved from the local ontology index.

## Scientific context

GenEpiO is being used as the initial ontology knowledge layer because it is specifically aimed at genomic epidemiology and imports relevant terms from multiple OBO ontologies, including concepts from resources such as NCBITaxon, DOID, ENVO, OBI, FOODON and others.

The long-term goal may be expanded retrieval across all ontologies in EMBL-EBI OLS. Therefore, avoid assumptions in the code that every identifier starts with `GENEPIO:`. Imported IDs such as `NCBITaxon:9685` and `DOID:13148` are expected and desirable.

Examples:

```text
raw: isolation_source=cat
semantic interpretation: organism -> domestic cat
ontology grounding: NCBITaxon:9685 / Felis catus
```

```text
raw: isolation_source=urine from woman with recurrent UTI
semantic concepts:
  - specimen -> urine
  - host -> Homo sapiens
  - disease -> urinary tract infection
  - recurrence may be represented if an adequate ontology term exists
```

The raw string should always be retained. Annotation is additive, not destructive normalization.

## Important distinction: factual annotations vs derived interpretations

Keep two conceptual layers separate.

### Layer 1: ontology-grounded factual concepts

Examples:

- `Felis catus`
- `Homo sapiens`
- urine
- blood
- urinary tract infection
- rectum
- wastewater
- poultry farm
- chicken meat

These should be grounded to actual ontology terms where possible.

### Layer 2: derived classifications

Examples:

- infection vs carriage/colonisation
- clinical vs non-clinical
- bloodstream infection
- urinary tract infection sample
- live animal vs food product
- hospital vs community

These may require reasoning across multiple Layer-1 annotations and the original metadata. Do **not** silently collapse Layer 2 into Layer 1. Derived claims should eventually have explicit provenance/rules/evidence.

For the current prototype, focus primarily on Layer 1.

## Current implementation

Main program: `genepio_rag.py`

Current stages:

1. `download`
   - downloads `genepio-full.owl`.

2. `build`
   - parses OWL with `rdflib`;
   - extracts ontology ID, IRI, label, synonyms, definition and direct parents;
   - builds word and character TF-IDF retrieval indices;
   - exact label/synonym matches receive a strong score bonus.

3. `query`
   - retrieves ontology candidates without loading the LLM;
   - useful for debugging retrieval quality.

4. `annotate`
   - first Ministral pass extracts concepts from the whole metadata record;
   - each concept is independently retrieved against the ontology index;
   - second Ministral pass chooses among retrieved candidates;
   - code validates that any returned ontology ID is actually in that candidate list.

5. `annotate-tsv`
   - runs the same process on tabular metadata;
   - outputs JSONL because one source field may yield several concepts and annotations.

## Local model

Use the already available model:

```text
mistralai/Ministral-3-8B-Instruct-2512
```

The current implementation uses the Mistral-supported Transformers interface:

```python
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
```

Do not replace this model simply because another model is easier to load. Changes of model should be explicit experiments.

Keep inference deterministic by default:

```text
temperature = 0
```

For throughput experiments, vLLM/OpenAI-compatible serving is a reasonable future backend, but the semantic pipeline should not depend on a particular inference backend.

## LLM responsibilities

The first LLM pass should:

- read the complete metadata record, not only one isolated field;
- identify biologically meaningful concepts;
- split compound free text into multiple concepts when justified;
- preserve the source field and original raw value;
- generate a concise normalized search query for each concept;
- avoid unsupported interpretation;
- avoid accessions, dates and administrative values unless biologically meaningful;
- never generate ontology identifiers.

Example first-pass output:

```json
{
  "concepts": [
    {
      "concept_type": "specimen",
      "source_field": "isolation_source",
      "raw_value": "urine from woman with recurrent UTI",
      "normalized_query": "urine",
      "evidence": "isolation_source=urine from woman with recurrent UTI"
    },
    {
      "concept_type": "organism",
      "source_field": "isolation_source",
      "raw_value": "urine from woman with recurrent UTI",
      "normalized_query": "Homo sapiens",
      "evidence": "woman"
    },
    {
      "concept_type": "disease",
      "source_field": "isolation_source",
      "raw_value": "urine from woman with recurrent UTI",
      "normalized_query": "urinary tract infection",
      "evidence": "recurrent UTI"
    }
  ]
}
```

The second LLM pass should:

- receive the original metadata context;
- receive each extracted concept;
- receive retrieved ontology candidates;
- select only an explicitly supplied candidate ID;
- choose `unresolved` when no candidate is adequate;
- prefer semantic precision over lexical similarity;
- return a brief reason.

## Non-negotiable invariants

Preserve these unless explicitly instructed otherwise:

1. **No hallucinated ontology IDs.**
   A final selected ID must exist in the candidate list supplied by retrieval.

2. **Preserve raw metadata.**
   Never replace the original record with normalized text.

3. **Unresolved is valid.**
   Do not force an ontology mapping merely to maximize coverage.

4. **One field may contain multiple concepts.**
   Do not impose one-field -> one-term mapping.

5. **Context matters.**
   `cat`, `chicken`, `culture`, `blood`, `stool`, etc. can be ambiguous. Whole-record context should be available to the semantic parser and grounding step.

6. **Imported ontologies are first-class.**
   NCBITaxon/DOID/ENVO/etc. IDs are not inferior to GENEPIO IDs.

7. **Separate extraction errors from retrieval errors.**
   Outputs should make it possible to determine whether the LLM extracted the wrong concept or retrieval failed to find the appropriate ontology term.

8. **Avoid over-annotation.**
   Only annotate concepts stated or strongly implied by the metadata.

9. **Machine-readable output first.**
   JSON/JSONL/Parquet should be preferred over prose output.

10. **Provenance should be retained.**
    Maintain source field, raw value, normalized query, selected ontology term, and ideally retrieval score/method.

## Retrieval strategy

The current TF-IDF retrieval is an intentionally simple baseline, not the final design.

It combines:

- word n-grams;
- character n-grams;
- the raw concept value;
- field context;
- exact label/synonym bonus.

This baseline is valuable because it is deterministic and interpretable.

Future comparisons should include semantic embedding retrieval, potentially using:

- EMBL-EBI OLS downloadable embeddings;
- a biomedical embedding model;
- FAISS;
- pgvector;
- another approximate-nearest-neighbour index.

Do not remove the lexical baseline when adding embeddings. Keep it available as a benchmark and potentially use hybrid retrieval.

## Full-OLS direction

The likely next major architecture is:

```text
all OLS ontology terms
      |
      +-- labels
      +-- synonyms
      +-- definitions
      +-- ontology source
      +-- parent/ancestor relationships
      +-- cross-ontology mappings
      |
      v
local lexical + vector index
```

GenEpiO should therefore be treated as the prototype corpus, not as a hard-coded assumption throughout the codebase.

A future term schema should be able to represent at least:

```json
{
  "id": "NCBITaxon:9685",
  "iri": "http://purl.obolibrary.org/obo/NCBITaxon_9685",
  "ontology": "NCBITaxon",
  "label": "Felis catus",
  "synonyms": ["cat", "cats", "domestic cat"],
  "definition": "...",
  "parents": ["..."],
  "ancestors": ["..."]
}
```

## Data scale

The intended eventual scale is very large: potentially millions of BioSample/ENA records.

Therefore:

- avoid loading/reloading Ministral once per row;
- reuse the model and ontology index across records;
- support batching where possible;
- cache repeated normalized-query retrievals;
- cache repeated exact mappings;
- keep retrieval local;
- avoid remote API calls in the core production pipeline;
- write streaming JSONL or Parquet rather than accumulating all results in memory;
- make restart/resume possible for large runs.

Do not prematurely optimize before correctness is established, but avoid designs that obviously require one expensive initialization per record.

## Test cases that should always work

Use these as smoke tests.

### Cat host/source

Input:

```text
isolation_source=cat
```

Expected ontology candidate/mapping:

```text
NCBITaxon:9685 / Felis catus
```

GenEpiO's imported NCBITaxon content contains synonyms including `cat`, `cats`, and `domestic cat`.

### UTI

Input:

```text
isolation_source=urine from woman with recurrent UTI
```

Expected decomposition should include at least:

```text
specimen -> urine
host -> Homo sapiens
condition/disease -> urinary tract infection
```

The GenEpiO import includes `DOID:13148` for urinary tract infection.

### Food-context ambiguity

Input:

```text
isolation_source=chicken breast from supermarket
```

The system should not blindly annotate this as a live chicken host only. Food/product context must be preserved and ideally represented by an appropriate food concept when available.

### Carriage-context ambiguity

Input:

```text
isolation_source=rectal swab
study_title=ESBL carriage in healthy volunteers
```

Do not infer active infection. A later Layer-2 classifier may infer carriage, but Layer 1 should preserve rectal/anatomical/specimen and healthy/carriage evidence if appropriate ontology terms are available.

### Blood clinical context

Input:

```text
isolation_source=blood culture
sample_title=E. coli from septic patient
```

Expected concepts may include blood/blood specimen, Homo sapiens/patient context, and sepsis if explicitly supported. Do not invent `bloodstream infection` unless the metadata supports that derived conclusion.

## Evaluation philosophy

Measure the stages separately.

### Concept extraction evaluation

Did Ministral identify the correct concepts from the metadata?

### Retrieval evaluation

Given the correct normalized query, is the correct ontology term present in top-k candidates?

Useful metric:

```text
Recall@k
```

### Grounding evaluation

When the correct term is present among candidates, does Ministral select it?

### End-to-end evaluation

Does the final annotation contain the correct grounded concepts with acceptable false-positive rate?

Coverage alone is not sufficient. False ontology annotations can be more damaging than unresolved values.

## Recommended development order

Unless instructed otherwise, work in this order:

1. Get the current GenEpiO full index running on the GPU workstation.
2. Verify `query` retrieval against known terms.
3. Run `annotate` on a handful of manually inspected records.
4. Create a small gold-standard benchmark of ugly metadata values/records.
5. Measure extraction, retrieval and grounding separately.
6. Improve retrieval (hybrid lexical + embeddings) if Recall@k is limiting.
7. Improve prompts/structured generation if extraction or grounding is limiting.
8. Only then scale to larger BioSample datasets.
9. Expand from GenEpiO to all OLS ontologies after the architecture is validated.

## Coding expectations

- Python 3.11+ preferred.
- Keep functions testable and separate LLM, ontology parsing, retrieval and I/O concerns.
- Add type hints when touching code.
- Prefer explicit schemas/dataclasses/Pydantic when output structures stabilize.
- Use deterministic random seeds for any stochastic benchmark.
- Avoid hidden remote dependencies in tests.
- Keep command-line operation available even if modules/classes are refactored.
- Do not delete debug/provenance information merely to simplify output.
- For large-data processing, prefer iterators/chunks and Parquet/JSONL.

## Environment/setup on the GPU workstation

Recommended PyCharm workflow:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

If the installed Transformers release does not yet contain the required Ministral-3 classes, install current Transformers from GitHub as documented in `README.md`.

Verify CUDA first:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
PY
```

Then build the ontology index:

```bash
python genepio_rag.py download
python genepio_rag.py build --owl genepio-full.owl --index genepio_rag.joblib
```

Test retrieval before loading the LLM:

```bash
python genepio_rag.py query --index genepio_rag.joblib --field isolation_source --value cat
```

Then test the LLM path:

```bash
python genepio_rag.py annotate \
  --index genepio_rag.joblib \
  --meta 'isolation_source=urine from woman with recurrent UTI' \
  --include-candidates \
  --debug
```

## What Codex should do first when opened in PyCharm

Before making substantial changes:

1. Read this `AGENTS.md` completely.
2. Read `README.md`.
3. Read `genepio_rag.py` end to end.
4. Inspect `requirements.txt`.
5. Verify the active Python interpreter and CUDA/PyTorch availability.
6. Check whether `mistralai/Ministral-3-8B-Instruct-2512` is already present in the local Hugging Face cache or available at the configured local model path.
7. Run the retrieval smoke test before trying full LLM inference.
8. Do not redesign the pipeline until baseline outputs have been recorded.

When reporting a bug or proposed change, identify which stage is failing:

```text
concept extraction
ontology retrieval
candidate ranking
grounding
ID validation
I/O / serialization
model loading / GPU
```

This separation is central to the project.

## Current research question

The prototype is intended to determine whether a small local reasoning/instruction model plus ontology retrieval can convert poor public sequence metadata into a reusable ontology-enhanced semantic resource accurately enough to support downstream queries.

The important comparison is not merely whether the model can label one field. It is whether the resulting grounded representation is accurate, auditable, scalable, and useful for multiple downstream biological/epidemiological questions.
