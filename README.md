# Local GenEpiO RAG + Ministral-3-8B-Instruct-2512

This prototype turns messy sample metadata into **ontology-grounded annotations** using:

1. local GenEpiO / imported ontology content;
2. `mistralai/Ministral-3-8B-Instruct-2512` to extract concepts;
3. local retrieval of candidate ontology terms;
4. a second Ministral pass that can select **only retrieved ontology IDs**.

The raw metadata is retained. The model is allowed to generate normalized search phrases, but ontology IDs are supplied by the ontology index and validated after generation.

## 1. Install

Inference uses vLLM with the official Mistral checkpoint format. Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

If the model is already in your Hugging Face cache, the default model ID will use it. You can alternatively give `--model /path/to/model`.

## 2. Download and index GenEpiO

```bash
python genepio_rag.py download
python genepio_rag.py build --owl genepio-full.owl --index genepio_rag.joblib
```

`genepio-full.owl` contains GenEpiO plus its imported ontology subset, which is important here: host taxa, diseases and other useful concepts may originate from NCBITaxon, DOID, ENVO, etc.

## 3. Test retrieval without the LLM

```bash
python genepio_rag.py query \
  --index genepio_rag.joblib \
  --field isolation_source \
  --value cat
```

You should see `NCBITaxon:9685 / Felis catus` near or at the top because GenEpiO imports `cat`, `cats`, and `domestic cat` as synonyms.

## 4. End-to-end single record

```bash
python genepio_rag.py annotate \
  --index genepio_rag.joblib \
  --meta 'run_accession=ERR4224313' \
  --meta 'isolation_source=cat' \
  --meta 'study_title=Discerning Environmental Pathways of Campylobacter Transmission' \
  --meta 'country=Netherlands' \
  --meta 'tax_id=197' \
  --include-candidates
```

A more complex example:

```bash
python genepio_rag.py annotate \
  --index genepio_rag.joblib \
  --meta 'organism=Escherichia coli' \
  --meta 'host=patient' \
  --meta 'isolation_source=urine from woman with recurrent UTI'
```

The first Ministral pass should decompose this into concepts such as urine, human/female host, and urinary tract infection. Each is then searched independently.

## 5. TSV batch test

```bash
python genepio_rag.py annotate-tsv benchmark.tsv \
  --index genepio_rag.joblib \
  --fields run_accession host_scientific_name isolation_source host_tax_id sample_title study_title host_status tax_id \
  --limit 20 \
  --output annotations.jsonl
```

Output is JSONL: one record per input row. This avoids flattening multiple ontology annotations into hundreds of TSV columns.

## Useful options

```text
--model PATH_OR_HF_ID       default: mistralai/Ministral-3-8B-Instruct-2512
-k 5                        ontology candidates per concept (grounding uses top five)
--temperature 0             deterministic extraction/grounding
--max-new-tokens N           optional override of both pass limits
--include-candidates        retain retrieved candidates in output
--debug                     retain raw Ministral responses
```

## Output shape

Abbreviated example:

```json
{
  "metadata": {
    "isolation_source": "cat"
  },
  "concepts": [
    {
      "concept_type": "organism",
      "source_field": "isolation_source",
      "raw_value": "cat",
      "normalized_query": "domestic cat",
      "evidence": "isolation_source=cat",
      "annotation": {
        "status": "matched",
        "ontology_id": "NCBITaxon:9685",
        "label": "Felis catus",
        "reason": "The metadata explicitly refers to a cat."
      }
    }
  ]
}
```

## Design safeguards

* The LLM never supplies the authoritative ontology identifier.
* A selected ID is accepted only when it is present in the candidate list returned by local retrieval.
* Canonical labels are replaced with the value stored in the ontology index.
* Unresolved is a valid answer.
* Original metadata is retained.

## Current retrieval method

The prototype deliberately starts with word + character TF-IDF rather than embeddings. Exact labels/synonyms receive a strong bonus. This provides an interpretable baseline before comparing an embedding index using OLS embeddings, SapBERT, BGE, etc.

## Compact LLM representations and GPU backend

Inference uses vLLM with `max_model_len=8192`, temperature 0.0 and the official
Mistral tokenizer/config/load formats. The engine supports 32 concurrent
sequences, reserves 85% of GPU memory, and disables image inputs and prefix caching.

Both passes use vLLM JSON-schema-constrained output. Extraction returns only:

```json
{"c": [["isolation_source", "organism", "domestic cat"]]}
```

Python reconstructs the source raw value and field-level evidence from the
original metadata. Grounding receives compact top-5 candidates numbered 0–4,
with labels, up to eight synonyms, retrieval scores and at most one 160-character
definition per concept. It receives no ontology-ID fields and returns only:

```json
{"s": [0, null]}
```

Selections follow concept order. Python validates each index and reconstructs
canonical IDs, labels, scores, statuses and deterministic reasons from the
retrieved candidates. Verbose final JSON/JSONL fields are retained;
`annotation.retrieval_score` is additive. The ontology parser and retrieval
scoring are unchanged.

The default measurement ceiling is 768 tokens for both passes. Truncated
responses are refused. Run the batch-32 representation benchmark:

```bash
python benchmark_compact_io.py
python -m unittest test_compact_io
```

Results are in `baseline-results/compact-io/`, including token distributions,
verbose reconstructed annotations and comparison with the saved valid 768-token
reference. The script exits nonzero if fixed-concept grounding or full two-pass
IDs/statuses differ from the reference.

The recorded run passed fixed-concept grounding parity (32/32 requests), but
failed full two-pass parity (13/32). Compact extraction changes the concept set.
This remains an experimental refactor pending that acceptance requirement.

Measured P99 output lengths were 90 tokens for extraction and 20 for grounding
with fixed concepts. Provisional caps of 128/32 follow 25% P99 headroom rounded
up to multiples of 16; they have not been applied. These measurements repeat
only three unique records and do not establish production tail behavior.

Historical verbose-representation benchmarks and response audits remain in
`baseline-results/vllm-benchmark/` and `baseline-results/grounding-comparison/`.

## Exploratory BioSample sample test

`biosample_sample_test.py` selects a reproducible random sample from a local
NCBI `biosample_set.xml.gz` (also plain XML) or `biosample.parquet`. It uses file
magic to autodetect the format; `--format xml` or `--format parquet` overrides it.
Defaults are `--n 200`, `--seed 12345`, and `--min-nonempty-fields 0`. With the
zero threshold, every input BioSample row is eligible, including sparse records.
Sampling is without replacement, and fewer than `n` eligible rows yields all
available eligible rows. The resulting sample is saved in source-row order.

The following command **does not load the model, ontology index or GPU**:

```bash
python biosample_sample_test.py biosample.parquet \
  --n 200 --seed 12345 --sample-only \
  --output-dir sample-test-runs/metadata-only
```

When ready to run inference, omit `--sample-only`:

```bash
python biosample_sample_test.py biosample_set.xml.gz \
  --n 200 --seed 12345 --index genepio_rag.joblib \
  --output-dir sample-test-runs/xml-exploration

python biosample_sample_test.py biosample.parquet \
  --n 200 --seed 12345 --index genepio_rag.joblib \
  --output-dir sample-test-runs/parquet-exploration
```

XML uses streaming parsing plus uniform reservoir sampling. Completed BioSample
elements are removed from their parent, keeping memory bounded by one record,
parser buffers and the selected sample. Reading the whole XML stream is necessary
to obtain a uniform reservoir sample.

For unfiltered Parquet, a seeded RNG samples global row positions using row-count
metadata. PyArrow reads only the selected row groups in bounded record batches;
all columns are retained, and only selected rows are converted to Python objects.
A positive `--min-nonempty-fields` requires scanning records to evaluate the
threshold and uses reservoir sampling. The threshold counts all non-empty
canonical metadata fields, including administrative fields and original-name
aliases. XML and Parquet sampling algorithms can select different records for
the same seed; each is reproducible for an unchanged input and format.

Both adapters produce the canonical structure:

```json
{
  "accession": "SAMN00000001",
  "metadata": {
    "accession": "SAMN00000001",
    "host": "cat",
    "original_attributes.Host": "cat",
    "organism": "Campylobacter jejuni",
    "some_unanticipated_field": "retained value"
  }
}
```

All non-empty fields are retained, with no whitelist for LLM input. Actual nulls,
NaN and blank values are removed; zero, false and literal missingness descriptions
such as `NA` or `not collected` are retained in saved metadata. The production
prompt formatter's existing handling of missing-value literals is unchanged.
Repeated values are joined with ` | `. Nested Parquet columns are flattened to
field paths. Sparse `attributes`, `identifiers` and `links` maps from the supplied
XML-to-Parquet converter are expanded; original attribute names remain under
`original_attributes.*`. XML structural names match that converter's columns;
extra XML text and attributes are retained under dotted paths. Information already
omitted by an upstream converter cannot be recovered from its Parquet file.

The workflow reuses the current model prompts, schema constraints, concept
reconstruction, GenEpiO top-5 retrieval, grounding and annotation reconstruction.
The model/index initialize once per run. It does not change production semantics.
Both LLM stages retain prompt/generated token counts, inference wall time,
`finish_reason`, truncation (`finish_reason == "length"`), strict JSON validity,
schema validity, raw response and model errors. Truncated, malformed,
schema-invalid and aborted responses fail that record; subsequent records
continue. Later stages are explicitly marked skipped if prerequisites fail.
Extracted concepts and retrieved candidates are retained if grounding fails;
failed calls are not mislabeled as unresolved ontology mappings.

Every run saves these files in a **new output directory**:

- `sampled_accessions.txt`: one line per selected row; a blank marks a missing accession.
- `sampled_metadata.jsonl`: canonical records, including all retained metadata.
- `enrichment_results.jsonl`: complete verbose results and per-stage diagnostics.
- `summary.json`: sampling provenance, processed/completed/failed counts, concepts
  per record, total concepts, matched/unresolved counts, timings, records/sec,
  token totals/distributions and stage failure/truncation rates.

Failure-rate denominators are attempted calls for that stage; skipped stages
are excluded. JSON/schema validity is null when it could not be evaluated,
which is reported separately and never counted as a pass. JSON/schema validity
is checked even for truncated output, but truncation always prevents acceptance.
Records/sec excludes sampling and model/index initialization; those timings
and total run wall time are recorded separately. Results stream to JSONL and
an atomic partial summary is updated after every record. Initialization errors
are saved against each sampled row; the CLI exits nonzero if any record failed.
In `--sample-only` mode the enrichment file is empty and processed records is zero.

This is qualitative exploration of public metadata, **not an accuracy benchmark**.
No real model/GPU inference was performed while implementing this workflow.
Run its offline tests with:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest test_biosample_sample_test test_compact_io
```
