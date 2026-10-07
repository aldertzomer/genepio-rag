# Recorded baseline — 2026-10-07

Environment: project `.venv`, Python 3.12.15, PyTorch 2.14.1+cu126,
NVIDIA GeForce RTX 4070 Ti SUPER (16 GB), driver 560.35.03.
PyCharm's environment tool still reports the old Python 3.8 version metadata;
the executable at `.venv/bin/python` is verified as Python 3.12.15.

Installed requirements.txt plus Transformers main, current mistral-common main,
accelerate, and kernels==0.17.0. See requirements-installed.txt for exact versions.
The initial automatic PyCharm setup selected incompatible Python 3.8; the venv
was replaced using a project-local uv bootstrap and project-local Python 3.12.
CUDA 13 PyTorch was replaced with CUDA 12.6 to match the existing driver.
No source code or pipeline changes were made.

Downloaded genepio-full.owl from the program's default URL, then built
9,815 labelled classes into genepio_rag.joblib. The full file includes imported
ontology subsets; no separate import downloads were needed.

Retrieval top results (see retrieval.json):
- cat: NCBITaxon:9685 / Felis catus
- urine: UBERON:0001088 / urine
- Homo sapiens: NCBITaxon:9606 / Homo sapiens
- urinary tract infection: DOID:0080784 / urinary tract infection

The AGENTS.md expectation DOID:13148 was absent from the top-8 UTI results;
this downloaded ontology instead returns DOID:0080784 as the exact match.
The initial assertion of the older ID failed; this is an expectation/corpus
mismatch, not a failed retrieval of the disease label.

README end-to-end cat record completed successfully with the existing cached
mistralai/Ministral-3-8B-Instruct-2512, default temperature 0.
See annotate-cat.json for machine-readable output, candidates and raw model
responses, and annotate-cat.log for the complete run log.
All selected matched IDs were checked against each concept's candidate list.
Cat mapped to NCBITaxon:9685; Netherlands to GAZ:00002946; tax_id=197 to
NCBITaxon:197. The compound Campylobacter infection/transmission query was
unresolved. That compound extraction is an extraction-quality limitation.
