# PyCharm + Codex setup

1. Unzip/open this directory as a PyCharm project.
2. Create or select a Python 3.11+ interpreter for the project.
3. Install `requirements.txt`.
4. Ensure PyCharm/Codex treats the repository root as its working directory.
5. `AGENTS.md` is the authoritative project instruction file for Codex. Ask Codex to read it before editing anything.
6. Verify GPU/PyTorch, then run the retrieval-only smoke test before loading Ministral.

Suggested first Codex prompt:

> Read AGENTS.md, README.md, requirements.txt and genepio_rag.py completely. Do not change code yet. Summarize the pipeline, list the non-negotiable invariants, inspect the current environment for CUDA and the local Ministral-3-8B-Instruct-2512 model, then propose the smallest sequence of tests needed to validate the prototype on this machine. Distinguish failures in concept extraction, ontology retrieval, grounding, ID validation and model/GPU setup.

Suggested second prompt after the environment check:

> Run the retrieval smoke tests in AGENTS.md, then run one end-to-end annotation using `isolation_source=urine from woman with recurrent UTI`. Preserve all raw outputs and report retrieved candidates separately from the model's final selections. Do not modify the architecture unless a test fails; if it fails, diagnose the failing stage first.
