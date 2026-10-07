# Response audit and compact grounding comparison

Production code and architecture were not changed.
Model: mistralai/Ministral-3-8B-Instruct-2512; max_model_len=8192; temperature=0.
Batch size 32, cyclic repeats of first three TSV records (11, 11, 10 occurrences).
Single timed batch per condition, after warmup; prefix caching disabled.
Grounding uses the original baseline extracted concepts because extraction at 128 tokens fails.

## Existing response audit

All 118 responses stopped with finish_reason=length: 59 extraction and 59 grounding.
JSON validity: 0/118. Truncation rate: 100% in both passes. Final ]} delimiters: absent in every response.
No extraction yields a usable parsed concepts list. Each extraction has one complete concept object inside an unfinished list, followed by a truncated concept.
Grounding has one complete partial annotation object in 39 responses and none in 20.
Partial objects are diagnostics only; no truncated response was accepted.
See response-audit.md and response-audit.json for every individual request.

## Grounding comparisons

Compact candidates retain only ID, label, up to eight synonyms and score.
At most one candidate per concept retains a definition, capped at 160 characters.
Concept fields, original metadata, system prompt, output schema and validator remain unchanged.
Payload JSON is minified; compact variants jointly change formatting, definition length and top-k.

| Output cap | Representation | Prompt tokens/record (1 / 2 / 3) | Batch mean prompt tokens/record | Wall s | Records/s | Valid and complete | Concept agreement | Exact record agreement |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 96 | current_top8 | 2108 / 2145 / 4353 | 2822.3 | 19.58 | 1.634 | 0/32 | N/A | N/A |
| 96 | compact_top3 | 741 / 756 / 1441 | 964.9 | 7.89 | 4.058 | 0/32 | N/A | N/A |
| 96 | compact_top5 | 996 / 1039 / 2010 | 1327.7 | 9.61 | 3.329 | 0/32 | N/A | N/A |
| 768 | current_top8 | 2108 / 2145 / 4353 | 2822.3 | 41.23 | 0.776 | 32/32 | 100.0% | 100.0% |
| 768 | compact_top3 | 741 / 756 / 1441 | 964.9 | 18.30 | 1.748 | 32/32 | 91.3% | 65.6% |
| 768 | compact_top5 | 996 / 1039 / 2010 | 1327.7 | 24.15 | 1.325 | 32/32 | 100.0% | 100.0% |

At 96 tokens, every response is truncated, so annotation agreement is undefined.
The 768-token runs are an explicit benchmark-only control; no production defaults were changed.
All 768-token responses stop normally and include complete annotations for every supplied concept.
Agreement compares validated (status, ontology_id) pairs, excluding reason text and label prose.
Top-5 matches both the current vLLM top-8 control and the original Transformers annotations exactly.
Top-3 differs on concept 0 of record 2: NCBITaxon:197 (Campylobacter jejuni) becomes NCBITaxon:194 (Campylobacter).
NCBITaxon:197 remains in that top-3 candidate list, so this difference is a contextual selection change, not removal of the previous selected ID.
All matched IDs were checked against each variant's own candidate lists; zero IDs rejected.
Batch counts are repeated requests, not 32 independent biological records; this is not an accuracy benchmark.
Peak observed total GPU VRAM was 15011 MiB (~14.66 GiB).

Compact top-5 reduced mean prompt tokens by 53.0% and valid-response wall time by 41.4% (~1.71x throughput).
Compact top-3 reduced mean prompt tokens by 65.8% and valid-response wall time by 55.6% (~2.25x throughput), with an annotation change.
