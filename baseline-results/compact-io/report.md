# Compact schema-constrained LLM representations

Model: Ministral-3-8B-Instruct-2512 via vLLM; max_model_len=8192; temperature=0; output ceiling=768.
Batch 32 repeats the same three records (11/11/10 requests); three independent examples only.
No truncated or invalid JSON responses; all 96 measured requests finished with stop.

## Token distributions

| Pass | Min | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| extraction | 55 | 58 | 90 | 90 | 90 | 90 |
| grounding_fixed_concepts | 12 | 12 | 20 | 20 | 20 | 20 |
| grounding_end_to_end | 13 | 13 | 18 | 18 | 18 | 18 |

## Performance

| Pass | Mean prompt tokens/record | Mean generated tokens/record | Wall s (32 records) | Records/s |
|---|---:|---:|---:|---:|
| Saved valid verbose top-8 reference | 2822.3 | 295.6 | 41.234 | 0.776 |
| extraction | 298.2 | 67.2 | 2.849 | 11.234 |
| grounding_fixed_concepts | 1221.6 | 14.5 | 5.214 | 6.138 |
| grounding_end_to_end | 1109.9 | 14.6 | 4.700 | 6.809 |

Grounding with fixed concepts is 7.91x faster than the saved verbose reference; this is a cross-run comparison, not simultaneous trials.

## Acceptance

Fixed-concept grounding IDs/statuses: 32/32 (100%) agree with the valid 768-token reference.
Full two-pass IDs/statuses: 13/32 records agree; **full acceptance fails**.
Compact extraction changes the concept set/search queries; record 3 has five rather than six concepts.
No reference results or IDs are inserted into compact inference outputs.
The script exits nonzero when either fixed-concept or full two-pass parity fails.
Only the fixed-concept grounding representation has passed the parity requirement.

## Proposed limits, not applied

Provisional limits use 25% headroom above the observed P99, rounded up to a multiple of 16 tokens.
- extraction: P99=90; provisional cap 128 tokens.
- grounding_fixed_concepts: P99=20; provisional cap 32 tokens.

Retain the 768-token ceiling until full semantic parity and a diverse, larger benchmark pass.
These percentiles reflect only three unique records, so they are not production-scale tail estimates.
For grounding, size limits with concept count: the observed six-concept output is 20 tokens; longer lists require a dynamic allowance.
No production limit change was made.

## Reconstruction and safeguards

Extraction tuple fields are constrained by schema; source_field must belong to input metadata.
Raw values are recovered from the input; evidence is the full source field/value, not an invented evidence span.
Grounding input contains top-5 numbered candidates without IDs; output contains one constrained index/null per concept.
Python constructs canonical ID, label and retrieval_score from its candidate list; reasons describe the selected index deterministically.
All existing final JSON fields remain present; retrieval_score is additive.
Invalid indices, booleans, floats and IDs are rejected by the Python validation layer even if schema constraints are bypassed.
Four offline reconstruction/schema/validation tests passed.
AST comparison confirms ontology parsing, TF-IDF scoring and record formatting are unchanged.
