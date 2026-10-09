"""Exploratory BioSample sample-test workflow; importing never loads an LLM/GPU."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from biosample_sampling import CanonicalRecord, sample_input


class StageFailure(ValueError):
    """An observed model response cannot be passed to reconstruction."""


def stage_diagnostics() -> dict[str, Any]:
    return {'attempted': False, 'skipped_reason': None, 'prompt_tokens': None,
            'generated_tokens': None, 'wall_seconds': 0.0, 'finish_reason': None,
            'truncated': None, 'json_valid': None, 'schema_valid': None,
            'raw_response': None, 'model_error': None, 'json_error': None,
            'schema_errors': [], 'reconstruction_error': None, 'accepted': False}


def inspect_response(raw: str, schema: dict, finish_reason: str | None) -> dict[str, Any]:
    diagnostics = stage_diagnostics()
    diagnostics.update(attempted=True, raw_response=raw, finish_reason=finish_reason,
                       truncated=finish_reason == 'length')
    try:
        # Fail closed on non-standard NaN/Infinity as well as malformed JSON.
        def reject_constant(value: str) -> None:
            raise ValueError(f'Non-standard JSON constant: {value}')
        obj = json.loads(raw, parse_constant=reject_constant)
        diagnostics['json_valid'] = True
    except (ValueError, TypeError) as exc:
        diagnostics.update(json_valid=False, json_error=str(exc))
        return diagnostics
    errors = sorted(Draft202012Validator(schema).iter_errors(obj), key=lambda error: str(error.path))
    diagnostics['schema_valid'] = not errors
    diagnostics['schema_errors'] = [error.message for error in errors]
    diagnostics['accepted'] = not errors and finish_reason == 'stop'
    return diagnostics


class DiagnosticLLM:
    """Observe the existing stage calls, preserving their prompts and schemas."""
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.last = stage_diagnostics()

    def chat(self, system: str, user: str, schema: dict | None = None) -> str:
        self.last = stage_diagnostics()
        self.last['attempted'] = True
        start = time.perf_counter()
        try:
            if schema is None:
                raise ValueError('Schema required')
            request = self.backend.chat_batch(system, [user], schemas=[schema])[0]
            output = request.outputs[0]
            elapsed = time.perf_counter() - start
        except Exception as exc:
            self.last.update(wall_seconds=time.perf_counter() - start,
                             model_error=f'{type(exc).__name__}: {exc}')
            raise StageFailure('Model call failed') from exc
        self.last = inspect_response(output.text, schema, output.finish_reason)
        self.last.update(prompt_tokens=len(request.prompt_token_ids) if request.prompt_token_ids is not None else None,
                         generated_tokens=len(output.token_ids), wall_seconds=elapsed)
        if not self.last['accepted']:
            raise StageFailure('Truncated, malformed, schema-invalid or incomplete model output')
        return output.text


def enrich_record(record: CanonicalRecord, index: Any, backend: Any,
                  retriever: Callable | None = None) -> dict[str, Any]:
    # These imports do not import vLLM or torch; model initialization is explicit in run().
    from genepio_rag import ministral_extract, ministral_ground, retrieve_for_concepts

    started = time.perf_counter()
    metadata = record['metadata']
    diagnostics = {'extraction': stage_diagnostics(), 'grounding': stage_diagnostics(),
                   'retrieval_seconds': 0.0, 'record_wall_seconds': 0.0}
    result = {'accession': record['accession'], 'metadata': metadata, 'concepts': [],
              'model': getattr(backend, 'model_id', None), 'status': 'failed',
              'failed_stage': None, 'error': None, 'diagnostics': diagnostics,
              'debug': {'extraction_raw': None, 'grounding_raw': None}}
    observed = DiagnosticLLM(backend)
    if not metadata:
        result.update(failed_stage='concept_extraction', error='No non-empty metadata fields')
        diagnostics['extraction']['skipped_reason'] = 'No non-empty metadata fields'
        diagnostics['grounding']['skipped_reason'] = 'Extraction not available'
        diagnostics['record_wall_seconds'] = time.perf_counter() - started
        return result
    stage = 'concept_extraction'
    try:
        try:
            concepts, extraction_raw = ministral_extract(observed, metadata)
        finally:
            diagnostics['extraction'] = observed.last.copy()
            result['debug']['extraction_raw'] = observed.last['raw_response']
        # Retain extracted concepts if later stages fail, without presenting model errors as unresolved mappings.
        result['concepts'] = [dict(concept) for concept in concepts]
        stage = 'ontology_retrieval'
        retrieval_start = time.perf_counter()
        try:
            bundles = (retriever or retrieve_for_concepts)(index, concepts, k=5)
        finally:
            diagnostics['retrieval_seconds'] = time.perf_counter() - retrieval_start
        for concept, bundle in zip(result['concepts'], bundles):
            concept['candidates'] = bundle['candidates']
        stage = 'grounding'
        if bundles:
            try:
                annotations, grounding_raw = ministral_ground(observed, metadata, bundles)
            finally:
                diagnostics['grounding'] = observed.last.copy()
                result['debug']['grounding_raw'] = observed.last['raw_response']
            for concept, annotation in zip(result['concepts'], annotations):
                concept['annotation'] = annotation
        else:
            diagnostics['grounding']['skipped_reason'] = 'No extracted concepts; production skips grounding'
            result['debug']['grounding_raw'] = '{"s":[]}'
        result['status'] = 'completed'
    except Exception as exc:
        result.update(failed_stage=stage, error=f'{type(exc).__name__}: {exc}')
        if stage in {'concept_extraction', 'grounding'}:
            name = 'extraction' if stage == 'concept_extraction' else 'grounding'
            if diagnostics[name]['accepted']:
                diagnostics[name].update(accepted=False, reconstruction_error=str(exc))
        if not diagnostics['grounding']['attempted']:
            diagnostics['grounding']['skipped_reason'] = f'Previous stage failed: {stage}'
    diagnostics['record_wall_seconds'] = time.perf_counter() - started
    return result


def numeric_statistics(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {'count': 0, 'total': 0, 'mean': None, 'min': None, 'median': None,
                'p90': None, 'p95': None, 'p99': None, 'max': None}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        pos = (len(ordered) - 1) * p
        left = int(pos)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (pos - left)
    return {'count': len(values), 'total': sum(values), 'mean': sum(values) / len(values),
            'min': ordered[0], 'median': percentile(.5), 'p90': percentile(.9),
            'p95': percentile(.95), 'p99': percentile(.99), 'max': ordered[-1]}


class Summary:
    """Aggregate diagnostics without retaining model responses/results in memory."""
    def __init__(self) -> None:
        self.processed = self.completed = 0
        self.concept_counts: list[int] = []
        self.annotations: Counter[str] = Counter()
        self.stages = {name: [] for name in ('extraction', 'grounding')}
        self.retrieval_seconds = 0.0
        self.record_seconds = 0.0
        self.failures: Counter[str] = Counter()

    def add(self, result: dict[str, Any]) -> None:
        self.processed += 1
        self.completed += result['status'] == 'completed'
        self.concept_counts.append(len(result['concepts']))
        if result['failed_stage']:
            self.failures[result['failed_stage']] += 1
        for concept in result['concepts']:
            if 'annotation' in concept:
                self.annotations[concept['annotation']['status']] += 1
        self.retrieval_seconds += result['diagnostics']['retrieval_seconds']
        self.record_seconds += result['diagnostics']['record_wall_seconds']
        for name in self.stages:
            # Only scalar summaries are retained, not raw responses/schema-error messages.
            d = result['diagnostics'][name]
            self.stages[name].append({key: d[key] for key in
                ('attempted', 'prompt_tokens', 'generated_tokens', 'wall_seconds',
                 'finish_reason', 'truncated', 'json_valid', 'schema_valid', 'model_error')})

    def as_dict(self, enrichment_wall_seconds: float) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for name, rows in self.stages.items():
            attempted = [d for d in rows if d['attempted']]
            n = len(attempted)
            def rate(count: int) -> float | None:
                return count / n if n else None
            failures = {key: sum(d[key] is False for d in attempted) for key in ('json_valid', 'schema_valid')}
            truncated = sum(d['truncated'] is True for d in attempted)
            wall = sum(d['wall_seconds'] for d in attempted)
            stages[name] = {'attempted_records': n, 'skipped_records': len(rows) - n,
                'truncated_records': truncated, 'truncation_rate': rate(truncated),
                'json_failures': failures['json_valid'], 'json_failure_rate': rate(failures['json_valid']),
                'schema_failures': failures['schema_valid'], 'schema_failure_rate': rate(failures['schema_valid']),
                'json_evaluated_records': sum(d['json_valid'] is not None for d in attempted),
                'schema_evaluated_records': sum(d['schema_valid'] is not None for d in attempted),
                'model_errors': sum(d['model_error'] is not None for d in attempted),
                'finish_reasons': dict(Counter(d['finish_reason'] for d in attempted if d['finish_reason'] is not None)),
                'wall_seconds': wall, 'records_per_second': n / wall if wall else None,
                'prompt_tokens': numeric_statistics([d['prompt_tokens'] for d in attempted if d['prompt_tokens'] is not None]),
                'generated_tokens': numeric_statistics([d['generated_tokens'] for d in attempted if d['generated_tokens'] is not None])}
        return {'processed_records': self.processed, 'completed_records': self.completed,
                'failed_records': self.processed - self.completed, 'failures_by_stage': dict(self.failures),
                'concepts_per_record': self.concept_counts, 'concept_count_statistics': numeric_statistics(self.concept_counts),
                'total_concepts': sum(self.concept_counts), 'matched_annotations': self.annotations['matched'],
                'unresolved_annotations': self.annotations['unresolved'],
                'concepts_without_annotations': sum(self.concept_counts) - sum(self.annotations.values()),
                'stages': stages, 'retrieval_wall_seconds': self.retrieval_seconds,
                'total_prompt_tokens': sum(stage['prompt_tokens']['total'] for stage in stages.values()),
                'total_generated_tokens': sum(stage['generated_tokens']['total'] for stage in stages.values()),
                'record_wall_seconds': self.record_seconds, 'enrichment_wall_seconds': enrichment_wall_seconds,
                'records_per_second': self.processed / enrichment_wall_seconds if enrichment_wall_seconds else None,
                'completed_records_per_second': self.completed / enrichment_wall_seconds if enrichment_wall_seconds else None,
                'failure_rate_denominator': 'Attempted calls for each stage. Null validity is untested, not a pass or failure.'}


def run(input_path: Path, output_dir: Path, n: int = 200, seed: int = 12345,
        input_format: str = 'auto', min_nonempty_fields: int = 0,
        index_path: Path = Path('genepio_rag.joblib'), model: str | None = None,
        max_new_tokens: int = 768, sample_only: bool = False,
        backend_factory: Callable[[], Any] | None = None, index_loader: Callable[[], Any] | None = None,
        retriever: Callable | None = None) -> dict[str, Any]:
    if n < 0 or min_nonempty_fields < 0 or max_new_tokens <= 0:
        raise ValueError('Invalid sample counts or token limit')
    from genepio_rag import DEFAULT_MODEL
    model = model or DEFAULT_MODEL
    # Avoid accidentally overwriting a previous experiment or its input.
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / filename for name, filename in
             [('accessions', 'sampled_accessions.txt'), ('metadata', 'sampled_metadata.jsonl'),
              ('results', 'enrichment_results.jsonl'), ('summary', 'summary.json')]}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Output directory already contains sample-test artifacts; choose a new directory')
    run_start = time.perf_counter()
    sampled, sampling = sample_input(input_path, n, seed, input_format, min_nonempty_fields)
    sampling_seconds = time.perf_counter() - run_start
    with paths['accessions'].open('w', encoding='utf-8') as fh:
        for record in sampled:
            fh.write((record['accession'] or '') + '\n')
    with paths['metadata'].open('w', encoding='utf-8') as fh:
        for record in sampled:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = Summary()
    initialization_seconds = enrichment_seconds = 0.0
    initialization_error = None
    if sample_only or not sampled:
        paths['results'].touch()
    else:
        from genepio_rag import DEFAULT_MODEL, MinistralLocal
        import joblib

        init_start = time.perf_counter()
        backend = index = None
        try:
            index = index_loader() if index_loader else joblib.load(index_path)
            backend = backend_factory() if backend_factory else MinistralLocal(
                model_id=model or DEFAULT_MODEL, max_new_tokens=max_new_tokens, temperature=0.0)
        except Exception as exc:
            initialization_error = f'{type(exc).__name__}: {exc}'
        initialization_seconds = time.perf_counter() - init_start
        enrichment_start = time.perf_counter()
        with paths['results'].open('w', encoding='utf-8') as fh:
            for record in sampled:
                if initialization_error:
                    d = {'extraction': stage_diagnostics(), 'grounding': stage_diagnostics(),
                         'retrieval_seconds': 0.0, 'record_wall_seconds': 0.0}
                    for stage in ('extraction', 'grounding'):
                        d[stage].update(skipped_reason='Initialization failed', model_error=initialization_error)
                    result = {'accession': record['accession'], 'metadata': record['metadata'],
                              'concepts': [], 'model': model or DEFAULT_MODEL, 'status': 'failed',
                              'failed_stage': 'initialization', 'error': initialization_error,
                              'diagnostics': d, 'debug': {'extraction_raw': None, 'grounding_raw': None}}
                else:
                    result = enrich_record(record, index, backend, retriever)
                fh.write(json.dumps(result, ensure_ascii=False) + '\n')
                fh.flush()
                summary.add(result)
                # Keep a usable partial summary if a long exploratory run is interrupted.
                partial = summary.as_dict(time.perf_counter() - enrichment_start)
                partial.update(run_status='running', sampling=sampling, sampled_records=len(sampled))
                write_summary(paths['summary'], partial)
        enrichment_seconds = time.perf_counter() - enrichment_start
    final = summary.as_dict(enrichment_seconds)
    final.update(run_status='sample_only' if sample_only else ('completed_with_errors' if summary.completed < len(sampled) else 'completed'),
                 purpose='Qualitative exploration; not an accuracy benchmark', input_path=str(input_path),
                 sampling=sampling, sampled_records=len(sampled),
                 missing_accessions=sum(not record['accession'] for record in sampled),
                 sampling_seconds=sampling_seconds, initialization_seconds=initialization_seconds,
                 initialization_error=initialization_error, total_wall_seconds=time.perf_counter() - run_start,
                 model=model, max_new_tokens=max_new_tokens, outputs={key: str(path) for key, path in paths.items()})
    write_summary(paths['summary'], final)
    return final


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Exploratory random BioSample enrichment; no accuracy claims')
    parser.add_argument('input', type=Path)
    parser.add_argument('--format', choices=['auto', 'xml', 'parquet'], default='auto')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--min-nonempty-fields', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=Path('genepio_rag.joblib'))
    parser.add_argument('--model', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=768)
    parser.add_argument('--sample-only', action='store_true', help='Sample and save metadata without loading ontology or model')
    args = parser.parse_args()
    summary = run(args.input, args.output_dir, args.n, args.seed, args.format,
                  args.min_nonempty_fields, args.index, args.model, args.max_new_tokens, args.sample_only)
    print(json.dumps({'run_status': summary['run_status'], 'sampled_records': summary['sampled_records'],
                      'processed_records': summary['processed_records'], 'summary': summary['outputs']['summary']}))
    if summary['failed_records']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
