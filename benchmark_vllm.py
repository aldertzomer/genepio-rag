"""Benchmark unchanged prompts on the first three TSV records and cyclic repeats."""
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from genepio_rag import (MinistralLocal, EXTRACT_SYSTEM, GROUND_SYSTEM,
                         record_to_text, extract_json, retrieve_for_concepts,
                         ministral_ground)

OUTPUT = Path('baseline-results/vllm-benchmark')

class PromptCapture:
    def __init__(self) -> None:
        self.user = ''
    def chat(self, system: str, user: str) -> str:
        self.user = user
        return '{"annotations": []}'

class Replay:
    def __init__(self, text: str) -> None:
        self.text = text
    def chat(self, system: str, user: str) -> str:
        return self.text

def measure(llm: MinistralLocal, system: str, users: list[str]) -> tuple[dict[str, Any], list]:
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    def monitor() -> None:
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                values = [float(v) for v in result.stdout.strip().split(',')]
                samples.append({'time': time.time(), 'utilization_percent': values[0], 'used_mib': values[1], 'total_mib': values[2]})
            stop.wait(0.2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        outputs = llm.chat_batch(system, users)
        elapsed = time.perf_counter()-start
    finally:
        stop.set()
        thread.join()
    prompts = sum(len(o.prompt_token_ids) for o in outputs)
    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    stats = {'pass': 'extraction' if system == EXTRACT_SYSTEM else 'grounding',
             'records': len(users), 'prompt_tokens': prompts, 'generated_tokens': generated,
             'wall_seconds': elapsed, 'generated_tokens_per_second': generated/elapsed,
             'total_tokens_per_second': (prompts+generated)/elapsed,
             'records_per_second': len(users)/elapsed,
             'gpu_utilization_mean_percent': statistics.mean(s['utilization_percent'] for s in samples) if samples else None,
             'gpu_utilization_peak_percent': max((s['utilization_percent'] for s in samples), default=None),
             'peak_vram_mib': max((s['used_mib'] for s in samples), default=None),
             'length_stops': sum(o.outputs[0].finish_reason == 'length' for o in outputs),
             'samples': samples}
    return stats, outputs

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = 'run_accession host_scientific_name isolation_source host_tax_id sample_title study_title host_status tax_id'.split()
    source = pd.read_csv('benchmark.tsv', sep='\t', dtype=str).fillna('').head(3)
    records = [{f: row[f] for f in fields} for _, row in source.iterrows()]
    baseline = [json.loads(line) for line in Path('annotations.jsonl').read_text().splitlines()][:3]
    idx = joblib.load('genepio_rag.joblib')
    extraction_users = ['Metadata record:\n'+record_to_text(r) for r in records]
    bundles = []
    grounding_users = []
    for record, old in zip(records, baseline):
        concepts = extract_json(old['debug']['extraction_raw'])['concepts']
        b = retrieve_for_concepts(idx, concepts)
        capture = PromptCapture()
        ministral_ground(capture, record, b)
        bundles.append(b)
        grounding_users.append(capture.user)
    load_start = time.perf_counter()
    llm = MinistralLocal(temperature=0.0)
    load_seconds = time.perf_counter()-load_start
    # Separate warmup from all measured scenarios.
    measure(llm, EXTRACT_SYSTEM, [extraction_users[0]])
    results = []
    raw = []
    def run(mode: str, indices: list[int], system: str, users: list[str]) -> None:
        stats, outputs = measure(llm, system, users)
        stats['mode'] = mode
        stats['record_indices'] = indices
        stats['parse_errors'] = 0
        for index, output in zip(indices, outputs):
            text = output.outputs[0].text
            error = None
            try:
                if system == EXTRACT_SYSTEM:
                    obj = extract_json(text)
                    if not isinstance(obj.get('concepts'), list): raise ValueError('Missing concepts list')
                else:
                    ministral_ground(Replay(text), records[index], bundles[index])
            except (ValueError, TypeError, AttributeError) as exc:
                error = str(exc)
                stats['parse_errors'] += 1
            raw.append({'mode':mode,'pass':stats['pass'],'record_index':index,'text':text,'finish_reason':output.outputs[0].finish_reason,'parse_error':error})
        results.append(stats)
        OUTPUT.joinpath('metrics.json').write_text(json.dumps({'model':llm.model_id,'load_seconds':load_seconds,'max_model_len':8192,'temperature':0.0,'extraction_max_tokens':128,'grounding_max_tokens':96,'grounding_input':'Original baseline extracted concepts, independently replayed to measure grounding even if extraction truncates. No substituted end-to-end results.','batch_inputs':'Cyclic repeats of the same first three records. Prefix caching disabled.','results':results},indent=2))
        OUTPUT.joinpath('responses.json').write_text(json.dumps(raw,indent=2))
        print(json.dumps({k:v for k,v in stats.items() if k!='samples'}),flush=True)
    for index in range(3):
        run('sequential',[index],EXTRACT_SYSTEM,[extraction_users[index]])
        run('sequential',[index],GROUND_SYSTEM,[grounding_users[index]])
    for size in [8,16,32]:
        indices = [i%3 for i in range(size)]
        run(f'batch_{size}',indices,EXTRACT_SYSTEM,[extraction_users[i] for i in indices])
        run(f'batch_{size}',indices,GROUND_SYSTEM,[grounding_users[i] for i in indices])

if __name__ == '__main__':
    main()
