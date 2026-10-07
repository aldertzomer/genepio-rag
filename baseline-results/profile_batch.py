"""Profile three records with the existing deterministic Transformers pipeline."""
import json
import subprocess
import threading
import time
from pathlib import Path
import statistics
import joblib
import pandas as pd
from genepio_rag import MinistralLocal, ministral_extract, retrieve_for_concepts, ministral_ground

out = Path('baseline-results')
samples = []
phase = 'initialization'
stop = threading.Event()
def monitor() -> None:
    while not stop.is_set():
        p = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        samples.append({'time':time.time(),'phase':phase,'gpu':p.stdout.strip()})
        stop.wait(1)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
idx = joblib.load('genepio_rag.joblib')
start = time.perf_counter()
llm = MinistralLocal()
load_seconds = time.perf_counter()-start
calls = []
original_generate = llm.model.generate
def timed_generate(*args, **kwargs):
    llm.torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_generate(*args, **kwargs)
    llm.torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    input_tokens = kwargs['input_ids'].shape[-1]
    generated_tokens = result.shape[-1]-input_tokens
    calls.append({'phase':phase,'seconds':seconds,'input_tokens':input_tokens,'generated_tokens':generated_tokens,'tokens_per_second':generated_tokens/seconds})
    return result
llm.model.generate = timed_generate
fields = 'run_accession host_scientific_name isolation_source host_tax_id sample_title study_title host_status tax_id'.split()
df = pd.read_csv('benchmark.tsv',sep='\t',dtype=str).fillna('').head(3)
records = []
for i, row in df.iterrows():
    record = {f:row[f] for f in fields}
    start = time.perf_counter()
    phase = f'record_{i+1}_extraction'
    concepts, _ = ministral_extract(llm, record)
    extraction_seconds = time.perf_counter()-start
    phase = f'record_{i+1}_retrieval'
    retrieval_start = time.perf_counter()
    bundles = retrieve_for_concepts(idx, concepts)
    retrieval_seconds = time.perf_counter()-retrieval_start
    phase = f'record_{i+1}_grounding'
    grounding_start = time.perf_counter()
    annotations, _ = ministral_ground(llm, record, bundles)
    records.append({'record':i+1,'accession':record['run_accession'],'seconds':time.perf_counter()-start,'extraction_seconds':extraction_seconds,'retrieval_seconds':retrieval_seconds,'grounding_seconds':time.perf_counter()-grounding_start,'concepts':len(concepts)})
    print(json.dumps(records[-1]), flush=True)
    out.joinpath('performance.json').write_text(json.dumps({'load_seconds':load_seconds,'device_map':getattr(llm.model, 'hf_device_map', {str(p.device): sum(q.numel() for q in llm.model.parameters() if q.device == p.device) for p in llm.model.parameters()}),'records':records,'generation_calls':calls},indent=2))
stop.set()
thread.join()
out.joinpath('gpu-samples.json').write_text(json.dumps(samples,indent=2))
print('COMPLETE',flush=True)
