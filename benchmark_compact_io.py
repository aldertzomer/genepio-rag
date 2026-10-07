"""Benchmark schema-constrained compact representations, with a strict parity gate."""
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from benchmark_vllm import measure
from genepio_rag import (MinistralLocal, EXTRACT_SYSTEM, GROUND_SYSTEM, extraction_schema,
    grounding_schema, grounding_prompt, reconstruct_concepts, reconstruct_annotations,
    retrieve_for_concepts, record_to_text, extract_json)

OUT=Path('baseline-results/compact-io')

class SchemaBatch:
    def __init__(self, llm: MinistralLocal, schemas: list[dict]) -> None:
        self.llm=llm
        self.schemas=schemas
    def chat_batch(self, system: str, users: list[str]) -> list:
        return self.llm.chat_batch(system,users,schemas=self.schemas)

def distribution(values: list[int]) -> dict:
    return dict(zip(['min','median','P90','P95','P99','max'],
                    [float(x) for x in np.percentile(values,[0,50,90,95,99,100])]))

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=[json.loads(l) for l in Path('annotations.jsonl').read_text().splitlines()][:3]
    reference_responses=json.loads(Path('baseline-results/grounding-comparison/comparison-responses.json').read_text())
    reference={i:next(r['annotations'] for r in reference_responses if r['variant']=='current_top8' and r['cap']==768 and r['record_index']==i) for i in range(3)}
    records=[r['metadata'] for r in baseline]
    indices=[i%3 for i in range(32)]
    idx=joblib.load('genepio_rag.joblib')
    llm=MinistralLocal(max_new_tokens=768,temperature=0.0)
    llm.chat_batch(EXTRACT_SYSTEM,['Metadata record:\n'+record_to_text(records[0])],schemas=[extraction_schema(records[0])])
    results=[]; responses=[]
    def run(name: str, system: str, users: list[str], schemas: list[dict]) -> list:
        stats,outputs=measure(SchemaBatch(llm,schemas),system,users)
        counts=[len(o.outputs[0].token_ids) for o in outputs]
        stats.update(name=name,generated_token_distribution=distribution(counts),generated_tokens_per_record=counts,
                     prompt_tokens_per_record=[len(o.prompt_token_ids) for o in outputs],finish_reasons=dict(Counter(o.outputs[0].finish_reason for o in outputs)))
        assert all(o.outputs[0].finish_reason!='length' for o in outputs), 'Truncation fails acceptance'
        results.append(stats)
        for i,o in zip(indices,outputs):
            responses.append({'pass':name,'record_index':i,'text':o.outputs[0].text,'generated_tokens':len(o.outputs[0].token_ids),'finish_reason':o.outputs[0].finish_reason})
        OUT.joinpath('metrics.json').write_text(json.dumps(results,indent=2))
        OUT.joinpath('responses.json').write_text(json.dumps(responses,indent=2))
        print(json.dumps({k:v for k,v in stats.items() if k!='samples'}),flush=True)
        return outputs
    extraction=run('extraction',EXTRACT_SYSTEM,['Metadata record:\n'+record_to_text(records[i]) for i in indices],[extraction_schema(records[i]) for i in indices])
    concepts=[reconstruct_concepts(o.outputs[0].text,records[i]) for i,o in zip(indices,extraction)]
    # Isolate representation agreement using identical baseline concepts.
    baseline_bundles=[retrieve_for_concepts(idx,extract_json(r['debug']['extraction_raw'])['concepts'],k=5) for r in baseline]
    b=[baseline_bundles[i] for i in indices]
    outputs=run('grounding_fixed_concepts',GROUND_SYSTEM,[grounding_prompt(records[i],bb) for i,bb in zip(indices,b)],[grounding_schema(bb) for bb in b])
    ann=[reconstruct_annotations(o.outputs[0].text,bb) for o,bb in zip(outputs,b)]
    fixed_parity=[[(a['status'],a['ontology_id']) for a in aa]==[(a['status'],a['ontology_id']) for a in reference[i]] for i,aa in zip(indices,ann)]
    # Full two-pass test; compare multisets since extraction may order concepts differently.
    cached={}
    actual=[]
    for c in concepts:
        key=json.dumps(c,sort_keys=True)
        if key not in cached: cached[key]=retrieve_for_concepts(idx,c,k=5)
        actual.append(cached[key])
    outputs=run('grounding_end_to_end',GROUND_SYSTEM,[grounding_prompt(records[i],bb) for i,bb in zip(indices,actual)],[grounding_schema(bb) for bb in actual])
    annotations=[reconstruct_annotations(o.outputs[0].text,bb) for o,bb in zip(outputs,actual)]
    parity=[]; records_out=[]
    for occurrence,(i,c,bb,aa) in enumerate(zip(indices,concepts,actual,annotations)):
        expected=Counter((a['status'],a['ontology_id']) for a in reference[i])
        got=Counter((a['status'],a['ontology_id']) for a in aa)
        parity.append(expected==got)
        enriched=[]
        for cc,bundle,a in zip(c,bb,aa): enriched.append({**cc,'annotation':a,'candidates':bundle['candidates']})
        records_out.append({'metadata':records[i],'concepts':enriched,'model':llm.model_id,
                            'debug':{'extraction_raw':extraction[occurrence].outputs[0].text,'grounding_raw':outputs[occurrence].outputs[0].text}})
    OUT.joinpath('annotations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records_out))
    summary={'fixed_concept_id_status_agreement':sum(fixed_parity)/32,'end_to_end_id_status_agreement':sum(parity)/32,
             'acceptance_passed':all(fixed_parity) and all(parity),'fixed_parity':fixed_parity,'end_to_end_parity':parity,
             'extracted_concepts_per_record':[len(c) for c in concepts],
             'reference_concepts_per_record':[len(reference[i]) for i in indices]}
    OUT.joinpath('acceptance.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if not summary['acceptance_passed']:
        raise SystemExit("Acceptance failed: compact end-to-end IDs/statuses differ from the reference")

if __name__=='__main__': main()
