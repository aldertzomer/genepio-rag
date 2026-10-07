"""Audit existing responses and compare grounding payloads without production edits."""
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from benchmark_vllm import PromptCapture, Replay, measure
from genepio_rag import (MinistralLocal, GROUND_SYSTEM, extract_json,
                         record_to_text, retrieve_for_concepts, ministral_ground)

OUT = Path('baseline-results/grounding-comparison')

def clean(text: str) -> str:
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.I)
    return re.sub(r'\s*```$', '', text).strip()

def audit() -> None:
    responses = json.loads(Path('baseline-results/vllm-benchmark/responses.json').read_text())
    rows = []
    occurrence: Counter = Counter()
    for response in responses:
        key = (response['mode'], response['pass'])
        occurrence[key] += 1
        text = clean(response['text'])
        field = 'concepts' if response['pass']=='extraction' else 'annotations'
        valid = False
        parsed_count = None
        try:
            obj = json.loads(text)
            valid = True
            parsed_count = len(obj[field]) if isinstance(obj.get(field),list) else None
        except (ValueError, TypeError, KeyError):
            pass
        # Count complete objects in the unfinished array for diagnostic use only.
        match = re.search(r'"'+field+r'"\s*:\s*\[', text)
        count = 0
        if match:
            pos = match.end()
            decoder = json.JSONDecoder()
            while pos < len(text):
                while pos < len(text) and text[pos] in ' \r\n\t,': pos += 1
                if pos >= len(text) or text[pos]!='{': break
                try:
                    item, end = decoder.raw_decode(text,pos)
                except ValueError:
                    break
                if isinstance(item,dict): count += 1
                pos = end
        rows.append({'mode':response['mode'],'pass':response['pass'],'request_in_pass':occurrence[key],
                     'record_index':response['record_index'],'finish_reason':response['finish_reason'],
                     'json_valid':valid,'truncated':response['finish_reason']=='length',
                     'extracted_concepts':parsed_count if field=='concepts' else None,
                     'complete_partial_concept_objects':count if field=='concepts' else None,
                     'complete_partial_annotation_objects':count if field=='annotations' else None,
                     'expected_closing_delimiters_present':bool(re.search(r'\]\s*\}\s*$',text)),
                     'response_tail':text[-100:]})
    OUT.joinpath('response-audit.json').write_text(json.dumps(rows,indent=2))
    lines=['| Mode | Pass | Request | Record | Finish | Valid JSON | Truncated | Parsed concepts | Complete partial objects | Closing `]}` |','|---|---|---:|---:|---|---|---|---|---:|---|']
    for r in rows:
        partial = r['complete_partial_concept_objects'] if r['pass']=='extraction' else r['complete_partial_annotation_objects']
        lines.append(f"| {r['mode']} | {r['pass']} | {r['request_in_pass']} | {r['record_index']+1} | {r['finish_reason']} | {r['json_valid']} | {r['truncated']} | {r['extracted_concepts']} | {partial} | {r['expected_closing_delimiters_present']} |")
    OUT.joinpath('response-audit.md').write_text('\n'.join(lines)+'\n')
    summary = {'responses':len(rows),'by_pass':{p:{'responses':len(a),'finish_reasons':dict(Counter(r['finish_reason'] for r in a)), 'valid_json':sum(r['json_valid'] for r in a),'truncation_rate':sum(r['truncated'] for r in a)/len(a),'closing_delimiters':sum(r['expected_closing_delimiters_present'] for r in a),'complete_partial_objects':dict(Counter((r['complete_partial_concept_objects'] if p=='extraction' else r['complete_partial_annotation_objects']) for r in a))} for p in ['extraction','grounding'] if (a := [r for r in rows if r['pass']==p])}}
    OUT.joinpath('audit-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

def compact_prompt(record: dict, bundles: list) -> str:
    payload = []
    for bundle in bundles:
        candidates = []
        definition_added = False
        for hit in bundle['candidates']:
            candidate = {k:hit[k] for k in ['id','label','score']}
            candidate['synonyms'] = hit['synonyms'][:8]
            # At most one definition per concept: first ranked non-empty one.
            definition = ' '.join(hit['definition'].split())
            if definition and not definition_added:
                candidate['definition'] = definition[:160]
                definition_added = True
            candidates.append(candidate)
        payload.append({'concept_index':bundle['concept_index'], 'concept':bundle['concept'], 'candidates':candidates})
    return ('Original metadata:\n'+record_to_text(record)+'\n\nExtracted concepts and retrieved candidates:\n'+json.dumps(payload,separators=(',',':'),ensure_ascii=False))

def signatures(annotations: list, n: int) -> list:
    by_index = {a['concept_index']:a for a in annotations}
    return [(by_index[i]['status'],by_index[i]['ontology_id']) if i in by_index else ('missing',None) for i in range(n)]

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    audit()
    baseline = [json.loads(line) for line in Path('annotations.jsonl').read_text().splitlines()][:3]
    idx = joblib.load('genepio_rag.joblib')
    records = [r['metadata'] for r in baseline]
    variants = {}
    for name, k in [('current_top8',8),('compact_top3',3),('compact_top5',5)]:
        bundles = [retrieve_for_concepts(idx,extract_json(r['debug']['extraction_raw'])['concepts'],k=k) for r in baseline]
        users = []
        for record, b in zip(records,bundles):
            if name=='current_top8':
                capture = PromptCapture()
                ministral_ground(capture,record,b)
                users.append(capture.user)
            else: users.append(compact_prompt(record,b))
        variants[name] = (bundles,users)
    indices = [i%3 for i in range(32)]
    llm = MinistralLocal(temperature=0.0)
    measure(llm,GROUND_SYSTEM,[variants['current_top8'][1][0]])
    results=[]
    responses=[]
    validated={}
    # Keep exact current cap, then an explicit benchmark-only control that can
    # produce complete JSON and enable annotation comparison.
    for cap in [96,768]:
        llm.max_new_tokens=cap
        for name,(bundles,users) in variants.items():
            stats, outputs = measure(llm,GROUND_SYSTEM,[users[i] for i in indices])
            stats.update({'variant':name,'max_tokens':cap,'prompt_tokens_per_record':[len(o.prompt_token_ids) for o in outputs], 'prompt_tokens_per_record_mean':stats['prompt_tokens']/32,'valid_json_records':0,'complete_annotation_records':0,'validator_rejected_ids':0})
            annotations=[]
            for occurrence,(index,output) in enumerate(zip(indices,outputs)):
                text=output.outputs[0].text
                error=None
                ann=None
                try:
                    obj=extract_json(text)
                    raw_ann=obj['annotations']
                    if not isinstance(raw_ann,list): raise ValueError('annotations is not a list')
                    stats['valid_json_records']+=1
                    ann,_=ministral_ground(Replay(text),records[index],bundles[index])
                    for a in ann:
                        if a['status']=='matched':
                            allowed={h['id'] for h in bundles[index][a['concept_index']]['candidates']}
                            assert a['ontology_id'] in allowed
                    stats['validator_rejected_ids']+=sum('rejected by validator' in a.get('reason','') for a in ann)
                    if {a['concept_index'] for a in ann}==set(range(len(bundles[index]))): stats['complete_annotation_records']+=1
                except (ValueError,KeyError,TypeError) as exc:
                    error=str(exc)
                annotations.append(ann)
                responses.append({'variant':name,'cap':cap,'request':occurrence+1,'record_index':index,'finish_reason':output.outputs[0].finish_reason,'generated_tokens':len(output.outputs[0].token_ids),'text':text,'annotations':ann,'error':error})
            validated[(cap,name)]=annotations
            agreement={}
            for reference,ref_ann in [('current_vllm_top8',validated.get((cap,'current_top8'))),('original_transformers_pipeline',[[c['annotation'] for c in baseline[i]['concepts']] for i in indices])]:
                comparable=equal=concept_equal=concept_total=0
                if ref_ann:
                    for occurrence,i in enumerate(indices):
                        a=annotations[occurrence]; b=ref_ann[occurrence]
                        if a is None or b is None: continue
                        sa=signatures(a,len(bundles[i])); sb=signatures(b,len(bundles[i]))
                        if any(x[0]=='missing' for x in sa+sb): continue
                        comparable+=1;equal+=sa==sb
                        concept_total+=len(sa);concept_equal+=sum(x==y for x,y in zip(sa,sb))
                agreement[reference]={'comparable_records':comparable,'exact_record_agreement':equal/comparable if comparable else None,'concept_agreement':concept_equal/concept_total if concept_total else None,'equal_concepts':concept_equal,'compared_concepts':concept_total}
            stats['agreement']=agreement
            results.append(stats)
            OUT.joinpath('comparison-metrics.json').write_text(json.dumps({'batch_size':32,'definition_policy':'At most one definition per concept, first ranked nonempty definition; whitespace normalized and capped at 160 characters. Other candidates omit definitions. Synonyms capped at eight, matching current prompt. Compact JSON separators. All concept fields, metadata, system prompt and output schema unchanged.','supplemental_control':'768 output tokens is benchmark-only, to enable valid annotation agreement; production cap remains 96.','results':results},indent=2))
            OUT.joinpath('comparison-responses.json').write_text(json.dumps(responses,indent=2))
            print(json.dumps({k:v for k,v in stats.items() if k not in ['samples','prompt_tokens_per_record']}),flush=True)

if __name__=='__main__':
    main()
