"""Offline tests: synthetic BioSamples, CPU-only Arrow, fake LLM responses."""
from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from biosample_sampling import (canonical_from_parquet, canonical_from_xml, detect_format,
    iter_xml_records, reservoir_sample, sample_input, sample_parquet)
from biosample_sample_test import (DiagnosticLLM, StageFailure, Summary, enrich_record,
    inspect_response, numeric_statistics, run)
from genepio_rag import EXTRACT_SYSTEM, GROUND_SYSTEM, extraction_schema

XML = '''<BioSample accession="SAMN00001" id="1" access="public">
  <Description><Title> A cat sample </Title>
    <Organism taxonomy_name="Felis catus" taxonomy_id="9685"/>
    <Comment><Paragraph>urine from a cat</Paragraph><Paragraph>second note</Paragraph></Comment>
  </Description>
  <Owner><Name abbreviation="LAB">Laboratory</Name></Owner>
  <Models><Model>Generic</Model><Model>Animal</Model></Models>
  <Package display_name="Generic sample">Generic.1.0</Package>
  <Status status="live" when="2020-01-01"/>
  <Ids><Id db="BioSample">SAMN00001</Id><Id db="SRA">SRS1</Id></Ids>
  <Attributes>
    <Attribute attribute_name="Host" harmonized_name="host">cat</Attribute>
    <Attribute attribute_name="isolation_source">urine</Attribute>
    <Attribute attribute_name="isolation_source">blood</Attribute>
    <Attribute attribute_name="empty"> </Attribute>
    <Attribute attribute_name="unexpected_field">novel metadata</Attribute>
  </Attributes>
  <Links><Link type="url" label="study">https://example.org/study</Link></Links>
</BioSample>'''

SCHEMA = {'type': 'object', 'properties': {'c': {'type': 'array'}},
          'required': ['c'], 'additionalProperties': False}


class FakeBackend:
    model_id = 'offline-fake-no-model'
    def __init__(self, responses: list) -> None:
        self.responses = iter(responses)
        self.calls: list = []
    def chat_batch(self, system: str, users: list, schemas: list) -> list:
        self.calls.append((system, users, schemas))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        text, reason = value
        return [SimpleNamespace(prompt_token_ids=list(range(10)), outputs=[
            SimpleNamespace(text=text, finish_reason=reason, token_ids=list(range(5)))])]


def fake_retrieval(index: object, concepts: list, k: int) -> list:
    assert k == 5
    hit = {'id': 'NCBITaxon:9685', 'label': 'Felis catus', 'synonyms': ['cat'],
           'score': 1.2, 'definition': 'Domestic cat', 'rank': 1}
    return [{'concept_index': i, 'concept': c, 'candidates': [dict(hit)]} for i, c in enumerate(concepts)]


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_parquet(self, rows: int = 50, row_group_size: int = 7) -> Path:
        path = self.root / 'input.parquet'
        table = pa.table({'accession': [f'SAMN{i:05}' for i in range(rows)],
                          'arbitrary_value': [f'v{i}' for i in range(rows)],
                          'empty': [None] * rows})
        pq.write_table(table, path, row_group_size=row_group_size)
        return path

    def test_reservoir_reproducibility_and_no_global_rng(self) -> None:
        import random
        rows = [{'accession': str(i), 'metadata': {'accession': str(i)}} for i in range(100)]
        state = random.getstate()
        first, info = reservoir_sample(iter(rows), 10)
        second, _ = reservoir_sample(iter(rows), 10)
        third, _ = reservoir_sample(iter(rows), 10, 9)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(len(first), 10)
        self.assertEqual(info['population_records'], 100)
        self.assertEqual(info['eligible_records'], 100)
        self.assertEqual(len(set(info['source_row_indices'])), 10)
        self.assertEqual(random.getstate(), state)

    def test_reservoir_small_empty_and_filter(self) -> None:
        rows = [{'accession': 'a', 'metadata': {}}, {'accession': 'b', 'metadata': {'host': 'cat'}}]
        self.assertEqual(reservoir_sample(rows, 9)[0], rows)
        self.assertEqual(reservoir_sample(rows, 0)[0], [])
        self.assertEqual(reservoir_sample(rows, 9, min_nonempty_fields=1)[0], rows[1:])
        self.assertEqual(reservoir_sample([], 9)[0], [])
        with self.assertRaises(ValueError): reservoir_sample(rows, -1)

    def test_parquet_reproducible_across_batch_and_row_group_sizes(self) -> None:
        path = self.make_parquet()
        first, info = sample_parquet(path, 12, batch_size=3)
        second, _ = sample_parquet(path, 12, batch_size=9)
        third, _ = sample_parquet(path, 12, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        original = pq.read_table(path)
        other = self.root / 'regrouped.parquet'
        pq.write_table(original, other, row_group_size=11)
        self.assertEqual(first, sample_parquet(other, 12, batch_size=4)[0])
        self.assertEqual(len(first), 12)
        self.assertEqual(info['population_records'], 50)
        self.assertEqual(len(set(info['source_row_indices'])), 12)

    def test_parquet_reads_selected_groups_only_and_never_full_table(self) -> None:
        path = self.make_parquet(100, 10)
        with patch('pyarrow.parquet.read_table', side_effect=AssertionError('No full-table read')):
            sampled, info = sample_parquet(path, 1)
        self.assertEqual(len(sampled), 1)
        self.assertEqual(info['row_groups_read'], 1)
        self.assertEqual(sample_parquet(path, 0)[0], [])

    def test_parquet_filter_reservoir_and_small_input(self) -> None:
        path = self.make_parquet(8)
        a, info = sample_parquet(path, 3, min_nonempty_fields=2, batch_size=2)
        b, _ = sample_parquet(path, 3, min_nonempty_fields=2, batch_size=3)
        self.assertEqual(a, b)
        self.assertEqual(info['method'], 'reservoir')
        self.assertEqual(info['eligible_records'], 8)
        self.assertEqual(len(sample_parquet(path, 100)[0]), 8)
        self.assertEqual(sample_parquet(path, 3, min_nonempty_fields=3)[0], [])

    def test_xml_canonical_equals_corresponding_parquet_map_row(self) -> None:
        xml = canonical_from_xml(ET.fromstring(XML))
        row = {'accession': 'SAMN00001', 'biosample_id': '1', 'access': 'public',
               'title': 'A cat sample', 'organism': 'Felis catus', 'tax_id': '9685',
               'description': 'urine from a cat | second note', 'owner': 'Laboratory',
               'owner_abbreviation': 'LAB', 'model': 'Generic | Animal', 'package': 'Generic.1.0',
               'package_display_name': 'Generic sample', 'status': 'live', 'status_date': '2020-01-01',
               'attributes': [('host', 'cat'), ('isolation_source', 'urine | blood'), ('empty', ''),
                              ('unexpected_field', 'novel metadata')],
               'original_attributes': [('Host', 'cat'), ('isolation_source', 'urine | blood'),
                                       ('unexpected_field', 'novel metadata')],
               'identifiers': [('id.BioSample', 'SAMN00001'), ('id.SRA', 'SRS1')],
               'links': [('link.url.study', 'https://example.org/study')], 'absent': None}
        self.assertEqual(xml, canonical_from_parquet(row))
        self.assertNotIn('empty', xml['metadata'])
        self.assertEqual(xml['metadata']['unexpected_field'], 'novel metadata')
        schema = pa.schema([(key, pa.map_(pa.string(), pa.string()) if key in
                            {'attributes', 'original_attributes', 'identifiers', 'links'} else pa.string())
                            for key in row])
        path = self.root / 'map.parquet'
        pq.write_table(pa.Table.from_pylist([row], schema=schema), path)
        self.assertEqual(sample_parquet(path, 1)[0], [xml])

    def test_xml_gzip_plain_namespaces_and_unlisted_metadata(self) -> None:
        data = '<BioSampleSet>' + XML + XML.replace('SAMN00001', 'SAMN00002') + '</BioSampleSet>'
        path = self.root / 'records.xml.gz'
        with gzip.open(path, 'wb') as fh: fh.write(data.encode())
        self.assertEqual(len(list(iter_xml_records(path))), 2)
        self.assertEqual(sample_input(path, 2)[1]['input_format'], 'xml')
        plain = self.root / 'unknown_extension.dat'
        plain.write_text(data)
        self.assertEqual(detect_format(plain), 'xml')
        self.assertEqual(sample_input(path, 1), sample_input(plain, 1))
        namespaced = ET.fromstring('<BioSample xmlns="urn:ncbi" accession="SAMN3"><Novel flag="yes">value</Novel></BioSample>')
        canonical = canonical_from_xml(namespaced)
        self.assertEqual(canonical['metadata']['novel'], 'value')
        self.assertEqual(canonical['metadata']['novel.flag'], 'yes')
        self.assertEqual(detect_format(self.make_parquet()), 'parquet')
        self.assertEqual(detect_format(plain, 'parquet'), 'parquet')

    def test_remove_only_empty_fields_preserve_arbitrary_and_nested(self) -> None:
        canonical = canonical_from_parquet({'accession': 'SAMN1', 'null': None, 'blank': ' \n ',
            'nan': math.nan, 'zero': 0, 'false': False, 'host': 'NA',
            'novel': {'a': 'keep', 'b': None}, 'list': ['first', '', None, 'second']})
        self.assertEqual(canonical['metadata'], {'accession': 'SAMN1', 'zero': '0', 'false': 'False',
            'host': 'NA', 'novel.a': 'keep', 'list': 'first | second'})

    def test_xml_detaches_records_instead_of_retaining_empty_children(self) -> None:
        path = self.root / 'placeholder.xml'
        path.write_text('<BioSampleSet/>')
        root = ET.Element('BioSampleSet')
        def events(*args: object, **kwargs: object):
            yield 'start', root
            for i in range(1000):
                child = ET.SubElement(root, 'BioSample', accession=f'SAMN{i}')
                yield 'start', child
                yield 'end', child
            yield 'end', root
        with patch('biosample_sampling.ET.iterparse', side_effect=events):
            for record in iter_xml_records(path):
                self.assertEqual(len(root), 0)
                self.assertIsNotNone(record['accession'])

    def test_attribute_collision_does_not_change_structural_accession(self) -> None:
        row = {'accession': 'SAMN1', 'attributes': {'accession': 'submitted-value'}}
        record = canonical_from_parquet(row)
        self.assertEqual(record['accession'], 'SAMN1')
        self.assertIn('submitted-value', record['metadata']['accession'])


class DiagnosticTests(unittest.TestCase):
    def test_truncation_even_when_json_and_schema_are_valid(self) -> None:
        d = inspect_response('{"c":[]}', SCHEMA, 'length')
        self.assertTrue(d['truncated'])
        self.assertTrue(d['json_valid'])
        self.assertTrue(d['schema_valid'])
        self.assertFalse(d['accepted'])

    def test_malformed_json_schema_invalid_and_aborted_responses(self) -> None:
        for text in ['{"c":[', '```json\n{"c":[]}\n```', '{"c":NaN}']:
            d = inspect_response(text, SCHEMA, 'stop')
            self.assertFalse(d['json_valid'])
            self.assertIsNone(d['schema_valid'])
            self.assertFalse(d['accepted'])
        d = inspect_response('{"wrong":[]}', SCHEMA, 'stop')
        self.assertTrue(d['json_valid'])
        self.assertFalse(d['schema_valid'])
        self.assertTrue(d['schema_errors'])
        self.assertFalse(inspect_response('{"c":[]}', SCHEMA, 'abort')['accepted'])

    def test_model_error_and_metrics_with_no_gpu(self) -> None:
        observed = DiagnosticLLM(FakeBackend([RuntimeError('out of context')]))
        with self.assertRaises(StageFailure): observed.chat('sys', 'user', schema=SCHEMA)
        self.assertIn('out of context', observed.last['model_error'])
        self.assertIsNone(observed.last['generated_tokens'])
        observed = DiagnosticLLM(FakeBackend([('{"c":[]}', 'stop')]))
        self.assertEqual(observed.chat('sys', 'user', SCHEMA), '{"c":[]}')
        self.assertEqual(observed.last['prompt_tokens'], 10)
        self.assertEqual(observed.last['generated_tokens'], 5)

    def test_full_two_pass_reconstruction_and_unchanged_prompts(self) -> None:
        backend = FakeBackend([('{"c":[["host","organism","domestic cat"]]}', 'stop'),
                               ('{"s":[0]}', 'stop')])
        record = {'accession': 'SAMN1', 'metadata': {'accession': 'SAMN1', 'host': 'cat', 'novel': 'x'}}
        result = enrich_record(record, None, backend, fake_retrieval)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['metadata'], record['metadata'])
        self.assertEqual(result['concepts'][0]['annotation']['ontology_id'], 'NCBITaxon:9685')
        self.assertEqual(result['concepts'][0]['raw_value'], 'cat')
        self.assertEqual(backend.calls[0][0], EXTRACT_SYSTEM)
        self.assertEqual(backend.calls[1][0], GROUND_SYSTEM)
        self.assertIn('novel=x', backend.calls[0][1][0])
        self.assertTrue(all(result['diagnostics'][name]['accepted'] for name in ['extraction', 'grounding']))

    def test_extraction_failure_skips_grounding_and_retrieval(self) -> None:
        record = {'accession': 'SAMN1', 'metadata': {'host': 'cat'}}
        for raw, finish in [('{"c":[]}', 'length'), ('{', 'stop'), ('{"c":[["unknown","organism","cat"]]}', 'stop')]:
            backend = FakeBackend([(raw, finish)])
            with patch('genepio_rag.retrieve_for_concepts', side_effect=AssertionError('Must not retrieve')):
                result = enrich_record(record, None, backend)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failed_stage'], 'concept_extraction')
            self.assertFalse(result['diagnostics']['grounding']['attempted'])
            self.assertEqual(result['diagnostics']['extraction']['raw_response'], raw)
            self.assertEqual(len(backend.calls), 1)

    def test_grounding_failure_retains_extraction_and_candidates(self) -> None:
        backend = FakeBackend([('{"c":[["host","organism","cat"]]}', 'stop'), ('{"s":[8]}', 'stop')])
        result = enrich_record({'accession': 'SAMN1', 'metadata': {'host': 'cat'}}, None, backend, fake_retrieval)
        self.assertEqual(result['failed_stage'], 'grounding')
        self.assertEqual(len(result['concepts']), 1)
        self.assertIn('candidates', result['concepts'][0])
        self.assertNotIn('annotation', result['concepts'][0])
        self.assertFalse(result['diagnostics']['grounding']['schema_valid'])

    def test_zero_concepts_and_empty_metadata_do_not_invent_errors(self) -> None:
        backend = FakeBackend([('{"c":[]}', 'stop')])
        result = enrich_record({'accession': 'SAMN1', 'metadata': {'host': 'cat'}}, None, backend, fake_retrieval)
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['diagnostics']['grounding']['attempted'])
        result = enrich_record({'accession': None, 'metadata': {}}, None, FakeBackend([]), fake_retrieval)
        self.assertEqual(result['status'], 'failed')

    def test_summary_statistics_rates_tokens_and_timings(self) -> None:
        record = {'accession': 'SAMN1', 'metadata': {'host': 'cat'}}
        good = enrich_record(record, None, FakeBackend([
            ('{"c":[["host","organism","cat"]]}', 'stop'), ('{"s":[0]}', 'stop')]), fake_retrieval)
        bad = enrich_record(record, None, FakeBackend([('{', 'length')]), fake_retrieval)
        good['diagnostics']['extraction']['wall_seconds'] = 2
        good['diagnostics']['grounding']['wall_seconds'] = 3
        bad['diagnostics']['extraction']['wall_seconds'] = 1
        summary = Summary()
        summary.add(good); summary.add(bad)
        result = summary.as_dict(10)
        self.assertEqual(result['processed_records'], 2)
        self.assertEqual(result['completed_records'], 1)
        self.assertEqual(result['total_concepts'], 1)
        self.assertEqual(result['matched_annotations'], 1)
        self.assertEqual(result['concepts_per_record'], [1, 0])
        self.assertEqual(result['records_per_second'], .2)
        self.assertEqual(result['stages']['extraction']['truncation_rate'], .5)
        self.assertEqual(result['stages']['extraction']['json_failure_rate'], .5)
        self.assertEqual(result['stages']['grounding']['skipped_records'], 1)
        self.assertEqual(result['stages']['extraction']['generated_tokens']['total'], 10)
        self.assertEqual(result['stages']['extraction']['wall_seconds'], 3)
        self.assertEqual(numeric_statistics([0, 10])['median'], 5)
        self.assertIsNone(Summary().as_dict(0)['stages']['grounding']['truncation_rate'])


class WorkflowTests(unittest.TestCase):
    def test_sample_only_artifacts_and_no_model_or_index_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'samples.xml'
            source.write_text('<BioSampleSet>' + XML + '</BioSampleSet>')
            with patch('genepio_rag.MinistralLocal', side_effect=AssertionError('No GPU')), \
                 patch('joblib.load', side_effect=AssertionError('No index')):
                summary = run(source, root / 'out', n=1, sample_only=True)
            self.assertEqual(summary['run_status'], 'sample_only')
            self.assertEqual(summary['processed_records'], 0)
            self.assertEqual(summary['sampled_records'], 1)
            self.assertEqual(summary['sampling']['seed'], 12345)
            self.assertEqual(summary['sampling']['min_nonempty_fields'], 0)
            self.assertEqual((root / 'out' / 'sampled_accessions.txt').read_text(), 'SAMN00001\n')
            self.assertEqual((root / 'out' / 'enrichment_results.jsonl').read_text(), '')
            self.assertEqual(json.loads((root / 'out' / 'sampled_metadata.jsonl').read_text()), canonical_from_xml(ET.fromstring(XML)))
            with self.assertRaises(FileExistsError): run(source, root / 'out', sample_only=True)

    def test_fake_run_records_model_errors_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'samples.xml'
            source.write_text('<BioSampleSet>' + XML + XML.replace('SAMN00001','SAMN00002') + '</BioSampleSet>')
            backend = FakeBackend([RuntimeError('model problem'),
                                   ('{"c":[["host","organism","cat"]]}', 'stop'), ('{"s":[null]}', 'stop')])
            summary = run(source, root / 'out', n=2, backend_factory=lambda: backend,
                          index_loader=lambda: None, retriever=fake_retrieval)
            rows = [json.loads(line) for line in (root / 'out' / 'enrichment_results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary['processed_records'], 2)
            self.assertEqual(summary['completed_records'], 1)
            self.assertEqual(summary['unresolved_annotations'], 1)
            self.assertEqual(summary['run_status'], 'completed_with_errors')
            self.assertIn('model problem', rows[0]['diagnostics']['extraction']['model_error'])
            self.assertEqual(json.loads((root / 'out' / 'summary.json').read_text()), summary)


if __name__ == '__main__': unittest.main()
