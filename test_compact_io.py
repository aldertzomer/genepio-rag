"""Offline checks for reconstruction and the index-only grounding boundary."""
import json
import unittest
from jsonschema import validate, ValidationError
from genepio_rag import (extraction_schema, grounding_schema, reconstruct_concepts,
                         reconstruct_annotations, grounding_prompt)

class CompactIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {'isolation_source': 'cat', 'tax_id': '197'}
        self.concept = {'source_field': 'isolation_source', 'concept_type': 'organism',
                        'normalized_query': 'domestic cat'}
        self.hit = {'id': 'NCBITaxon:9685', 'label': 'Felis catus', 'score': 1.5,
                    'synonyms': ['cat'], 'definition': 'A domestic cat.'}
        self.bundles = [{'concept_index': 0, 'concept': self.concept, 'candidates': [self.hit]}]

    def test_reconstructed_provenance_is_exact_input(self) -> None:
        raw = '{"c":[["isolation_source","organism","domestic cat"]]}'
        validate(json.loads(raw), extraction_schema(self.record))
        concept = reconstruct_concepts(raw, self.record)[0]
        self.assertEqual(concept['raw_value'], 'cat')
        self.assertEqual(concept['evidence'], 'isolation_source=cat')
        with self.assertRaises(ValueError):
            reconstruct_concepts('{"c":[["invented_field","organism","cat"]]}', self.record)

    def test_canonical_id_label_score_come_from_retrieval(self) -> None:
        raw = '{"s":[0]}'
        validate(json.loads(raw), grounding_schema(self.bundles))
        ann = reconstruct_annotations(raw, self.bundles)[0]
        self.assertEqual((ann['ontology_id'], ann['label'], ann['retrieval_score']),
                         ('NCBITaxon:9685', 'Felis catus', 1.5))
        self.assertNotIn('NCBITaxon:9685', grounding_prompt(self.record, self.bundles))

    def test_invalid_indices_cannot_create_matched_ids(self) -> None:
        for index in [-1, 1, 4, True, 0.0, 'NCBITaxon:9685']:
            raw = json.dumps({'s': [index]})
            ann = reconstruct_annotations(raw, self.bundles)[0]
            self.assertEqual(ann['status'], 'unresolved')
            self.assertIsNone(ann['ontology_id'])
        with self.assertRaises(ValidationError):
            validate({'s': ['NCBITaxon:9685']}, grounding_schema(self.bundles))

    def test_selection_count_and_empty_candidates(self) -> None:
        with self.assertRaises(ValueError): reconstruct_annotations('{"s":[]}', self.bundles)
        self.bundles[0]['candidates'] = []
        validate({'s': [None]}, grounding_schema(self.bundles))
        self.assertEqual(reconstruct_annotations('{"s":[null]}', self.bundles)[0]['status'], 'unresolved')
        validate({'s': []}, grounding_schema([]))

if __name__ == '__main__': unittest.main()
