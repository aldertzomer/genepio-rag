"""CPU-only BioSample canonicalization and reproducible bounded-memory sampling."""
from __future__ import annotations

import gzip
import math
import random
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

CanonicalRecord = dict[str, Any]


def nonempty_text(value: Any) -> str | None:
    """Drop actual nulls/blank values; keep literal missingness descriptions and zero."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (date, datetime)):
        value = value.isoformat()
    text = str(value).strip()
    return text if text else None


def add_value(metadata: dict[str, str], key: str, value: Any) -> None:
    text = nonempty_text(value)
    if not key or text is None:
        return
    if key not in metadata:
        metadata[key] = text
    elif text not in metadata[key].split(' | '):
        metadata[key] += ' | ' + text


def flatten_value(metadata: dict[str, str], key: str, value: Any) -> None:
    """Arrow maps arrive as pairs; structs as dicts. Preserve every leaf."""
    if isinstance(value, Mapping):
        for subkey, subvalue in value.items():
            flatten_value(metadata, f'{key}.{subkey}' if key else str(subkey), subvalue)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) for v in value):
            for subkey, subvalue in value:
                flatten_value(metadata, f'{key}.{subkey}' if key else subkey, subvalue)
        else:
            for item in value:
                flatten_value(metadata, key, item)
    else:
        add_value(metadata, key, value)


def canonical_record(metadata: dict[str, str], accession: str | None = None) -> CanonicalRecord:
    return {'accession': accession or metadata.get('accession') or metadata.get('id.BioSample'),
            'metadata': dict(sorted(metadata.items()))}


def canonical_from_parquet(row: Mapping[str, Any]) -> CanonicalRecord:
    metadata: dict[str, str] = {}
    for key, value in row.items():
        # These are container names in the supplied XML-to-Parquet converter,
        # not a whitelist of metadata fields. All other columns are retained.
        prefix = '' if key in {'attributes', 'identifiers', 'links'} else key
        flatten_value(metadata, prefix, value)
    return canonical_record(metadata, nonempty_text(row.get('accession')))


def local_tag(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def canonical_from_xml(element: ET.Element) -> CanonicalRecord:
    metadata: dict[str, str] = {}
    for key, value in element.attrib.items():
        add_value(metadata, 'biosample_id' if key == 'id' else key, value)
    # Structural aliases match the supplied converter's Parquet columns.
    paths = {'Description/Title': 'title', 'Description/Comment/Paragraph': 'description',
             'Owner/Name': 'owner', 'Models/Model': 'model', 'Package': 'package'}
    aliases = {('Description/Organism', 'taxonomy_name'): 'organism',
               ('Description/Organism', 'taxonomy_id'): 'tax_id',
               ('Owner/Name', 'abbreviation'): 'owner_abbreviation',
               ('Package', 'display_name'): 'package_display_name',
               ('Status', 'status'): 'status', ('Status', 'when'): 'status_date'}

    def visit(node: ET.Element, path: str) -> None:
        tag = local_tag(node.tag)
        if tag == 'Attribute' and path == 'Attributes/Attribute':
            name = node.get('harmonized_name') or node.get('attribute_name')
            text = ''.join(node.itertext())
            if name:
                add_value(metadata, name, text)
            if node.get('attribute_name'):
                add_value(metadata, 'original_attributes.' + node.get('attribute_name', ''), text)
            for key, value in node.attrib.items():
                if key not in {'attribute_name', 'harmonized_name'}:
                    add_value(metadata, f'attribute_details.{name or "unnamed"}.{key}', value)
            if not name:
                add_value(metadata, 'attributes.unnamed', text)
            return
        if path == 'Ids/Id':
            key = 'id.' + node.get('db', 'unknown')
            if node.get('db_label'):
                key += '.' + node.get('db_label', '')
            add_value(metadata, key, ''.join(node.itertext()))
            for attr, value in node.attrib.items():
                if attr not in {'db', 'db_label'}:
                    add_value(metadata, key + '.' + attr, value)
            return
        if path == 'Links/Link':
            parts = [node.get('type', 'unknown')]
            parts += [node.get(k, '') for k in ('target', 'label') if node.get(k)]
            key = 'link.' + '.'.join(parts)
            add_value(metadata, key, ''.join(node.itertext()))
            for attr, value in node.attrib.items():
                if attr not in {'type', 'target', 'label'}:
                    add_value(metadata, key + '.' + attr, value)
            return
        key = paths.get(path, path.replace('/', '.').lower())
        for attr, value in node.attrib.items():
            add_value(metadata, aliases.get((path, attr), key + '.' + attr), value)
        if path == 'Description/Comment/Paragraph':
            add_value(metadata, key, ''.join(node.itertext()))
            # Preserve attributes of nested formatting elements too.
            for descendant in node.iter():
                if descendant is not node:
                    for attr, value in descendant.attrib.items():
                        add_value(metadata, key + '.' + local_tag(descendant.tag).lower() + '.' + attr, value)
            return
        add_value(metadata, key, node.text)
        for child in node:
            visit(child, path + '/' + local_tag(child.tag))
            add_value(metadata, key + '.tail', child.tail)

    for child in element:
        visit(child, local_tag(child.tag))
    return canonical_record(metadata, nonempty_text(element.get('accession')))


def detect_format(path: Path, override: str = 'auto') -> str:
    if override != 'auto':
        if override not in {'xml', 'parquet'}:
            raise ValueError('Format must be auto, xml or parquet')
        return override
    with path.open('rb') as fh:
        magic = fh.read(4)
    if magic == b'PAR1':
        return 'parquet'
    opener = gzip.open if magic[:2] == b'\x1f\x8b' else open
    with opener(path, 'rb') as fh:
        prefix = fh.read(1024).lstrip(b'\xef\xbb\xbf \t\r\n')
    if prefix.startswith(b'<'):
        return 'xml'
    raise ValueError(f'Cannot detect XML or Parquet format for {path}; use --format')


def iter_xml_records(path: Path) -> Iterator[CanonicalRecord]:
    with path.open('rb') as fh:
        compressed = fh.read(2) == b'\x1f\x8b'
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as fh:
        stack: list[ET.Element] = []
        for event, elem in ET.iterparse(fh, events=('start', 'end')):
            if event == 'start':
                stack.append(elem)
                continue
            if local_tag(elem.tag) == 'BioSample':
                record = canonical_from_xml(elem)
                # clear alone leaves millions of empty children attached to root.
                if len(stack) > 1:
                    stack[-2].remove(elem)
                elem.clear()
                yield record
            elif not any(local_tag(parent.tag) == 'BioSample' for parent in stack[:-1]):
                if len(stack) > 1:
                    stack[-2].remove(elem)
                elem.clear()
            stack.pop()


def reservoir_sample(records: Iterable[CanonicalRecord], n: int, seed: int = 12345,
                     min_nonempty_fields: int = 0) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    if n < 0 or min_nonempty_fields < 0:
        raise ValueError('n and min_nonempty_fields must be non-negative')
    rng = random.Random(seed)
    sample: list[tuple[int, CanonicalRecord]] = []
    seen = eligible = 0
    for record in records:
        seen += 1
        if len(record['metadata']) < min_nonempty_fields:
            continue
        eligible += 1
        if len(sample) < n:
            sample.append((seen - 1, record))
        elif n:
            slot = rng.randrange(eligible)
            if slot < n:
                sample[slot] = (seen - 1, record)
    sample.sort(key=lambda item: item[0])
    return [record for _, record in sample], {'method': 'reservoir', 'population_records': seen,
            'eligible_records': eligible, 'source_row_indices': [i for i, _ in sample]}


def sample_parquet(path: Path, n: int, seed: int = 12345,
                   min_nonempty_fields: int = 0, batch_size: int = 4096
                   ) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    import pyarrow.parquet as pq

    if n < 0 or min_nonempty_fields < 0 or batch_size <= 0:
        raise ValueError('Invalid sampling counts or batch size')
    with pq.ParquetFile(path) as parquet:
        if min_nonempty_fields:
            rows = (canonical_from_parquet(row) for batch in parquet.iter_batches(batch_size=batch_size)
                    for row in batch.to_pylist())
            return reservoir_sample(rows, n, seed, min_nonempty_fields)
        total = parquet.metadata.num_rows
        selected = sorted(random.Random(seed).sample(range(total), min(n, total)))
        ends: list[int] = []
        offset = 0
        for group in range(parquet.num_row_groups):
            offset += parquet.metadata.row_group(group).num_rows
            ends.append(offset)
        targets: dict[int, list[int]] = {}
        for index in selected:
            group = bisect_right(ends, index)
            start = ends[group - 1] if group else 0
            targets.setdefault(group, []).append(index - start)
        sampled: list[CanonicalRecord] = []
        for group, local_rows in targets.items():
            start = cursor = 0
            for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group]):
                stop = start + batch.num_rows
                within: list[int] = []
                while cursor < len(local_rows) and local_rows[cursor] < stop:
                    within.append(local_rows[cursor] - start)
                    cursor += 1
                if within:
                    sampled.extend(canonical_from_parquet(row) for row in batch.take(within).to_pylist())
                start = stop
                if cursor == len(local_rows):
                    break
        return sampled, {'method': 'parquet_random_rows', 'population_records': total,
                         'eligible_records': total, 'source_row_indices': selected,
                         'row_groups_read': len(targets)}


def sample_input(path: Path, n: int = 200, seed: int = 12345, input_format: str = 'auto',
                 min_nonempty_fields: int = 0) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    input_format = detect_format(path, input_format)
    if input_format == 'xml':
        sample, details = reservoir_sample(iter_xml_records(path), n, seed, min_nonempty_fields)
    else:
        sample, details = sample_parquet(path, n, seed, min_nonempty_fields)
    details.update(input_format=input_format, requested_records=n, seed=seed,
                   min_nonempty_fields=min_nonempty_fields)
    return sample, details
