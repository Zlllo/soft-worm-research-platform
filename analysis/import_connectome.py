"""Reproduce the compact Cook2019Herm export from the pinned upstream cache.

Use --cache PATH for an offline original cache, or --download with network access.
Only data is read; no downloaded Python is imported or executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


COMMIT = 'ffe570268b7cffb9329cd009a1bb37d60db2d465'
BLOB = '2c44e7abaa1efa71c4d814a945692f5775692abc'
SOURCE_PATH = 'cect/cache/Cook2019HermReader.json'
SOURCE_URL = f'https://raw.githubusercontent.com/openworm/ConnectomeToolbox/{COMMIT}/{SOURCE_PATH}'


def compact_export(raw):
    digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if digest != BLOB:
        raise ValueError(f'upstream Git blob hash mismatch: {digest}')
    source = json.loads(raw)
    edges = []
    for kind, matrix in source['connections'].items():
        label = {'Generic_CS': 'chemical', 'Generic_GJ': 'gap_junction'}[kind]
        for i, row in enumerate(matrix):
            for j, weight in enumerate(row):
                if weight:
                    edges.append([i, j, label, int(weight) if weight == int(weight) else weight])
    return dict(schema_version=1, nodes=source['nodes'], edges=edges,
        matrix_orientation='row=presynaptic/source, column=postsynaptic/target',
        weight_meaning='total EM serial sections of connectivity; not functional synaptic efficacy',
        provenance=dict(dataset='Cook2019Herm (July 2020 corrected)', paper_doi='10.1038/s41586-019-1352-7',
            repository='https://github.com/openworm/ConnectomeToolbox', commit=COMMIT,
            source_path=SOURCE_PATH, source_git_blob_sha1=BLOB,
            workbook_git_blob_sha1='4d53c16e9a07c1ecfd068a02b39ccffd6b8a3eaa',
            retrieved_date='2026-10-24', includes_non_neuronal_cells=True))


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cache', type=Path)
    source.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('data/connectome/cook2019_herm.json'))
    args = parser.parse_args()
    if args.cache:
        raw = args.cache.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    export = compact_export(raw)
    text = json.dumps(export, ensure_ascii=False, separators=(',', ':')) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), nodes=len(export['nodes']), edges=len(export['edges']),
                         source_sha256=hashlib.sha256(raw).hexdigest(),
                         derived_sha256=hashlib.sha256(text.encode()).hexdigest())))


if __name__ == '__main__':
    main()
