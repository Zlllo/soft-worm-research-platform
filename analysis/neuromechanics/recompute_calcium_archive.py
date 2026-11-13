"""Recompute an extracted assay using only files below the current directory.

Preserves the original lock bytes. The one absolute Python __file__ entry is
explicitly aliased to the identical archived runner; no original-tree reads.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import calcium_body_transfer as assay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/calcium-body-transfer-2026-11-13'))
    parser.add_argument('--output',type=Path,default=Path('docs/reports/assets/calcium-body-transfer-2026-11-13'))
    args=parser.parse_args()
    aliases={}
    allowed=Path('analysis/neuromechanics/calcium_body_transfer.py')
    for name in json.loads((args.root/'source-lock.json').read_text()):
        path=Path(name)
        if path.is_absolute():
            assert path.parts[-3:]==allowed.parts, f'Unknown absolute lock entry: {name}'
            aliases[name]=allowed
    original_sha=assay.sha
    base=Path.cwd().resolve()
    def relocated_sha(path):
        path=Path(path)
        if path.is_absolute():
            path=aliases[str(path)]
        resolved=(base/path).resolve()
        assert resolved.is_relative_to(base), f'Hash read outside extracted payload: {resolved}'
        return original_sha(resolved)
    assay.sha=relocated_sha
    sys.argv=['summarize_calcium_body_transfer.py','--root',str(args.root),'--output',str(args.output)]
    runpy.run_path('analysis/neuromechanics/summarize_calcium_body_transfer.py',run_name='__main__')


if __name__=='__main__':main()
