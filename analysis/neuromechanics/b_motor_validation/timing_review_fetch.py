"""Fetch only pinned official documentation/source, never import NEURON."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import urllib.request

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/c1-current-timing-review-2026-11-10'
OFFICIAL={
    'fadvance.cpp':'https://raw.githubusercontent.com/neuronsimulator/nrn/9.0.2/src/nrnoc/fadvance.cpp',
    'treeset.cpp':'https://raw.githubusercontent.com/neuronsimulator/nrn/9.0.2/src/nrnoc/treeset.cpp',
    'hh.mod':'https://raw.githubusercontent.com/neuronsimulator/nrn/9.0.2/src/nrnoc/hh.mod',
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=ROOT/'official-9.0.2';dest.mkdir(parents=True,exist_ok=True)
    records={}
    for name,url in OFFICIAL.items():
        target=dest/name
        if not target.exists():
            with urllib.request.urlopen(url,timeout=20) as r:
                payload=r.read();assert r.status==200 and len(payload)>100
            target.write_bytes(payload)
        records[name]=dict(url=url,sha256=sha(target),bytes=target.stat().st_size)
    package=Path('/tmp/worm-progress-env/lib/python3.12/site-packages/neuron/.data')
    local=ROOT/'installed-package';local.mkdir(exist_ok=True)
    for rel in ['include/nrnsemanticversion.h','share/modfile/hh.mod']:
        p=package/rel;target=local/Path(rel).name;shutil.copy2(p,target)
        records['installed/'+Path(rel).name]=dict(source=str(p),sha256=sha(target))
    records['environment']=dict(neuron_distribution_version=importlib.metadata.version('neuron'),neuron_imported=False)
    records['documentation']=dict(url='https://www.neuronsimulator.org/en/9.0.2/progref/simctrl/programmatic.html',
        access='read through web tool; no local HTML download claimed',sections=['fadvance','fcurrent','secondorder','finitialize'])
    records['transport_note']='Official raw sources saved from the GitHub connector because direct shell DNS access was unavailable. No simulator was loaded.'
    (ROOT/'official-sources.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(official_files=len(OFFICIAL),package_version=records['environment']['neuron_distribution_version'])))

if __name__=='__main__':main()
