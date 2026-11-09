import json
from pathlib import Path

import numpy as np
import pytest

from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring
from thermotaxis.neuromechanics.full302_c1 import connection_plan


def metadata():
    w=load_anatomical_body_wiring()
    return {'neuron_info':{n:[None,None,None,['GABA'] if n.startswith(('VD','DD')) else []] for n in w.neuron_names}}


def test_full302_plan_keeps_every_nmj_and_unique_bidirectional_gap():
    plan=connection_plan(metadata(),unresolved_polarity='exc')
    assert len(plan['neurons'])==302 and len(plan['muscles'])==95
    assert len(plan['chemical'])==4665
    assert len(set(plan['gap_pairs']))==len(plan['gap_pairs'])
    assert all((b,a) not in set(plan['gap_pairs']) for a,b in plan['gap_pairs'])
    assert all(a!=b for a,b in plan['gap_pairs'])
    names=plan['neurons']+plan['muscles'];w=load_anatomical_body_wiring()
    mm={(a,b) for a,b in plan['gap_pairs'] if a in plan['muscles'] and b in plan['muscles']}
    assert len(mm)==np.count_nonzero(np.triu(w.muscle_gap,1))
    cross={(a,b) for a,b in plan['gap_pairs'] if a in plan['neurons'] and b in plan['muscles']}
    assert len(cross)==5
    assert len({n for n in names})==397


def test_signs_require_explicit_assumption_and_preserve_inhibition():
    a=connection_plan(metadata(),unresolved_polarity='exc')
    b=connection_plan(metadata(),unresolved_polarity='inh')
    assert any(sign=='inh' for _,_,sign in a['chemical'])
    signs={r['name']:r for r in a['polarity_table']}
    assert signs['VD12']['polarity']=='inh'
    assert signs['AFDL']['status']=='receptor_unresolved_explicit_assumption'
    assert all(sign=='inh' for _,_,sign in b['chemical'])
    with pytest.raises(ValueError,match='explicit'):
        connection_plan(metadata(),unresolved_polarity=None)
