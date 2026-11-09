"""Persistent NEURON assembly of pinned c302 C1 components on Cook wiring.

Anatomy sets masks only. Unit-mask conductances and unresolved chemical
polarities are declared assumptions, not estimates from EM section counts.
NEURON is imported only when constructing the runtime in its isolated env.
"""
import json
import os
from pathlib import Path

import numpy as np

from .anatomical_body import load_anatomical_body_wiring


def connection_plan(metadata, *, unresolved_polarity):
    if unresolved_polarity not in ('exc', 'inh'):
        raise ValueError('explicit unresolved polarity exc/inh is required')
    w = load_anatomical_body_wiring()
    names = w.neuron_names + w.muscle_names
    n = len(names)
    chemical = np.zeros((n, n), bool)
    chemical[:302, :302] = w.neuron_chemical > 0
    chemical[:302, 302:] = w.nmj_chemical > 0
    chemical[302:, 302:] = w.muscle_chemical > 0
    gap = np.zeros((n, n))
    gap[:302, :302] = w.neuron_gap
    gap[:302, 302:] = w.neuron_muscle_gap
    gap[302:, :302] = w.neuron_muscle_gap.T
    gap[302:, 302:] = w.muscle_gap
    assert np.array_equal(gap, gap.T)
    signs = []
    for name in w.neuron_names:
        nt = metadata['neuron_info'][name][3]
        gaba = 'GABA' in nt
        signs.append(dict(name=name, cached_neurotransmitters=nt,
            polarity='inh' if gaba else unresolved_polarity,
            status='GABA_inhibitory_model_rule' if gaba else 'receptor_unresolved_explicit_assumption'))
    rows = [(names[i], names[j], signs[i]['polarity']) for i, j in zip(*np.nonzero(chemical))]
    pairs = [(names[i], names[j]) for i, j in zip(*np.nonzero(np.triu(gap > 0, 1)))]
    return dict(neurons=list(w.neuron_names), muscles=list(w.muscle_names), chemical=rows, gap_pairs=pairs,
        polarity_table=signs, gap_diagonal_entries_omitted=int(np.count_nonzero(np.diag(gap))),
        weighting='one C1 base conductance per present edge; EM magnitudes are preserved in input data but not interpreted as efficacy')


class C1Network:
    """One persistent network per process, with continuous membrane/synapse state."""
    def __init__(self, kernel_directory, plan, *, dt_ms=.05, chemical_lesion=()):
        import neuron
        self.neuron = neuron
        self.h = neuron.h
        h = self.h
        if list(h.allsec()):
            raise RuntimeError('construct each network in a fresh process')
        kernel = Path(kernel_directory).resolve()
        # HOC's filename boundary accepts ASCII only in this NEURON build.
        # Resolve the real directory in Python and pass relative ASCII filenames.
        original_directory = Path.cwd()
        try:
            os.chdir(kernel)
            if not neuron.load_mechanisms('.'):
                raise RuntimeError('cannot load pinned C1 mechanisms')
            h.load_file('stdrun.hoc')
            for name in ('GenericNeuronCell', 'GenericMuscleCell'):
                if not h.load_file(name+'.hoc'):
                    raise RuntimeError('cannot load C1 cell template')
        finally:
            os.chdir(original_directory)
        self.plan = plan
        self.cells = {name: h.GenericNeuronCell() for name in plan['neurons']}
        self.cells.update({name: h.GenericMuscleCell() for name in plan['muscles']})
        # Match the exported C1 initialization before concentration mechanisms run.
        h.cai0_ca_ion = 0
        h.cao0_ca_ion = 2
        self.synapses = []
        self.chemical_connections = {}
        lesion = set(map(tuple, chemical_lesion))
        for pre, post, sign in plan['chemical']:
            syn = getattr(h, 'neuron_to_neuron_'+sign+'_syn')(self.cells[post].soma(.5))
            syn.weight = 0 if (pre, post) in lesion else 1
            h.setpointer(self.cells[pre].soma(.5)._ref_v, 'vpeer', syn)
            self.synapses.append(syn)
            self.chemical_connections[(pre, post)] = syn
        self.gap_connections = []
        for a, b in plan['gap_pairs']:
            # One anatomical pair -> two equal/opposite current point processes.
            pair = []
            for pre, post in ((a, b), (b, a)):
                syn = h.neuron_to_neuron_elec_syn(self.cells[post].soma(.5))
                syn.weight = 1
                h.setpointer(self.cells[pre].soma(.5)._ref_v, 'vpeer', syn)
                self.synapses.append(syn); pair.append(syn)
            self.gap_connections.append(pair)
        self.clamps = []
        self.initializers = [h.FInitializeHandler(0, self._initial_voltage),
                             h.FInitializeHandler(1, self._initial_ions)]
        h.dt = dt_ms
        h.steps_per_ms = 1 / dt_ms
        h.cvode_active(0)
        self.dt_ms = dt_ms

    def _initial_voltage(self):
        for cell in self.cells.values():
            cell.set_initial_v()

    def _initial_ions(self):
        for cell in self.cells.values():
            cell.set_initial_ion_properties()

    def add_current(self, name, *, start_ms, duration_ms, amplitude_pA):
        stim = self.h.IClamp(self.cells[name].soma(.5))
        stim.delay, stim.dur, stim.amp = start_ms, duration_ms, amplitude_pA / 1000
        self.clamps.append(stim)
        return stim

    def reset(self):
        self.h.finitialize()

    def advance(self, duration_ms):
        count = round(duration_ms / self.dt_ms)
        if count < 1 or abs(count * self.dt_ms-duration_ms) > 1e-10:
            raise ValueError('duration must be a positive integer number of neural steps')
        for _ in range(count):
            self.h.fadvance()

    def state(self):
        result = dict(time_s=float(self.h.t / 1000),
            neuron_voltage_mV=np.array([self.cells[n].soma(.5).v for n in self.plan['neurons']]),
            muscle_voltage_mV=np.array([self.cells[n].soma(.5).v for n in self.plan['muscles']]),
            muscle_calcium_mM=np.array([self.cells[n].soma(.5).cai for n in self.plan['muscles']]))
        if not all(np.isfinite(v).all() for v in result.values()):
            raise RuntimeError('nonfinite C1 state')
        return result
