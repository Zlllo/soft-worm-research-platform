"""Exploratory fixed-feature pulse in archived real8 actors; run from repository root."""
import io, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import torch
from thermotaxis.navigation_window_policy import WindowActorCritic
from diagnose_navigation_failures import trace_actor

archive = Path('docs/reports/assets/navigation-study-2026-10-25/models-and-sources.zip')
torch.set_num_threads(1)
rows = []
with zipfile.ZipFile(archive) as z:
    for i in range(6):
        interface = 'full' if i < 3 else 'restricted'
        model = WindowActorCritic('real8', interface)
        name = f'results/navigation-factorial-2026-10-25/fit-{i:03d}/model-RL262144.pt'
        data = z.read(name)
        model.load_state_dict(torch.load(io.BytesIO(data), map_location='cpu', weights_only=True))
        model.eval()
        # Direct standardized feature impulse, deliberately not a physical temperature window.
        x = torch.zeros(2, 41, 7); x[:,:,6] = 1
        x[1,10,0] = .02  # 1 mK / 0.05 K scaling
        with torch.no_grad():
            out, states, _ = trace_actor(model,x)
            assert torch.allclose(out, model.actor_mean(x), atol=1e-6)
            delta=(states[1]-states[0]).numpy()
            action=(model.actor(states[1,:,model.outputs])-model.actor(states[0,:,model.outputs])).flatten().numpy()
        rows.append(dict(interface=interface,seed=[7,17,27][i%3],sha256=hashlib.sha256(data).hexdigest(),
            pulse_node_delta=delta[10].tolist(),after_05s_node_delta=delta[15].tolist(),after_10s_node_delta=delta[20].tolist(),
            pulse_action=float(action[10]),after_05s_action=float(action[15]),after_10s_action=float(action[20])))
print(json.dumps(dict(protocol='Direct single-sample feature pulse; zero phase, no window mean preprocessing; exploratory, not physical temperature or biological memory.', archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), nodes=['AFDL','AFDR','AIYL','AIYR','AIZL','AIZR','RIAL','RIAR'],rows=rows),indent=2))
