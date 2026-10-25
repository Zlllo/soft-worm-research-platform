import sys,json
from pathlib import Path
sys.path.insert(0,'analysis')
import torch
from thermotaxis.navigation_policy import NavigationPolicy
from thermotaxis.navigation import NavigationConfig
from navigation_teacher import data,fit_bc
from train_navigation import evaluate
from navigation_preflight import summary
root=Path('results/navigation-gru16-bc-preflight-2026-10-25');root.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.manual_seed(7);model=NavigationPolicy('gru16','full','adaptive')
record=fit_bc(model,data(),7,updates=8192,burn=64,length=64)
torch.save(model.state_dict(),root/'gru16-adaptive-7.pt')
rows=evaluate(model,NavigationConfig(),list(range(42001,42017)))
result=dict(status='development_extended_imitation_check',training=record,summary=summary(rows),rows=rows)
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['summary'],flush=True)
