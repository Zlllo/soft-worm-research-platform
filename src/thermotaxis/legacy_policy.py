"""Frozen, explicitly privileged adapters for the existing four-action policies.

No training, replay, temperature oracle substitution, or silent state padding occurs
here. Grid indices are the old Worm2D screen coordinates (positive y downward).
"""
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import numpy as np

LEGACY_ACTION_VECTORS = ((0, -1), (0, 1), (-1, 0), (1, 0))
LEGACY_ACTION_HEADINGS = (-math.pi/2, math.pi/2, math.pi, 0.)


def legacy_action_heading(action, *, y_axis_up=False):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
        raise ValueError('legacy action must be an integer in [0, 3]')
    heading = LEGACY_ACTION_HEADINGS[int(action)]
    return -heading if y_axis_up else heading


class FrozenQTable:
    """Exact greedy Q lookup; ties use a private, explicit seeded RNG.

    Observation condition is absolute grid position (privileged). Table storage
    is copied, immutable and does not share memory with a training worm.
    """
    observation_condition = 'legacy_absolute_grid_position'

    def __init__(self, q_table, *, tie_seed):
        q = np.array(q_table, dtype=float, copy=True)
        if q.ndim != 3 or q.shape[2] != 4 or min(q.shape[:2]) < 2 or not np.isfinite(q).all():
            raise ValueError('Q table must have finite shape (height>=2, width>=2, 4)')
        if isinstance(tie_seed, bool) or not isinstance(tie_seed, (int, np.integer)) or tie_seed < 0:
            raise ValueError('tie_seed must be a nonnegative integer')
        q.setflags(write=False)
        self._q = q
        self._rng = np.random.default_rng(tie_seed)
        self.height, self.width = q.shape[:2]

    @property
    def q_table(self):
        # Copy prevents callers re-enabling writes on the private owner array.
        result = self._q.copy()
        result.setflags(write=False)
        return result

    def select_action(self, x_index, y_index):
        for value, limit in ((x_index, self.width), (y_index, self.height)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < limit:
                raise ValueError('grid indices must be bounded integers; coordinates are not silently clipped')
        values = self._q[int(y_index), int(x_index)]
        choices = np.flatnonzero(values == values.max())
        return int(self._rng.choice(choices))

    @classmethod
    def from_json(cls, path, *, tie_seed):
        payload = json.loads(Path(path).read_text())
        if payload.get('schema') != 'legacy-four-action-qtable-v1':
            raise ValueError('unknown Q table schema')
        if payload.get('action_vectors') != [list(v) for v in LEGACY_ACTION_VECTORS]:
            raise ValueError('Q table action mapping differs from Worm2D')
        return cls(payload['q_table'], tie_seed=tie_seed)

    @classmethod
    def from_legacy_text(cls, path, *, width, height, tie_seed):
        """Read save_q_table's tab-separated text, even when named .npy.

        The old exporter rounds Q values to 3 decimals; imported greedy ties
        therefore refer to the exported policy, not an unavailable exact table.
        """
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        if len(lines) < 3 or lines[1].split('\t')[2:6] != ['上Q', '下Q', '左Q', '右Q']:
            raise ValueError('expected the old four-action save_q_table header')
        q = np.full((height, width, 4), np.nan)
        seen = set()
        for line in lines[3:]:
            fields = line.split('\t')
            match = re.fullmatch(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', fields[0])
            if not match or len(fields) != 7:
                raise ValueError('malformed legacy Q table row')
            x, y = map(int, match.groups())
            if not 0 <= x < width or not 0 <= y < height or (x,y) in seen:
                raise ValueError('duplicate or out-of-range legacy Q table cell')
            seen.add((x,y)); q[y,x] = [float(v) for v in fields[2:6]]
        if len(seen) != width*height:
            raise ValueError('legacy Q table has missing cells')
        return cls(q, tie_seed=tie_seed)


@dataclass(frozen=True)
class LegacyGridMap:
    """Declared SI rectangle mapped to an old grid; positive y is downward.

    Every old unit is length_x_m/(width-1) in x, length_y_m/(height-1)
    in y. This is an experimental coordinate calibration, not a historical
    physical unit. Values outside the rectangle raise instead of clipping.
    """
    width: int
    height: int
    length_x_m: float
    length_y_m: float
    y_axis_up: bool = False

    def __post_init__(self):
        for value in (self.width, self.height):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError('grid dimensions must be integers >= 2')
        if not all(math.isfinite(v) and v > 0 for v in (self.length_x_m,self.length_y_m)):
            raise ValueError('physical lengths must be finite positive')

    def indices(self, position_m):
        p = np.asarray(position_m, dtype=float)
        lengths = np.array([self.length_x_m,self.length_y_m])
        if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > lengths):
            raise ValueError('physical position must lie in the declared rectangle')
        if self.y_axis_up:
            p = p.copy(); p[1] = self.length_y_m - p[1]
        indices = np.rint(p/lengths*np.array([self.width-1,self.height-1])).astype(int)
        return int(indices[0]), int(indices[1])

    def callback(self, policy):
        if (policy.width, policy.height) != (self.width,self.height):
            raise ValueError('map and Q table grid sizes differ')
        def choose(position_m):
            return policy.select_action(*self.indices(position_m))
        return choose


class LegacyFrameStack:
    """Four genuine 8-D frames, warmup by repeating first actual observation."""
    def __init__(self):
        self._frames = deque(maxlen=4)

    def push(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (8,) or not np.isfinite(frame).all():
            raise ValueError('legacy environmental frame must have exactly 8 finite values')
        if not self._frames:
            self._frames.extend(frame.copy() for _ in range(4))
        else:
            self._frames.append(frame.copy())
        return np.concatenate(tuple(self._frames))


class FrozenLegacyDQN:
    """Optional PyTorch inference from real legacy state_dict (32 -> 4).

    Caller supplies genuine legacy 8-D frames. New local-memory observations
    must never be padded/relabelled to feed this policy. Architecture explicit.
    """
    observation_condition = 'legacy_eight_feature_oracle_four_frame_stack'

    def __init__(self, checkpoint, *, architecture):
        import torch
        from torch import nn
        if architecture not in ('simple', 'dueling'):
            raise ValueError('architecture must be simple or dueling')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or 'fc1.weight' not in state:
            raise ValueError('checkpoint must be a raw legacy network state_dict')
        hidden, input_dim = state['fc1.weight'].shape
        if input_dim != 32:
            raise ValueError('only genuine legacy four-frame 8-D checkpoints accepted')
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, hidden); self.fc2 = nn.Linear(hidden, hidden)
                if architecture == 'simple': self.fc3 = nn.Linear(hidden,4)
                else:
                    self.value_head = nn.Linear(hidden,1); self.advantage_head = nn.Linear(hidden,4)
            def forward(self,x):
                z = torch.relu(self.fc2(torch.relu(self.fc1(x))))
                if architecture == 'simple': return self.fc3(z)
                v,a = self.value_head(z),self.advantage_head(z)
                return v+a-a.mean(dim=1,keepdim=True)
        self._torch = torch
        self._network = Network()
        self._network.load_state_dict(state,strict=True)
        self._network.eval()
        for parameter in self._network.parameters(): parameter.requires_grad_(False)
        self._stack = LegacyFrameStack()

    def select_action(self, actual_legacy_frame):
        x = self._stack.push(actual_legacy_frame)
        with self._torch.inference_mode():
            q = self._network(self._torch.from_numpy(x).unsqueeze(0))
        if tuple(q.shape) != (1,4) or not bool(self._torch.isfinite(q).all()):
            raise ValueError('legacy network produced invalid Q values')
        return int(q.argmax().item())
