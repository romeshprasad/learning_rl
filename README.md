# Learning RL with the Traveling Salesman Problem

This repo is a teaching sandbox for reinforcement learning, built around the
Traveling Salesman Problem (TSP). It starts with classic dynamic programming
(policy iteration) and a from-scratch DQN agent, and is meant to grow with
more algorithms over time.

## Repo layout

| File | What it is |
|---|---|
| [tsp_env.py](tsp_env.py) | The TSP environment (Gymnasium `Env`), plus a random-agent baseline and tour visualization. |
| [tsp_env_policy_iteartion.py](tsp_env_policy_iteartion.py) | Same environment, extended with tabular policy evaluation/improvement (policy iteration) and a value-convergence plot. |
| [dqn_tsp.py](dqn_tsp.py) | A from-scratch DQN agent (no RL library) that learns to solve the TSP env: replay buffer, target network, epsilon-greedy exploration, and action masking. |
| [datasets/](datasets/) | Real TSP cost matrices (TSPLIB-style instances) you can swap in instead of the toy 6-city example. |

## The environment

`TravelingSalesmanEnv` (in `tsp_env.py`) takes a symmetric cost matrix and
exposes a Gymnasium-compatible interface:

- **Observation**: `{'current_node': int, 'available_nodes': MultiBinary}`
- **Action**: `Discrete(num_nodes)` — pick the next city to visit
- **Reward**: negative travel cost per step; on the final step the cost of
  returning to the start city is added too
- An action on an already-visited city ends the episode immediately with a
  large penalty (`-1000`)

Every script in this repo builds on this same environment.

## Datasets

The `datasets/` folder contains cost matrices for standard TSP benchmark
instances (TSPLIB), one CSV per instance. Each file is an N x N symmetric
matrix of travel costs with a zero diagonal — exactly the format
`TravelingSalesmanEnv` expects.

| File | Cities |
|---|---|
| `gr17_TSPJ_TT.csv` | 17 |
| `gr21_TSPJ_TT.csv` | 21 |
| `fri26_TSPJ_TT.csv` | 26 |
| `gr24_TSPJ_TT.csv` | 24 |
| `bays29_TSPJ_TT.csv` | 29 |
| `gr48_TSPJ_TT.csv` | 48 |
| `eil51_TSPJ_TT.csv` | 51 |
| `berlin52_TSPJ_TT.csv` | 52 |
| `eil76_TSPJ_TT.csv` | 76 |
| `eil101_TSPJ_TT.csv` | 101 |

To run on a real dataset instead of the toy 6-city matrix hardcoded in the
scripts' `if __name__ == "__main__":` blocks, load a CSV with `numpy` and
pass it to `TravelingSalesmanEnv`:

```python
import numpy as np
from tsp_env import TravelingSalesmanEnv

cost_matrix = np.loadtxt("datasets/gr17_TSPJ_TT.csv", delimiter=",")
env = TravelingSalesmanEnv(cost_matrix)
```

Note: the larger instances (`eil101`, `berlin52`, etc.) have far more states
than the toy example. Policy iteration's `get_all_states` enumerates
`num_nodes * 2^num_nodes` states, so it is only practical for small
instances (roughly up to ~20 cities). DQN scales much better and is the
right tool to try on the bigger datasets.

## Running things

```bash
pip install gymnasium numpy matplotlib torch

python tsp_env.py                      # random-agent baseline + tour plot
python tsp_env_policy_iteartion.py     # policy iteration + convergence plot
python dqn_tsp.py                      # train DQN and run a greedy rollout
```

## What's next

More RL algorithms will be added on top of the same `TravelingSalesmanEnv`
(e.g. value iteration, REINFORCE/policy gradient, PPO). Each new algorithm
should follow the same pattern: reuse the environment, add a new training
script, and demonstrate it against both the toy matrix and the datasets in
`datasets/`.
