import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class TravelingSalesmanEnv(gym.Env):
    def __init__(self, cost_matrix):
        super(TravelingSalesmanEnv, self).__init__()

        self.cost_matrix = np.array(cost_matrix)
        self.num_nodes = len(cost_matrix)

        if not np.allclose(self.cost_matrix, self.cost_matrix.T):
            raise ValueError("Cost matrix must be symmetric")
        if not np.allclose(np.diagonal(self.cost_matrix), 0):
            raise ValueError("Diagonal elements must be 0")

        self.observation_space = spaces.Dict({
            'current_node': spaces.Discrete(self.num_nodes),
            'available_nodes': spaces.MultiBinary(self.num_nodes)
        })
        self.action_space = spaces.Discrete(self.num_nodes)

        self.current_node = None
        self.available_nodes = None
        self.total_cost = None
        self.path_taken = None

    def step(self, action):
        if not self.available_nodes[action] or action >= self.num_nodes:
            return self._get_observation(), -1000, True, False, {'invalid_action': True}

        travel_cost = self.cost_matrix[self.current_node][action]
        self.total_cost += travel_cost
        self.path_taken.append(action)
        self.available_nodes[action] = 0
        self.current_node = action

        done = not np.any(self.available_nodes)
        if done:
            return_cost = self.cost_matrix[self.current_node][0]
            self.total_cost += return_cost
            self.path_taken.append(0)
            reward = -self.total_cost
        else:
            reward = -travel_cost

        return self._get_observation(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = 0
        self.available_nodes = np.ones(self.num_nodes, dtype=np.int8)
        self.available_nodes[0] = 0
        self.total_cost = 0
        self.path_taken = [0]
        return self._get_observation(), {}

    def _get_observation(self):
        return {
            'current_node': self.current_node,
            'available_nodes': self.available_nodes.copy()
        }

    def render(self):
        print(f"Path: {' -> '.join(map(str, self.path_taken))}")
        print(f"Total cost: {self.total_cost}")


def run_random_agent(env):
    obs, _ = env.reset()
    done = False
    while not done:
        valid_actions = np.where(obs['available_nodes'] == 1)[0]
        action = np.random.choice(valid_actions)
        obs, reward, done, _, _ = env.step(action)
    return env.path_taken.copy(), env.total_cost


# ---------------------------------------------------------------------------
# Policy Iteration helpers
# ---------------------------------------------------------------------------

def get_all_states(num_nodes):
    """All valid (current_node, available_mask) pairs.
    available_mask is an int where bit i=1 means city i is still unvisited.
    The current node is never in the available set.
    """
    states = []
    for current in range(num_nodes):
        for mask in range(1 << num_nodes):
            if not (mask >> current & 1):          # current not available → valid
                states.append((current, mask))
    return states


def available_from_mask(mask, num_nodes):
    return [i for i in range(num_nodes) if mask >> i & 1]


def transition(current, mask, action, cost_matrix):
    """One step: move to `action`, return (next_state, reward)."""
    reward    = -cost_matrix[current][action]
    new_mask  = mask & ~(1 << action)
    next_state = (action, new_mask)
    if new_mask == 0:                              # all cities visited → return home
        reward -= cost_matrix[action][0]
    return next_state, reward


def build_random_policy(states, num_nodes, seed=0):
    """Deterministic random policy: for each state fix one random action."""
    rng = np.random.default_rng(seed)
    policy = {}
    for state in states:
        current, mask = state
        available = available_from_mask(mask, num_nodes)
        policy[state] = int(rng.choice(available)) if available else None
    return policy


def policy_evaluation(policy, states, cost_matrix, gamma=1.0, theta=1e-6):
    """Iterative policy evaluation.  Returns V (dict) and history (list of value arrays)."""
    V       = {s: 0.0 for s in states}
    history = []

    while True:
        delta = 0.0
        for state in states:
            action = policy[state]
            if action is None:
                continue
            current, mask = state
            next_state, reward = transition(current, mask, action, cost_matrix)
            new_v  = reward + gamma * V[next_state]
            delta  = max(delta, abs(new_v - V[state]))
            V[state] = new_v
        history.append([V[s] for s in states])
        if delta < theta:
            break
    return V, history


def policy_improvement(V, states, cost_matrix, num_nodes, gamma=0.9):
    """Greedy one-step improvement over V.  Returns new policy and stable flag."""
    policy = {}
    for state in states:
        current, mask = state
        available = available_from_mask(mask, num_nodes)
        if not available:
            policy[state] = None
            continue
        best_action = max(
            available,
            key=lambda a: transition(current, mask, a, cost_matrix)[1]
                          + gamma * V[transition(current, mask, a, cost_matrix)[0]]
        )
        policy[state] = best_action
    return policy


def run_policy(policy, cost_matrix, num_nodes):
    """Follow a deterministic policy from city 0 and return (path, cost)."""
    current   = 0
    mask      = ((1 << num_nodes) - 1) & ~1   # all cities available except 0
    path      = [0]
    total     = 0
    while mask:
        action           = policy[(current, mask)]
        total           += cost_matrix[current][action]
        mask             = mask & ~(1 << action)
        current          = action
        path.append(current)
    total += cost_matrix[current][0]
    path.append(0)
    return path, total


def visualize_policy_iteration(history_random, history_improved, states):
    """Two-panel line plot showing state values across sweeps for both policies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')

    panels = [
        (axes[0], history_random,   'Random Policy — evaluation sweeps',   cm.plasma),
        (axes[1], history_improved, 'Improved Policy — evaluation sweeps',  cm.viridis),
    ]

    for ax, history, title, cmap in panels:
        ax.set_facecolor('#0f0f1a')
        arr     = np.array(history)           # (sweeps, states)
        n_states = arr.shape[1]
        colors  = cmap(np.linspace(0.15, 0.9, n_states))
        x       = np.arange(len(history))
        for j in range(n_states):
            ax.plot(x, arr[:, j], color=colors[j], linewidth=0.9, alpha=0.55)
        ax.set_title(title, color='white', fontsize=11)
        ax.set_xlabel('Sweep', color='#aaaacc')
        ax.set_ylabel('V(s)', color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.yaxis.label.set_color('#aaaacc')

    fig.suptitle('Policy Iteration — State Values Converging',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tsp_policy_iteration.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()


def visualize_tours(tours, costs, city_coords, num_nodes):
    colors = cm.plasma(np.linspace(0.1, 0.9, len(tours)))
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.patch.set_facecolor('#0f0f1a')
    axes = axes.flatten()

    best_idx = int(np.argmin(costs))

    for i, (path, cost, color) in enumerate(zip(tours, costs, colors)):
        ax = axes[i]
        ax.set_facecolor('#0f0f1a')

        # Tour edges
        xs = [city_coords[c][0] for c in path]
        ys = [city_coords[c][1] for c in path]
        ax.plot(xs, ys, '-', color=color, linewidth=1.8, alpha=0.85, zorder=1)

        # Arrows on each edge
        for j in range(len(path) - 1):
            x0, y0 = city_coords[path[j]]
            x1, y1 = city_coords[path[j + 1]]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2), zorder=2)

        # City nodes
        cx = [city_coords[c][0] for c in range(num_nodes)]
        cy = [city_coords[c][1] for c in range(num_nodes)]
        ax.scatter(cx, cy, s=80, color='white', zorder=3, edgecolors='#aaaacc', linewidths=0.8)

        # City labels
        for c in range(num_nodes):
            ax.text(city_coords[c][0], city_coords[c][1] + 0.05, str(c),
                    color='white', fontsize=8, ha='center', va='bottom', fontweight='bold')

        # Highlight start city
        ax.scatter(*city_coords[0], s=130, color='#ff4d6d', zorder=4,
                   edgecolors='white', linewidths=1.0)

        title_color = '#ffd700' if i == best_idx else 'white'
        suffix = '  ★ BEST' if i == best_idx else ''
        ax.set_title(f'Tour {i+1}  |  cost: {cost:.1f}{suffix}',
                     color=title_color, fontsize=9, pad=4)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])

        border_color = '#ffd700' if i == best_idx else '#333355'
        border_width = 1.5 if i == best_idx else 0.8
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

    fig.suptitle('TSP Random Agent — 10 Tours', color='white',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('tsp_tours.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    cost_matrix = [
        [0,  5,  9, 12, 10,  6],
        [5,  0,  7,  9, 12, 10],
        [9,  7,  0,  5, 10, 12],
        [12, 9,  5,  0,  6, 10],
        [10, 12, 10,  6,  0,  7],
        [6,  10, 12, 10,  7,  0]
    ]

    # 2D coordinates for each city (used only for visualization)
    city_coords = {
        0: (0.1, 0.5),
        1: (0.3, 0.9),
        2: (0.7, 0.85),
        3: (0.9, 0.5),
        4: (0.7, 0.15),
        5: (0.3, 0.1),
    }

    env = TravelingSalesmanEnv(cost_matrix)

    tours, costs = [], []
    for i in range(10):
        path, cost = run_random_agent(env)
        tours.append(path)
        costs.append(cost)
        print(f"Tour {i+1:2d}: {' -> '.join(map(str, path))}  |  cost = {cost}")

    print(f"\nBest:  {min(costs):.1f}")
    print(f"Worst: {max(costs):.1f}")
    print(f"Mean:  {np.mean(costs):.1f}")

    visualize_tours(tours, costs, city_coords, env.num_nodes)

    # ------------------------------------------------------------------
    # Policy Iteration
    # ------------------------------------------------------------------
    num_nodes_pi = env.num_nodes
    cost_mat_pi  = env.cost_matrix
    states       = get_all_states(num_nodes_pi)
    print(f"\nTotal states: {len(states)}")

    # 1. Build a fixed random policy and evaluate it
    rand_policy               = build_random_policy(states, num_nodes_pi, seed=0)
    V_random, history_random  = policy_evaluation(rand_policy, states, cost_mat_pi)
    path_rand, cost_rand      = run_policy(rand_policy, cost_mat_pi, num_nodes_pi)
    print(f"\nRandom policy  — sweeps to converge: {len(history_random)}")
    print(f"  Tour: {' -> '.join(map(str, path_rand))}  |  cost = {cost_rand}")

    # 2. Improve the policy greedily, then re-evaluate
    improved_policy              = policy_improvement(V_random, states, cost_mat_pi, num_nodes_pi)
    V_improved, history_improved = policy_evaluation(improved_policy, states, cost_mat_pi)
    path_imp, cost_imp           = run_policy(improved_policy, cost_mat_pi, num_nodes_pi)
    print(f"\nImproved policy — sweeps to converge: {len(history_improved)}")
    print(f"  Tour: {' -> '.join(map(str, path_imp))}  |  cost = {cost_imp}")

    visualize_policy_iteration(history_random, history_improved, states)