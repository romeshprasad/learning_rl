"""
A simple, from-scratch DQN for the TSP environment in tsp_env.py.

No RL library is used (no stable-baselines, no rllib, etc). We use PyTorch
only for the neural network and its optimizer. Everything else -- the
replay buffer, target network update, epsilon-greedy exploration, action
masking, and the training loop -- is written by hand so students can see
exactly how DQN works end to end.

State representation:
    We turn the dict observation {'current_node', 'available_nodes'} into
    a single flat vector:
        [one_hot(current_node)  (num_nodes,)
         available_nodes        (num_nodes,)   -- 1 if visitable, 0 otherwise]
    So the input size to the network is 2 * num_nodes.

Action masking:
    The env has a fixed action space (Discrete(num_nodes)) but only a
    subset of actions are legal at any time (the unvisited cities). We
    never let the agent pick an illegal action: both during exploration
    (random choice is restricted to valid actions) and during exploitation
    (argmax is taken only over Q-values of valid actions).
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tsp_env import TravelingSalesmanEnv


# ----------------------------------------------------------------------
# 1. Turn the dict observation into a flat numpy vector for the network
# ----------------------------------------------------------------------
def obs_to_state(obs, num_nodes):
    current_one_hot = np.zeros(num_nodes, dtype=np.float32)
    current_one_hot[obs['current_node']] = 1.0
    available = obs['available_nodes'].astype(np.float32)
    return np.concatenate([current_one_hot, available])


# ----------------------------------------------------------------------
# 2. The Q-network: a small MLP mapping state -> Q-value per action
# ----------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 3. Replay buffer: stores past transitions so we can sample random
#    mini-batches instead of training only on the most recent step.
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# 4. Action selection: epsilon-greedy, restricted to valid actions only
# ----------------------------------------------------------------------
def select_action(q_network, state, available_nodes, epsilon, device):
    valid_actions = np.where(available_nodes == 1)[0]

    if random.random() < epsilon:
        return int(np.random.choice(valid_actions))

    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_network(state_t).squeeze(0).cpu().numpy()

    # Mask out invalid actions by setting their Q-value to -inf so argmax
    # can never pick them.
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q))


# ----------------------------------------------------------------------
# 5. One gradient step on a mini-batch sampled from the replay buffer
# ----------------------------------------------------------------------
def train_step(q_network, target_network, optimizer, buffer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones, next_masks = buffer.sample(batch_size)

    states = torch.as_tensor(states, device=device)
    actions = torch.as_tensor(actions, device=device)
    rewards = torch.as_tensor(rewards, device=device)
    next_states = torch.as_tensor(next_states, device=device)
    dones = torch.as_tensor(dones, device=device)
    next_masks = torch.as_tensor(next_masks, device=device)

    # Q(s, a) for the actions actually taken
    q_values = q_network(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a') over VALID next actions only.
    with torch.no_grad():
        next_q_values = target_network(next_states)
        # Mask invalid next actions with -inf before taking the max.
        next_q_values = next_q_values.masked_fill(next_masks == 0, float('-inf'))
        # If terminal, there are no next actions -- avoid -inf propagating
        # into the target by zeroing it out via (1 - done) below.
        next_q_values = torch.nan_to_num(next_q_values, neginf=0.0)
        max_next_q = next_q_values.max(dim=1).values
        target = rewards + gamma * (1.0 - dones) * max_next_q

    loss = nn.functional.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ----------------------------------------------------------------------
# 6. Main training loop
# ----------------------------------------------------------------------
def train(env, num_episodes=2000, gamma=0.99, lr=1e-3, batch_size=64,
          buffer_capacity=10000, epsilon_start=1.0, epsilon_end=0.05,
          epsilon_decay=0.995, target_update_every=20, device=None):

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes = env.num_nodes
    state_size = 2 * num_nodes

    q_network = QNetwork(state_size, num_nodes).to(device)
    target_network = QNetwork(state_size, num_nodes).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    episode_costs = []
    best_cost = float('inf')
    best_path = None

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = obs_to_state(obs, num_nodes)
        done = False

        while not done:
            action = select_action(q_network, state, obs['available_nodes'], epsilon, device)
            next_obs, reward, done, _, info = env.step(action)
            next_state = obs_to_state(next_obs, num_nodes)

            # Mask of which actions are valid from next_state. On a
            # terminal step nothing is valid, but it won't be used since
            # we zero out the bootstrap term for done=True anyway.
            next_mask = next_obs['available_nodes'].astype(np.float32)

            buffer.push(state, action, reward, next_state, float(done), next_mask)

            state = next_state
            obs = next_obs

            train_step(q_network, target_network, optimizer, buffer,
                       batch_size, gamma, device)

        if episode % target_update_every == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        cost = env.total_cost
        episode_costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_path = env.path_taken.copy()

        if episode % 100 == 0:
            avg_cost = np.mean(episode_costs[-100:])
            print(f"Episode {episode:5d} | epsilon {epsilon:.3f} | "
                  f"avg cost (last 100) {avg_cost:8.2f} | best cost {best_cost:8.2f}")

    return q_network, episode_costs, best_path, best_cost


# ----------------------------------------------------------------------
# 7. Greedy rollout with a trained network (epsilon = 0)
# ----------------------------------------------------------------------
def run_greedy(env, q_network, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = env.num_nodes

    obs, _ = env.reset()
    done = False
    while not done:
        state = obs_to_state(obs, num_nodes)
        action = select_action(q_network, state, obs['available_nodes'], epsilon=0.0, device=device)
        obs, reward, done, _, _ = env.step(action)

    return env.path_taken.copy(), env.total_cost


if __name__ == "__main__":
    cost_matrix = [
        [0,  5,  9, 12, 10,  6],
        [5,  0,  7,  9, 12, 10],
        [9,  7,  0,  5, 10, 12],
        [12, 9,  5,  0,  6, 10],
        [10, 12, 10,  6,  0,  7],
        [6,  10, 12, 10,  7,  0]
    ]

    env = TravelingSalesmanEnv(cost_matrix)

    q_network, episode_costs, best_path, best_cost = train(env, num_episodes=2000)

    print(f"\nBest cost found during training: {best_cost:.1f}")
    print(f"Best path: {' -> '.join(map(str, best_path))}")

    path, cost = run_greedy(env, q_network)
    print(f"\nGreedy rollout after training:")
    print(f"Path: {' -> '.join(map(str, path))}")
    print(f"Cost: {cost:.1f}")
