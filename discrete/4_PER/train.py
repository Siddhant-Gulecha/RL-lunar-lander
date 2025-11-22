import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLander-v3"
EPISODES = 1000
LR = 5e-4
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 100000
WARMUP = 1000
TAU = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Epsilon (for exploration) ---
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.996

# --- PER Config ---
PER_ALPHA = 0.6  # Prioritization exponent (0=uniform, 1=full priority)
PER_BETA_START = 0.4  # Initial IS weight exponent
PER_BETA_END = 1.0  # Final IS weight exponent
PER_BETA_ANNEAL_STEPS = 500000  # Steps to anneal beta
PER_EPSILON = 1e-6  # Small constant to ensure non-zero priority


# ----------------------------
# SumTree (for PER)
# ----------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.data_ptr = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.data_ptr + self.capacity - 1
        self.data[self.data_ptr] = data
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


# ----------------------------
# Prioritized Replay Buffer
# ----------------------------
class PERBuffer:
    def __init__(self, capacity, alpha, beta_start, beta_end, beta_anneal_steps):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (beta_end - beta_start) / beta_anneal_steps
        self.max_priority = 1.0

    def push(self, s, a, r, ns, d):
        # New transitions get max priority to ensure they are sampled
        self.tree.add(self.max_priority, (s, a, r, ns, d))

    def sample(self, batch_size):
        batch = []
        indices = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        s, a, r, ns, d = zip(*batch)

        sampling_probs = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()  # Normalize weights

        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
            indices,
            torch.tensor(is_weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

    def update_priorities(self, indices, td_errors):
        priorities = (np.abs(td_errors) + PER_EPSILON) ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())
        for idx, p in zip(indices, priorities):
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# ----------------------------
# Dueling DQN Model
# ----------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(net, env, episodes=5):
    net.eval()
    scores = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a = net(s_t).argmax(1).item()
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = ns
            total += r
        scores.append(total)
    net.train()
    return np.mean(scores)


# ----------------------------
# Training Loop
# ----------------------------
def train():
    env = gym.make(ENV_NAME)
    replay = PERBuffer(REPLAY_SIZE, PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_BETA_ANNEAL_STEPS)
    policy = DuelingDQN().to(DEVICE)
    target = DuelingDQN().to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimz = optim.Adam(policy.parameters(), lr=LR)

    epsilon = EPS_START
    training_scores = []
    best_score = -1e9
    total_timesteps = 0

    print(f"Starting PER+Dueling+Double DQN training on {DEVICE}...")

    for ep in range(1, EPISODES + 1):
        s, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            total_timesteps += 1
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    a = policy(s_t).argmax(1).item()

            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            replay.push(s, a, r, ns, float(terminated))
            s = ns
            ep_reward += r

            if len(replay) >= WARMUP:
                bs, ba, br, bns, bd, indices, weights = replay.sample(BATCH_SIZE)

                with torch.no_grad():
                    # Double DQN Logic
                    next_a = policy(bns).argmax(1).unsqueeze(1)
                    next_q = target(bns).gather(1, next_a).squeeze(1)
                    target_q = br + GAMMA * next_q * (1 - bd)

                q_vals = policy(bs).gather(1, ba).squeeze(1)

                # Calculate TD Error for PER
                td_errors = target_q - q_vals

                # Calculate PER-weighted loss
                loss = F.smooth_l1_loss(q_vals, target_q, reduction='none')
                loss = (weights.squeeze(1) * loss).mean()

                optimz.zero_grad()
                loss.backward()
                optimz.step()

                # Update priorities in the SumTree
                replay.update_priorities(indices, td_errors.detach().cpu().numpy())

                # Anneal beta for IS weights
                replay.beta = min(1.0, replay.beta + replay.beta_increment)

                # Soft update target network
                for tp, pp in zip(target.parameters(), policy.parameters()):
                    tp.data.copy_(tp.data * (1 - TAU) + pp.data * TAU)

        training_scores.append(ep_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % 20 == 0:
            avg_eval = evaluate(policy, env)
            print(
                f"EP {ep:4d} | Timesteps: {total_timesteps:6d} | Eval Avg: {avg_eval:7.1f} | eps: {epsilon:.3f} | beta: {replay.beta:.3f}")
            if avg_eval >= best_score:
                best_score = avg_eval
                torch.save(policy.state_dict(), "per_dqn_best.pth")

    torch.save(policy.state_dict(), "per_dqn_final.pth")
    np.save("per_dqn_scores.npy", np.array(training_scores))
    print(f"Training complete. Best eval score: {best_score:.1f}")
    env.close()


if __name__ == "__main__":
    train()