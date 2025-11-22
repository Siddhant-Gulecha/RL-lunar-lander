import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ----------------------------
# Config (Stabilized)
# ----------------------------
ENV_NAME = "LunarLander-v3"
EPISODES = 1000
LR = 1e-4  # Reduced LR for stability
GAMMA = 0.99
BATCH_SIZE = 128  # Increased batch size for smoother gradients
REPLAY_SIZE = 100000
WARMUP = 1000
TAU = 0.001  # Slower soft update
GRAD_CLIP = 0.5  # Tighter gradient clipping
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE)
        )

    def __len__(self):
        return len(self.buf)


# ----------------------------
# Dueling DQN Model
# ----------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Added extra layer for capacity
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
    target_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    replay = ReplayBuffer(REPLAY_SIZE)

    epsilon = EPS_START
    training_scores = []
    best_score = -1e9

    print(f"Starting Stabilized Dueling DQN training on {DEVICE}...")

    for ep in range(1, EPISODES + 1):
        s, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    a = policy_net(s_t).argmax(1).item()

            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            replay.push(s, a, r, ns, float(terminated))
            s = ns
            ep_reward += r

            if len(replay) >= WARMUP:
                bs, ba, br, bns, bd = replay.sample(BATCH_SIZE)

                with torch.no_grad():
                    next_actions = policy_net(bns).argmax(1).unsqueeze(1)
                    next_q = target_net(bns).gather(1, next_actions).squeeze(1)
                    target_q = br + GAMMA * next_q * (1 - bd)

                q_vals = policy_net(bs).gather(1, ba).squeeze(1)
                loss = loss_fn(q_vals, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
                optimizer.step()

                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.copy_(tp.data * (1 - TAU) + pp.data * TAU)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        training_scores.append(ep_reward)

        if ep % 20 == 0:
            avg_eval = evaluate(policy_net, env)
            print(
                f"Ep {ep:4d} | Train Avg (last 10): {np.mean(training_scores[-10:]):7.1f} | Eval: {avg_eval:7.1f} | Eps: {epsilon:.3f}")
            if avg_eval > best_score and avg_eval > 200:
                best_score = avg_eval
                torch.save(policy_net.state_dict(), "dueling_dqn_lunar_best.pth")
                print(f"--- New Best Model Saved: {best_score:.1f} ---")

    np.save("dueling_dqn_lunar_scores.npy", np.array(training_scores))
    print("Training complete.")
    env.close()


if __name__ == "__main__":
    train()