import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, NormalizeObservation, NormalizeReward, \
    TransformReward

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLanderContinuous-v3"
EPISODES = 2000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
REPLAY_SIZE = 100000
WARMUP = 5000
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
EXPLORE_NOISE = 0.1
MAX_EPISODE_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TQC Specific Config ---
N_QUANTILES = 25
N_DROP = 5


# ----------------------------
# Custom Reward Wrapper
# ----------------------------
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Additional Penalties
        reward -= 0.1  # Living penalty
        reward -= np.sum(np.abs(action)) * 0.1  # Fuel penalty

        # 2. Crash Logic
        if terminated:
            if reward < -50:  # If crash
                reward += 100.0  # Refund default penalty

                vx = obs[2]
                vy = obs[3]
                angle = obs[4]
                speed = np.sqrt(vx ** 2 + vy ** 2)

                # Physics penalties
                reward -= (speed * 50.0)
                reward -= (np.abs(angle) * 20.0)

        return obs, reward, terminated, truncated, info


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(action), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(reward), dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(done), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------
# Networks
# ----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles):
        super(Critic, self).__init__()
        self.n_quantiles = n_quantiles
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_quantiles)
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_quantiles)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net1(sa), self.net2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net1(sa)


# ----------------------------
# Quantile Huber Loss
# ----------------------------
def quantile_huber_loss(predictions, targets, tau):
    predictions_expanded = predictions.unsqueeze(2)
    targets_expanded = targets.detach().unsqueeze(1)

    n_quantiles = predictions.shape[1]
    n_target_quantiles = targets.shape[1]

    predictions_broadcast = predictions_expanded.expand(-1, -1, n_target_quantiles)
    targets_broadcast = targets_expanded.expand(-1, n_quantiles, -1)

    huber_loss = F.smooth_l1_loss(predictions_broadcast, targets_broadcast, reduction='none')
    diff = targets_broadcast - predictions_broadcast
    tau_expanded = tau.unsqueeze(2).expand(-1, -1, n_target_quantiles)
    weight = torch.abs(tau_expanded - (diff < 0).float())

    return (weight * huber_loss).mean()


# ----------------------------
# Environment Factory
# ----------------------------
def make_env(env_id):
    env = gym.make(env_id)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    # 1. Apply Custom Logic FIRST
    env = CustomRewardWrapper(env)
    # 2. Log modified rewards
    env = RecordEpisodeStatistics(env)
    # 3. Normalize obs
    env = NormalizeObservation(env)
    # 4. Normalize rewards
    env = NormalizeReward(env, gamma=GAMMA)
    env = TransformReward(env, lambda r: np.clip(r, -10, 10))
    return env


# ----------------------------
# Training Loop
# ----------------------------
def train():
    env = make_env(ENV_NAME)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)

    critic = Critic(state_dim, action_dim, N_QUANTILES).to(DEVICE)
    critic_target = Critic(state_dim, action_dim, N_QUANTILES).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    best_score = -np.inf
    total_timesteps = 0

    # Lists to track scores
    recent_scores = deque(maxlen=20)
    all_scores = []  # <--- Added this to store all scores

    tau = (torch.arange(N_QUANTILES, device=DEVICE).float() + 0.5) / N_QUANTILES
    tau = tau.unsqueeze(0)

    print(f"Starting TQC training with Custom Reward Shaping on {DEVICE}...")

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        done = False

        while not done:
            total_timesteps += 1

            if total_timesteps < WARMUP:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = actor(state_tensor).cpu().numpy()[0]
                noise = np.random.normal(0, max_action * EXPLORE_NOISE, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, infos = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state

            # Collect scores
            if "episode" in infos:
                r = infos["episode"]["r"]
                # Handle numpy wrapping if present
                if isinstance(r, np.ndarray): r = r.item()
                recent_scores.append(r)
                all_scores.append(r)  # <--- Save to master list

            if total_timesteps >= WARMUP:
                state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(BATCH_SIZE)

                with torch.no_grad():
                    noise = (torch.randn_like(action_b) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                    next_action = (actor_target(next_state_b) + noise).clamp(-max_action, max_action)

                    q1_target, q2_target = critic_target(next_state_b, next_action)
                    q1_target_sorted, _ = torch.sort(q1_target, dim=1)
                    q2_target_sorted, _ = torch.sort(q2_target, dim=1)

                    n_keep = N_QUANTILES - N_DROP
                    target_q_trunc = torch.cat([q1_target_sorted[:, :n_keep], q2_target_sorted[:, :n_keep]], dim=1)
                    target_quantiles = reward_b + (1 - done_b) * GAMMA * target_q_trunc

                current_q1, current_q2 = critic(state_b, action_b)
                loss = quantile_huber_loss(current_q1, target_quantiles, tau) + \
                       quantile_huber_loss(current_q2, target_quantiles, tau)

                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()

                if total_timesteps % POLICY_FREQ == 0:
                    actor_loss = -critic.Q1(state_b, actor(state_b)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        if ep % 20 == 0 and len(recent_scores) > 0:
            avg_score = np.mean(recent_scores)
            print(f"Ep {ep:4d} | Timesteps: {total_timesteps:6d} | Avg Shaped Score: {avg_score:7.2f}")

            # Save scores
            np.save("tqc_scores.npy", np.array(all_scores))  # <--- Added save logic

            if avg_score > best_score:
                best_score = avg_score
                torch.save(actor.state_dict(), "tqc_actor_best.pth")
                torch.save(critic.state_dict(), "tqc_critic_best.pth")
                obs_rms = env.get_wrapper_attr('obs_rms')
                torch.save((obs_rms.mean, obs_rms.var), "tqc_norm_stats.pth")

    env.close()
    np.save("tqc_scores.npy", np.array(all_scores))  # Final save


if __name__ == "__main__":
    train()