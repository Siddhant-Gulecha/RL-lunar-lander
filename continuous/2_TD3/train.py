from collections import deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from gymnasium.wrappers import TimeLimit

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
WARMUP = 5000  # More warmup for TD3
POLICY_NOISE = 0.2  # Noise added to target policy during critic update
NOISE_CLIP = 0.5  # Range to clip target policy noise
POLICY_FREQ = 2  # Frequency of delayed policy updates
EXPLORE_NOISE = 0.25  # Standard deviation of Gaussian exploration noise
MAX_EPISODE_STEPS = 400  # New config for time limit
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Critic 2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# ----------------------------
# Training Loop
# ----------------------------
def train():
    # Wrap the environment with a new time limit
    env = gym.make(ENV_NAME)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.AdamW(actor.parameters(), lr=ACTOR_LR)

    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.AdamW(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    training_scores = []
    best_score = -np.inf
    total_timesteps = 0

    print(f"Starting TD3 training on {DEVICE}...")

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            total_timesteps += 1

            if total_timesteps < WARMUP:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = actor(state_tensor).cpu().numpy()[0]
                # Gaussian exploration noise
                noise = np.random.normal(0, max_action * EXPLORE_NOISE, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            episode_reward += reward

            if total_timesteps >= WARMUP:
                # Sample replay buffer
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(BATCH_SIZE)

                with torch.no_grad():
                    # Target Policy Smoothing
                    noise = (torch.randn_like(action_batch) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                    next_action = (actor_target(next_state_batch) + noise).clamp(-max_action, max_action)

                    # Twin Critic Targets
                    target_Q1, target_Q2 = critic_target(next_state_batch, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward_batch + (1 - done_batch) * GAMMA * target_Q

                # Critic update
                current_Q1, current_Q2 = critic(state_batch, action_batch)
                critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Delayed Policy Updates
                if total_timesteps % POLICY_FREQ == 0:
                    actor_loss = -critic.Q1(state_batch, actor(state_batch)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Soft update targets
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        training_scores.append(episode_reward)

        if ep % 20 == 0:
            avg_score = np.mean(training_scores[-20:])
            print(f"Episode {ep:4d} | Total Timesteps: {total_timesteps:6d} | Avg Score: {avg_score:7.2f}")
            if avg_score > best_score:
                best_score = avg_score
                torch.save(actor.state_dict(), "td3_actor.pth")
                torch.save(critic.state_dict(), "td3_critic.pth")

    np.save("td3_scores.npy", np.array(training_scores))
    env.close()


if __name__ == "__main__":
    train()