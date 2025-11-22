import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLanderContinuous-v3"
EPISODES = 2000
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 64
REPLAY_SIZE = 100000
WARMUP = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Ornstein-Uhlenbeck Noise
# ----------------------------
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


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
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)


# ----------------------------
# Training Loop
# ----------------------------
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)

    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    noise = OUNoise(action_dim)

    training_scores = []
    best_score = -np.inf

    print(f"Starting DDPG training on {DEVICE}...")

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        noise.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = actor(state_tensor).cpu().numpy()[0]

            # Add noise for exploration
            action = (action + noise.sample()).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > WARMUP:
                # Sample from replay buffer
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(BATCH_SIZE)

                # Critic update
                with torch.no_grad():
                    target_action = actor_target(next_state_batch)
                    target_q = critic_target(next_state_batch, target_action)
                    target_value = reward_batch + (1 - done_batch) * GAMMA * target_q

                current_q = critic(state_batch, action_batch)
                critic_loss = nn.MSELoss()(current_q, target_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update
                actor_loss = -critic(state_batch, actor(state_batch)).mean()
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
            print(f"Episode {ep}: Avg Score = {avg_score:.2f}")
            if avg_score > best_score:
                best_score = avg_score
                torch.save(actor.state_dict(), "ddpg_actor.pth")
                torch.save(critic.state_dict(), "ddpg_critic.pth")

    np.save("ddpg_scores.npy", np.array(training_scores))
    env.close()


if __name__ == "__main__":
    train()