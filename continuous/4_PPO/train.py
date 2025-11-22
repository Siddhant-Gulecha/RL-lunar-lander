import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, NormalizeObservation, NormalizeReward, \
    TransformReward

# ----------------------------
# Config
# ----------------------------
ENV_NAME = "LunarLanderContinuous-v3"
MAX_TIMESTEPS = 3_000_000
NUM_ENVS = 8
N_STEPS = 2048
N_EPOCHS = 10
MINIBATCH_SIZE = 64
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01  # Keep this for exploration
CRITIC_COEF = 0.5
MAX_GRAD_NORM = 0.5
MAX_EPISODE_STEPS = 500  # Keep 500 to penalize "hovering"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# ----------------------------
# Actor-Critic Network
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()
        self.max_action = max_action

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),  # Changed back to ReLU for stability
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(256, 1)
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, state):
        return self.critic_head(self.shared_net(state))

    def get_action_and_value(self, state, action=None):
        x = self.shared_net(state)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std.expand_as(mean)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(1, keepdim=True)
        entropy = dist.entropy().sum(1, keepdim=True)
        value = self.critic_head(x)

        action_squashed = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action_squashed.pow(2) + 1e-6).sum(1, keepdim=True)

        return action_squashed * self.max_action, log_prob, entropy, value

    def get_deterministic_action(self, state):
        x = self.shared_net(state)
        mean = self.actor_mean(x)
        return torch.tanh(mean) * self.max_action


# ----------------------------
# Rollout Buffer
# ----------------------------
class RolloutBuffer:
    def __init__(self, n_steps, num_envs, state_dim, action_dim, device):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        self.states = torch.zeros((self.n_steps, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.n_steps, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        self.log_probs = torch.zeros((self.n_steps, self.num_envs, 1), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.n_steps, self.num_envs, 1), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros((self.n_steps, self.num_envs, 1), dtype=torch.float32).to(self.device)
        self.values = torch.zeros((self.n_steps, self.num_envs, 1), dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros((self.n_steps, self.num_envs, 1), dtype=torch.float32).to(self.device)
        self.returns = torch.zeros((self.n_steps, self.num_envs, 1), dtype=torch.float32).to(self.device)
        self.ptr = 0

    def push(self, state, action, log_prob, reward, done, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.n_steps

    def compute_gae_and_returns(self, last_value, last_done, gamma, gae_lambda):
        last_gae_lambda = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1.0 - next_done) - self.values[t]
            self.advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae_lambda
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        total_transitions = self.n_steps * self.num_envs
        indices = np.random.permutation(total_transitions)
        flat_states = self.states.reshape(-1, self.state_dim)
        flat_actions = self.actions.reshape(-1, self.action_dim)
        flat_log_probs = self.log_probs.reshape(-1, 1)
        flat_advantages = self.advantages.reshape(-1, 1)
        flat_returns = self.returns.reshape(-1, 1)

        # Normalize advantages (Crucial for PPO stability)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        for start in range(0, total_transitions, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield flat_states[batch_indices], flat_actions[batch_indices], flat_log_probs[batch_indices], \
            flat_advantages[batch_indices], flat_returns[batch_indices]


# ----------------------------
# Environment Factory
# ----------------------------
def make_env(env_id, seed, max_steps):
    def _init():
        env = gym.make(env_id)

        # 1. Enforce Time Limit
        env = TimeLimit(env, max_episode_steps=max_steps)

        # 2. Log REAL rewards (Must be before NormalizeReward)
        env = RecordEpisodeStatistics(env)

        # 3. Normalize Observations (Critical for learning)
        env = NormalizeObservation(env)

        # 4. Normalize Rewards (Helps critic converge, scales rewards)
        # We set gamma=GAMMA to correctly discount future rewards in the running mean
        env = NormalizeReward(env, gamma=GAMMA)

        # 5. Clip Rewards (Optional but good: prevents massive spikes)
        env = TransformReward(env, lambda r: np.clip(r, -10, 10))

        env.reset(seed=seed)
        return env

    return _init


def linear_schedule(optimizer, epoch, total_epochs, initial_lr):
    lr = initial_lr * (1.0 - (epoch / total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ----------------------------
# Training Loop
# ----------------------------
def train():
    print(f"Starting PPO training (Normalized) on {DEVICE}...")

    # Vectorized Env
    envs = gym.vector.AsyncVectorEnv(
        [make_env(ENV_NAME, seed + 1000, MAX_EPISODE_STEPS) for seed in range(NUM_ENVS)]
    )

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    max_action = float(envs.single_action_space.high[0])

    agent = ActorCritic(state_dim, action_dim, max_action).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    buffer = RolloutBuffer(N_STEPS, NUM_ENVS, state_dim, action_dim, DEVICE)

    total_timesteps = 0
    ep_rewards = []
    rollout_count = 0
    best_avg_score = -np.inf
    total_rollouts = MAX_TIMESTEPS // (N_STEPS * NUM_ENVS)

    state, _ = envs.reset()
    state = torch.tensor(state, dtype=torch.float32).to(DEVICE)

    while total_timesteps < MAX_TIMESTEPS:
        rollout_count += 1
        linear_schedule(optimizer, rollout_count, total_rollouts, LR)

        # --- Collection ---
        for step in range(N_STEPS):
            total_timesteps += NUM_ENVS
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(state)

            # Interaction
            next_state, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())

            # --- LOGGING REAL REWARD ---
            # The 'reward' variable is NOW the *normalized* reward.
            # The *real* reward is in the infos dict, thanks to RecordEpisodeStatistics.
            if "episode" in infos:
                for r in infos["episode"]["r"]:
                    ep_rewards.append(r)
                    if len(ep_rewards) % 20 == 0:
                        avg = np.mean(ep_rewards[-20:])
                        # This "Avg" is the TRUE, un-normalized score
                        print(f"T: {total_timesteps:7d} | Ep: {len(ep_rewards):5d} | Avg: {avg:7.2f}", flush=True)

            buffer.push(
                state, action, log_prob,
                # We store the NORMALIZED reward for training
                torch.tensor(reward, dtype=torch.float32).to(DEVICE).unsqueeze(1),
                torch.tensor(terminated, dtype=torch.float32).to(DEVICE).unsqueeze(1),
                value
            )
            state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)

        # --- GAE ---
        with torch.no_grad():
            last_value = agent.get_value(state)
            last_done = torch.tensor(np.logical_or(terminated, truncated), dtype=torch.float32).to(DEVICE).unsqueeze(1)
            buffer.compute_gae_and_returns(last_value, last_done, GAMMA, GAE_LAMBDA)

        # --- Update ---
        for _ in range(N_EPOCHS):
            for batch in buffer.get_batches(MINIBATCH_SIZE):
                b_s, b_a, b_log, b_adv, b_ret = batch
                _, new_log, entropy, new_val = agent.get_action_and_value(b_s, b_a)

                critic_loss = F.mse_loss(new_val, b_ret)

                log_ratio = new_log - b_log
                ratio = log_ratio.exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Note: We SUBTRACT entropy loss, as we want to *maximize* entropy
                loss = actor_loss + CRITIC_COEF * critic_loss - ENTROPY_COEF * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        buffer.reset()

        # --- Save Best ---
        if rollout_count % 5 == 0 and len(ep_rewards) > 50:
            # We use the REAL (un-normalized) scores for this check
            avg_reward = np.mean(ep_rewards[-50:])
            np.save("ppo_scores.npy", np.array(ep_rewards))

            if avg_reward > best_avg_score:
                best_avg_score = avg_reward
                print(f"--- NEW BEST! Saving model. Avg: {avg_reward:.2f} ---", flush=True)
                # We save the agent, which knows how to handle normalized obs
                torch.save(agent.state_dict(), "ppo_actor_critic_best.pth")
                # We must ALSO save the normalization stats!
                # `envs.get_attr("obs_rms")` gets the running mean/std from the wrappers
                norm_stats = (envs.get_attr("obs_rms")[0].mean, envs.get_attr("obs_rms")[0].var)
                torch.save(norm_stats, "ppo_norm_stats.pth")

    envs.close()
    print("Training complete.", flush=True)


if __name__ == "__main__":
    train()