import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def discount_rewards(rewards, gamma):
    discounted_r = torch.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.ones(action_space) * init_sigma)
        self.sigma_activation = F.softplus

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = torch.clamp(self.sigma_activation(self.sigma), min=1e-3, max=1.0)
        dist = Normal(action_mean, sigma)

        return dist

class Agent:
    def __init__(self, policy, device='cpu'):
        self.device = device
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.device)
        dist = self.policy(x)

        if evaluation:
            return dist.mean.detach().cpu().numpy(), None

        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum()

        return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.done.append(done)

    def update_policy(self, use_baseline, constant_baseline):
        action_log_probs = torch.stack(self.action_log_probs).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device)

        returns = discount_rewards(rewards, self.gamma)

        if use_baseline:
            advantage = returns - constant_baseline
        else:
            advantage = returns

        # Optional: normalize advantage to stabilize training
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        policy_loss = -(action_log_probs * advantage.detach()).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()

        return policy_loss.item(), returns.sum().item()
