# agent.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
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

        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.ones(self.action_space) * init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh(self.fc1_actor(x))
        x = self.tanh(self.fc2_actor(x))
        action_mean = self.fc3_actor_mean(x)
        sigma = self.sigma_activation(self.sigma)
        return Normal(action_mean, sigma)


class Agent:
    def __init__(self, policy, lr, entropy_coeff, gamma=0.99, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.entropy_coeff = entropy_coeff
        self.gamma = gamma

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.moving_avg_return = None  # NEW

    def update_policy(self, use_baseline, constant_baseline):
        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)

        returns = discount_rewards(rewards, self.gamma)

        # Update moving average return
        batch_return = returns.mean().item()
        if self.moving_avg_return is None:
            self.moving_avg_return = batch_return
        else:
            alpha = 0.05  # Smoothing factor
            self.moving_avg_return = alpha * batch_return + (1 - alpha) * self.moving_avg_return

        if use_baseline:
            if constant_baseline == 0.0:
                baseline = self.moving_avg_return
                normalize_advantage = True
            else:
                baseline = constant_baseline
                normalize_advantage = False
            

            advantage = returns - baseline
        else:
            advantage = returns.clone()
            normalize_advantage = True 
            
        if normalize_advantage :
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        normal_dists = [self.policy(s.to(self.train_device)) for s in self.states]
        entropy = torch.stack([dist.entropy().sum() for dist in normal_dists]).mean()

        policy_loss = -(action_log_probs * advantage.detach()).mean() - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()

        return policy_loss.item(), batch_return, entropy.item()

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy(x)

        if evaluation:
            return normal_dist.mean, None

        action = normal_dist.sample()
        action_log_prob = normal_dist.log_prob(action).sum()
        return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward]))
        self.done.append(done)
