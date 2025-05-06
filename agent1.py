import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributions import Normal

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Critic(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.state_space, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.model(state)

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
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)
        sigma = self.sigma_activation(self.sigma)
        return Normal(action_mean, sigma)

class Agent(object):
    def __init__(self, policy, critic, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-5)

        self.gamma = 0.99

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.entropies = []

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy(x)

        if evaluation:
            return normal_dist.mean, None, normal_dist
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob, normal_dist

    def store_outcome(self, state, next_state, action_log_prob, reward, done, entropy):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)
        self.entropies.append(entropy)

    def update_critic(self):
        state = torch.stack(self.states).to(self.train_device)
        next_state = torch.stack(self.next_states).to(self.train_device)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)
        done = torch.tensor(self.done).float().to(self.train_device)

        v = self.get_critic(state).squeeze(-1)
        v_next = self.get_critic(next_state).squeeze(-1)
        v_next = (1 - done) * v_next

        target = rewards + self.gamma * v_next
        critic_loss = F.mse_loss(v, target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        delta = (target - v).detach()
        wandb.log({"critic_loss": critic_loss.item()})
        return delta

    def update_policy(self, I, delta):
        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device)
        entropies = torch.stack(self.entropies).to(self.train_device)

        # Normalize advantages (delta)
        delta = (delta - delta.mean()) / (delta.std() + 1e-8)

        # Entropy regularization
        entropy_coeff = 0.01
        policy_loss = -(delta * action_log_probs).mean() - entropy_coeff * entropies.mean()


        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        wandb.log({"policy_loss": policy_loss.item()})

        # Clear stored episode data
        self.states.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.done.clear()
        self.action_log_probs.clear()
        self.entropies.clear()

        return self.gamma * I

    def get_critic(self, state):
        return self.critic(state.to(self.train_device)).squeeze(-1)
