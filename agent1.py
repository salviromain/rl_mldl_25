import numpy as np
import torch
import wandb
import torch.nn.functional as F
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
        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.model=torch.nn.Sequential(
            torch.nn.Linear(self.state_space, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )


        self.model.apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    def forward(self, state):
        value=self.model(state)
        return value

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        return normal_dist

class Agent(object):
    def __init__(self, policy,critic, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self,I):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)

        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)

        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)

        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        done = torch.Tensor(self.done).to(self.train_device)

        self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #


        #
        # TASK 3:
        v_next = self.get_critic(next_states)
        v_next = (1 - done) * v_next  # zero-out terminal states
        v=self.get_critic(states)
        delta=rewards+self.gamma*v_next-v
        policy_loss=I*delta*action_log_probs
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        I = self.gamma*I
        wandb.log({"policy_loss":policy_loss.item()})
        #wandb.log({"Q_value":Q_value.item()})

        return delta, v, I

    def update_critic(self, delta, state):
        state=torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        self.state=[]
        v=self.get_critic(state)
        critic_loss=delta*v
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        wandb.log({"critic_loss":critic_loss.item()})
        self.critic_optimizer.step()


        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob
    def get_critic(self, state):
        x = state.float().to(self.train_device)

        value= self.critic(x)
        return value

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)