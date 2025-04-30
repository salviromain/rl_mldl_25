"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym
import wandb

from env.custom_hopper import *
from agent1 import Agent, Policy, Critic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=5000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

wandb.init(project="hopper-rl", config={
    "episodes": args.n_episodes,
    "print_every": args.print_every,
    "device": args.device,
    "lr_policy": 1e-3,
    "lr_critic": 1e-3,
    "gamma": 0.99
})

def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	critic = Critic(observation_space_dim)
	agent = Agent(policy, critic, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		done = False

		train_reward = 0
		
		state = env.reset()  # Reset the environment and observe the initial state
		
		I=1
		
		while not done:  # Loop until the episode is over
		
			action, action_log_prob=agent.get_action(state)
		
			previous_state = state
		
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_log_prob, reward, done)

			train_reward += reward

			delta, I= agent.update_policy(I)

			agent.update_critic(delta)


		wandb.log({"episode":episode,"return":train_reward})
		
		if (episode+1)%args.print_every == 0:

			print('Training episode:', episode)

			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model_new_pseudocode.mdl")
	wandb.save("model_critic_modified")
	wandb.finish()
	

if __name__ == '__main__':
	main()