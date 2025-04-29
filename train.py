"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=10000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	import wandb
	import time
	
	wandb.init(
	    project="reinforce-baseline",
	    config={
	        "use_baseline": True,
	        "baseline_type": "constant",         # <-- not "value_function"
	        "constant_baseline_value": 20.0,      # or 0.0, or use dynamic mean
	        "gamma": 0.99,
	        "algorithm": "REINFORCE"
	    }
	)


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
		policy_loss, episode_return = agent.update_policy(
				    use_baseline=True,
				    constant_baseline=20      # or the average return if you want
		)

		
		wandb.log({
		    "episode": episode,
		    "episode_return": episode_return,
		    "policy_loss": policy_loss,
		    "baseline_type": wandb.config.baseline_type,
		    "use_baseline": wandb.config.use_baseline,
		    "constant_baseline_value": wandb.config.constant_baseline_value if wandb.config.baseline_type == "constant" else None
		})


	torch.save(agent.policy.state_dict(), "modelBS20.mdl")

	

if __name__ == '__main__':
	main()
