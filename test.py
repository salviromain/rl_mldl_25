import argparse
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--save_path', default='test_rewards_plot.png', type=str, help='Path to save reward plot')
    return parser.parse_args()

args = parse_args()

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model), strict=True)

    agent = Agent(policy, lr=0.0, entropy_coeff=0.0, device=args.device)

    rewards = []

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            if args.render:
                env.render()
            test_reward += reward

        print(f"Episode: {episode + 1} | Return: {test_reward}")
        rewards.append(test_reward)

    # Plot rewards
    episodes = np.arange(1, args.episodes + 1)
    rewards = np.array(rewards)
    mean = rewards.mean()
    std = rewards.std()
    np.save("REINFORCE2.npy",rewards)
    print("SAVED")

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Episode Return')
    plt.axhline(mean, color='r', linestyle='--', label=f'Mean = {mean:.2f}')
    plt.fill_between(episodes, mean - std, mean + std, color='r', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Test Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot
    #plt.savefig(args.save_path)
    print(f"Saved reward plot to: {args.save_path}")
    plt.close()

if __name__ == '__main__':
    main()
