"""Test an RL agent on the OpenAI Gym Hopper environment (SB3 version)"""
import argparse
import gym
from stable_baselines3 import PPO

from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Path to the SB3 PPO model (zip or pt)')
    parser.add_argument('--device', default='cpu', type=str, help='Device to run the model on [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the environment')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    return parser.parse_args()

args = parse_args()

def main():
    #env = gym.make('CustomHopper-source-v0')
    env = gym.make('CustomHopper-target-v0')
    list_values=[]
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    # Load PPO model using Stable Baselines3
    model = PPO.load(args.model, device=args.device)

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            if args.render:
                env.render()
            test_reward += reward
        liat_values.append(test_reward)

        print(f"Episode: {episode + 1} | Return: {test_reward}")
    arr = np.array(list_values)
    np.save('PPOST.npy', arr)


if __name__ == '__main__':
    main()
