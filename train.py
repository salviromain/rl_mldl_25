# train.py
import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy
import wandb
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=3000, type=int)
    parser.add_argument('--batch-episodes', default=10, type=int)
    parser.add_argument('--print-every', default=500, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    return parser.parse_args()

args = parse_args()


def main():
    env = gym.make('CustomHopper-source-v0')
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # Grid of hyperparams
    lr_grid = [1e-3, 5e-4, 1e-4]
    entropy_grid = [0.0, 0.01, 0.02]

    combinations = list(itertools.product(lr_grid, entropy_grid))

    for lr, entropy_coeff in combinations:
        wandb.init(
            project="reinforce-grid-search",
            name=f"nobs-lr{lr}-ent{entropy_coeff}",
            config={"lr": lr, "entropy_coeff": entropy_coeff, "algorithm": "REINFORCE", "baseline": False},
        )

        policy = Policy(obs_dim, act_dim)
        agent = Agent(policy, lr=lr, entropy_coeff=entropy_coeff, device=args.device)

        episode_counter = 0

        for episode in range(args.n_episodes):
            done = False
            train_reward = 0
            state = env.reset()
            steps = 0

            while not done:
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
                agent.store_outcome(state, next_state, log_prob, reward, done)
                state = next_state
                train_reward += reward
                steps += 1

            episode_counter += 1

            if episode_counter % args.batch_episodes == 0:
                policy_loss, episode_return, entropy = agent.update_policy(use_baseline=False)
            else:
                policy_loss, episode_return, entropy = None, None, None

            log_dict = {
                "episode_return": train_reward,
                "episode_steps": steps,
            }
            if policy_loss is not None:
                log_dict.update({
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                })
            wandb.log(log_dict, step=episode)

            if (episode + 1) % args.print_every == 0:
                print(f"[{episode + 1}] LR={lr}, Ent={entropy_coeff} | Return: {train_reward:.2f}")

        torch.save(agent.policy.state_dict(), f"nobs-lr{lr}-ent{entropy_coeff}.mdl")
        wandb.finish()


if __name__ == '__main__':
    main()
