# train.py
import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int)
    parser.add_argument('--batch-episodes', default=10, type=int)
    parser.add_argument('--print-every', default=2000, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    return parser.parse_args()


args = parse_args()

def main():
    env = gym.make('CustomHopper-source-v0')
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    agents = [
        #{
            #"label": "NoBaseline",
           # "agent": Agent(Policy(obs_dim, act_dim), lr=1e-3, entropy_coeff=0.01, device=args.device),
           # "use_baseline": False,
           # "baseline_val": 0.0
        #},
        {
           "label": "BaselineMean",
           "agent": Agent(Policy(obs_dim, act_dim), lr=1e-4, entropy_coeff=0.01, device=args.device),
           "use_baseline": True,
           "baseline_val": 0.0
        },
        #{
         #  "label": "Baseline20",
         #   "agent": Agent(Policy(obs_dim, act_dim), lr=1e-3, entropy_coeff=0.01, device=args.device),
          #  "use_baseline": True,
           # "baseline_val": 20.0
       # }
    ]

    for config in agents:
        wandb.init(
            project="reinforce-multirun",
            name=f"{config['label']}",
            config={"baseline": config['use_baseline'], "baseline_val": config['baseline_val'], "entropy_coeff": config['agent'].entropy_coeff},
        )

        episode_counter = 0

        for episode in range(args.n_episodes):
            agent = config['agent']
            label = config['label']
            use_bs = config['use_baseline']
            bs_val = config['baseline_val']

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
                policy_loss, episode_return, entropy = agent.update_policy(use_baseline=use_bs, constant_baseline=bs_val)
            else:
                policy_loss, episode_return, entropy = None, None, None

            log_dict = {
                f"{label}/episode_return": train_reward,
                f"{label}/episode_steps": steps,
            }
            if policy_loss is not None:
                log_dict.update({
                    f"{label}/policy_loss": policy_loss,
                    f"{label}/entropy": entropy,
                })

            wandb.log(log_dict, step=episode)

            if (episode + 1) % args.print_every == 0:
                print(f"[{episode + 1}] {label} | Return: {train_reward:.2f}")

        torch.save(agent.policy.state_dict(), f"{label}.mdl")
        wandb.finish()


if __name__ == '__main__':
    main()
