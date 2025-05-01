import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int)
    parser.add_argument('--batch-episodes', default=10, type=int)  # NEW
    parser.add_argument('--print-every', default=2000, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    return parser.parse_args()

args = parse_args()

def main():
    env = gym.make('CustomHopper-source-v0')

    wandb.init(project="reinforce-dual", config={"gamma": 0.99, "algorithm": "REINFORCE"})

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy_bs = Policy(obs_dim, act_dim)
    agent_bs = Agent(policy_bs, device=args.device)

    policy_nobs = Policy(obs_dim, act_dim)
    agent_nobs = Agent(policy_nobs, device=args.device)

    episode_counter = 0

    for episode in range(args.n_episodes):
        train_rew = {}

        for agent, label, use_bs, bs_val in [
            (agent_bs, 'NOBSchat', False, 30.0),
            # (agent_nobs, 'BS80', True, 80.0)
        ]:
            done = False
            train_reward = 0
            state = env.reset()
            steps = 0

            while not done:
                action, log_prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action.detach().cpu().numpy())
                agent.store_outcome(state, next_state, log_prob, reward, done)
                state = next_state
                train_reward += reward
                steps += 1

            train_rew[label] = train_reward
            episode_counter += 1

            # Only update every batch_episodes
            if episode_counter % args.batch_episodes == 0:
                policy_loss, episode_return, entropy = agent.update_policy(
                    use_baseline=use_bs,
                    constant_baseline=bs_val
                )
            else:
                policy_loss, episode_return, entropy = None, None, None  # Skip logging loss on non-update episodes

            # Logging every episode
            log_dict = {
                f"{label}/episode_return": train_reward,
                f"{label}/episode_steps": steps,
            }
            if policy_loss is not None:
                log_dict.update({
                    f"{label}/policy_loss": policy_loss,
                    f"{label}/use_baseline": use_bs,
                    f"{label}/constant_baseline_value": bs_val,
                    f"{label}/entropy": entropy,
                })
            wandb.log(log_dict, step=episode)

        if (episode + 1) % args.print_every == 0:
            print(f"[{episode + 1}] Returns - NOBSchat: {train_rew['NOBSchat']:.2f}")

    torch.save(agent_nobs.policy.state_dict(), "NOBSchat.mdl")

if __name__ == '__main__':
    main()
