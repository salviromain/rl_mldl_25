import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int)
    parser.add_argument('--print-every', default=2000, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()

args = parse_args()

def main():
    env = gym.make('CustomHopper-source-v0')

    wandb.init(project="reinforce-dual", config={"gamma": 0.99, "algorithm": "REINFORCE"})

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # Agent with constant baseline 20.0
    policy_bs = Policy(obs_dim, act_dim)
    agent_bs = Agent(policy_bs, device=args.device)

    # Agent with NO baseline
    policy_nobs = Policy(obs_dim, act_dim)
    agent_nobs = Agent(policy_nobs, device=args.device)

    for episode in range(args.n_episodes):
        
        episode_returns={}
        for agent, label, use_bs, bs_val in [
            (agent_bs, 'BS20', True, 20.0),
            (agent_nobs, 'NOBS', False, 0.0)
        ]:
            done = False
            train_reward = 0
            state = env.reset()
            steps=0

            while not done:
                action, log_prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action.detach().cpu().numpy())
                agent.store_outcome(state, next_state, log_prob, reward, done)
                state = next_state
                train_reward += reward
                step+=1

            policy_loss, episode_return = agent.update_policy(
                use_baseline=use_bs,
                constant_baseline=bs_val
            )
            episode_returns[label] = episode_return

            wandb.log({
            f"{label}/episode_return": episode_return,
            f"{label}/policy_loss": policy_loss,
            f"{label}/use_baseline": use_bs,
            f"{label}/constant_baseline_value": bs_val,
            f"{label}/episode_steps": steps,
            }, step=episode)
        if (episode + 1) % args.print_every == 0:
            print(f"[{episode+1}] Returns - BS20: {episode_returns['BS20']:.2f} | NOBS: {episode_returns['NOBS']:.2f}")

    torch.save(agent_bs.policy.state_dict(), "modelBS20.mdl")
    torch.save(agent_nobs.policy.state_dict(), "modelNOBS.mdl")

if __name__ == '__main__':
    main()
