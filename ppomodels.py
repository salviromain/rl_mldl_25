from stable_baselines3 import PPO
import torch

model = PPO.load("ppo_model_k59dwd5r.zip")
torch.save(model.policy.state_dict(), "ppo_model_k59dwd5r.pt")
