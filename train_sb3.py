import torch
import gym
import wandb
import itertools
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from env.custom_hopperUDR import *  # Ensure this is implemented correctly, this one is for UDR
#from env.custom_hopper import * # this for normal runs 
# === Logging callback for wandb ===
class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                episode_return = info["episode"]["r"]
                wandb.log({"episode_return": episode_return}, step=self.episode_count)
        return True

# === Environment setup ===
def make_env():
    #env = gym.make('CustomHopper-target-v0')
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)
    return env

# === Training loop ===
def train(config):
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        device=config["device"],
        learning_rate=config["lr"],
        gamma=config["gamma"],
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_model_{wandb.run.id}",
        log_path=f"./logs_{wandb.run.id}",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    wandb_cb = WandbLoggingCallback()

    model.learn(
        total_timesteps=config["timesteps"],
        callback=[eval_callback, wandb_cb]
    )

    # Save final model and weights
    model_path = f"ppo_model_{wandb.run.id}.zip"
    weights_path = f"ppo_weights_{wandb.run.id}.pt"

    model.save(model_path)
    torch.save(model.policy.state_dict(), weights_path)

    wandb.save(model_path)
    wandb.save(weights_path)

    print(f"Saved model to {model_path}")
    print(f"Saved weights to {weights_path}")

# === Manual grid search setup ===
if __name__ == "__main__":
    # Hyperparameter grid
    learning_rates = [0.0008]
    gammas = [0.99]
    devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    timesteps = 5_000_000

    # Cartesian product of all combinations
    for lr, gamma, device in itertools.product(learning_rates, gammas, devices):
        config = {
            "lr": lr,
            "gamma": gamma,
            "device": device,
            "timesteps": timesteps
        }

        # Start new wandb run
        run_name = f"lr_{lr}-source"

# Start new wandb run with that name
        wandb.init(
            project="hopper-rl-gridsearch",
            name=run_name,
            config=config,
            reinit=True,
        )

        train(config)
        wandb.finish()
