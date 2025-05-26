import torch
import gym
import wandb
import itertools
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
# Make sure your env registration file is imported somewhere before this script runs!
from env.custom_hopper import CustomHopper
# or if you want UDR variant:
# from env.custom_hopperUDR import CustomHopperUDR

# === Logging callback for wandb ===
class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                episode_return = info["episode"]["r"]
                wandb.log({"episode_return": episode_return}, step=self.episode_count)
        return True


# === Environment setup ===
def make_env(env_id='CustomHopper-source-v0'):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        return env
    return _init


# === Training loop ===
def train(config):
    train_env = DummyVecEnv([make_env(env_id='CustomHopper-source-v0')])
    eval_env = DummyVecEnv([make_env(env_id='CustomHopper-source-v0')])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
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
        render=False,
    )

    wandb_cb = WandbLoggingCallback()

    model.learn(
        total_timesteps=config["timesteps"],
        callback=[eval_callback, wandb_cb]
    )

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
    learning_rates = [0.0008]
    gammas = [0.99]
    devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    timesteps = 1_000_000

    for lr, gamma, device in itertools.product(learning_rates, gammas, devices):
        config = {
            "lr": lr,
            "gamma": gamma,
            "device": device,
            "timesteps": timesteps
        }

        run_name = f"lr_{lr}-source"

        wandb.init(
            project="hopper-rl-gridsearch",
            name=run_name,
            config=config,
            reinit=True,
        )

        train(config)
        wandb.finish()
