import torch 
import torch.nn as nn 
import wandb
import argparse
from   omegaconf import OmegaConf
import pickle 
import gymnasium as gym
from rainbowDQN.rainbowdqn import RainbowDQN
import ale_py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv, ClipRewardEnv

gym.register_envs(ale_py)
os.environ["WANDB_MODE"] = "online"


def run_exp():
    wandb_run = wandb.init(project="rdqn_exp")
    config = wandb.config 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used ", device)
    env_name = config.env
    env = gym.make(env_name)
    env = NoopResetEnv(env, noop_max=42)
    # env = MaxAndSkipEnv(env, skip=4)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    

    agent = RainbowDQN(env, config, wandb, device)
    agent.train(config.episodes)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    project_name = "rdqn_exp"
    sweep_id   = wandb.sweep(sweep=config_dict, project=project_name)
    agent      = wandb.agent(sweep_id, function=run_exp, count = 10)



    
    