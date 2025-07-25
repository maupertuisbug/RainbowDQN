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
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

gym.register_envs(ale_py)


def run_exp():
    wandb_run = wandb.init(project="RDQN")
    config = wandb.config 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used ", device)
    env_name = config.env
    env = gym.make(env_name, frameskip=4)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = MaxAndSkipEnv(env, skip=4)

    agent = RainbowDQN(env, config, wandb, device)
    agent.train(config.episodes)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    project_name = "RDQN"
    sweep_id   = wandb.sweep(sweep=config_dict, project=project_name)
    agent      = wandb.agent(sweep_id, function=run_exp, count = 10)



    
    