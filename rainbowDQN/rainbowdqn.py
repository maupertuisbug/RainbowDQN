from torchrl.data import TensorDictPrioritizedReplayBuffer
import torch
from torchrl.data import LazyTensorStorage
from tensordict import TensorDict
import torch.optim as optim
import numpy as np
import random
from tensordict import TensorDict
from networks.duelingNetwork import DuelingNetwork
import pickle
from gym.wrappers import RecordVideo
import gc

def get_epsilon(start, final, total_steps, step):
    epsilon = start - (step / total_steps) * (start - final)
    return max(epsilon, final)


class RainbowDQN:
    def __init__(self, env, config, wandb, device):
        self.env = env
        self.replaybuffer = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=LazyTensorStorage(max_size=config.buffer_capacity, device="cpu"), priority_key="td_error", batch_size=config.batch_size)
        self.config = config
        self.wandb = wandb
        self.device = device

    def train(self, episodes):

        # Create the two Q-Nets
        noisyLayer = self.config.noisyLayer
        QNetA = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, noisyLayer, self.device).to(self.device)
        QNetA_target = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, noisyLayer, self.device).to(self.device)
        optimizerA = optim.Adam(QNetA.parameters(), lr=0.00025)

        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 1e-4
        training_batch_size = self.config.batch_size
        gamma = 0.99
        sync_freq = 10000
        steps = 0
        rewards_l = []
        ep = 0
        epl = 0
        inference_mode = torch.no_grad
        video_dir = "/video"
        env = RecordVideo(
                        self.env,
                        video_folder=video_dir,
                        episode_trigger=lambda ep: ep % 1000 == 0)

        while steps < self.config.steps:
            torch.cuda.empty_cache()
            gc.collect()
            state , _ = self.env.reset()
            state = np.array(state)
            total_reward = 0
            ep = ep+1
            epl = 0
            done = False
            losses = []
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            while not done:
                if steps < 1000000:
                    epsilon = get_epsilon(1.0, 0.01, 1000000, steps)
                epl+=1
                with inference_mode():
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.int64(QNetA.optimal_action(state).squeeze(0).item())
                    
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_state = torch.tensor(np.expand_dims(next_state, 0), device=self.device)
                    done =  torch.tensor(done, dtype=torch.float32, device=self.device)
                    error = reward + (1-done)*gamma*(QNetA_target(next_state, QNetA.optimal_action(next_state)))  - QNetA(state, action)
                    transition = TensorDict({
                        "obs": torch.tensor(state,device="cpu").unsqueeze(0),      
                        "action": torch.tensor([[action]], device="cpu"),
                        "reward": torch.tensor(reward, device="cpu", dtype=torch.float32).unsqueeze(0),
                        "done": torch.tensor(done, device="cpu").unsqueeze(0),
                        "next": torch.tensor(next_state, device="cpu").unsqueeze(0),
                        "td_error": torch.abs(torch.as_tensor(error, device="cpu")).view(1)
                    }, batch_size=[1])
                    self.replaybuffer.extend(transition) 
                        
            
                state = next_state
                steps += 1
                total_reward += reward

                if len(self.replaybuffer) > 50000 and steps % self.config.update_freq == 0:
                    for iter in range(0, self.config.epochs):
                        sample = self.replaybuffer.sample(training_batch_size)
                        states = sample["obs"].to(self.device).squeeze(1)
                        actions = sample["action"].to(self.device)
                        next_states = sample["next"].to(self.device).squeeze(1)
                        rewards = sample["reward"].to(self.device)
                        dones = sample["done"].to(self.device)
                        
                        q_values = QNetA(states, actions)
                        target_q_values = rewards + (1-dones)*gamma*QNetA_target(next_states, QNetA.optimal_action(next_states))
                        td_error = target_q_values - q_values
                        sample.set("td_error", td_error.abs().detach())
                        self.replaybuffer.update_tensordict_priority(sample)
                        loss_fn = torch.nn.MSELoss()
                        loss = loss_fn(q_values, target_q_values.detach())
                        losses.append(loss.detach().cpu().item())
                        optimizerA.zero_grad()
                        loss.backward()
                        optimizerA.step()

                    if steps%sync_freq == 0:
                        # let me do a soft update 
                        tau = 0.9
                        for target_param, param in zip(QNetA_target.parameters(), QNetA.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
           
            rewards_l.append(total_reward)
            if ep%5000 == 0:
                self.wandb.log({'total reward' : total_reward}, step = int(ep/5000))
                self.wandb.log({'average reward' : np.mean(rewards_l)}, step = int(ep/5000))
                self.wandb.log({"avg loss" : np.mean(losses)}, step = int(ep/5000))
                self.wandb.log({"steps" : steps}, step = int(ep/5000))
        
        del QNetA, QNetA_target
        del optimizerA, optimizerB
        gc.collect()
        torch.cuda.empty_cache()


    def save_buffer(self):
        file_name = self.env + str(self.config.random_seed) + "replay_buffer_state.pkl"
        with open(file_name, "wb") as f:
                pickle.dump(self.replaybuffer.state_dict(), f)




