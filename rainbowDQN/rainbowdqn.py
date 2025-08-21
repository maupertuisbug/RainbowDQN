from torchrl.data import TensorDictPrioritizedReplayBuffer
import torch
from torchrl.data import LazyTensorStorage
from tensordict import TensorDict
import torch.optim as optim
import numpy as np
import random
from tensordict import TensorDict
from networks.duelingNetwork import DuelingNetwork
from networks.network  import Network
import pickle
from gym.wrappers import RecordVideo
import gc
import os

def get_epsilon(start, final, total_steps, step):
    epsilon = start - (step / total_steps) * (start - final)
    return max(epsilon, final)


class RainbowDQN:
    def __init__(self, env, config, wandb, device):
        
        # initialize logging and device
        self.config = config
        self.wandb = wandb
        self.device = device
        
        # initialize the environment
        self.env = env
        self.replaybuffer = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=LazyTensorStorage(max_size=config.buffer_capacity, device="cpu"), priority_key="td_error", batch_size=config.batch_size)
       
        # initialize the networks and optimizers
        if self.config.dueling:
            self.QNetA = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, self.device, noisylayer = self.config.noisyLayer).to(self.device)
            self.QNetA_target = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, self.device, noisylayer = self.config.noisyLayer).to(self.device)
        else :
            self.QNetA = Network(self.env.observation_space.shape, self.env.action_space.n, self.device, noisyLayer = self.config.noisyLayer).to(self.device)
            self.QNetA_target = Network(self.env.observation_space.shape, self.env.action_space.n, self.device, noisyLayer = self.config.noisyLayer).to(self.device)
        self.optimizerA = optim.Adam(self.QNetA.parameters(), lr=self.config.learning_rate)

    def train(self, episodes):
        
        # Initialize the training parameters
        epsilon = self.config.epsilon
        epsilon_min = self.config.epsilon_min
        training_batch_size = self.config.batch_size
        gamma = self.config.gamma
        sync_freq = self.config.sync_freq

        # Initialize iterators
        steps = 0
        ep = 0
        epl = 0
        inference_mode = torch.no_grad


        # Initialize storage
        # video_dir = "/video"
        # env = RecordVideo(
        #                 self.env,
        #                 video_folder=video_dir,
        #                 episode_trigger=lambda ep: ep % 1000 == 0)
        losses = [0]
        rewards_l = []

        # Checkpoint directory
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        use_checkpoint = self.config.use_checkpoint
       
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir) and use_checkpoint:
            target_model_path = os.path.join(checkpoint_dir, "QNetA_target.pth")
            if os.path.isfile(target_model_path):
                self.QNetA_target.load_state_dict(torch.load(target_model_path))
                self.QNetA_target.to(self.device)
            target_model_path = os.path.join(checkpoint_dir, "QNetA.pth")
            if os.path.isfile(target_model_path):
                self.QNetA.load_state_dict(torch.load(target_model_path))
                self.QNetA.to(self.device)
            
        while steps < self.config.steps:
            # Initialize agent parameters
            state , _ = self.env.reset()
            state = np.array(state)
            done = False
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            
            # Initialize iterators
            total_reward = 0
            ep = ep+1
            epl = 0
            print("Episode: ", ep)
            while not done and epl < self.config.max_steps:
                if steps < self.config.exploration_steps:
                    epsilon = get_epsilon(epsilon, epsilon_min, self.config.exploration_steps, steps)
                
                epl+=1
                # inference
                with inference_mode():
                    self.QNetA.setEvaluationMode(eval=True)
                    self.QNetA_target.setEvaluationMode(eval=True)
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.int64(self.QNetA.optimal_action(state).squeeze(0).item())
                    
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_state = torch.tensor(np.expand_dims(next_state, 0), device=self.device)
                    done =  torch.tensor(done, dtype=torch.float32, device=self.device)
                    error = reward + (1-done)*gamma*(self.QNetA_target(next_state, self.QNetA.optimal_action(next_state)))  - self.QNetA(state, action)
                    
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


                # training 
                if len(self.replaybuffer) > self.config.initial_steps and steps % self.config.update_freq == 0:
                    
                    self.QNetA.setEvaluationMode(eval=False)
                    self.QNetA_target.setEvaluationMode(eval=False)
                    for iter in range(0, self.config.epochs):
                        sample = self.replaybuffer.sample(training_batch_size)
                        states = sample["obs"].to(self.device).squeeze(1)
                        actions = sample["action"].to(self.device)
                        next_states = sample["next"].to(self.device).squeeze(1)
                        rewards = sample["reward"].to(self.device)
                        dones = sample["done"].to(self.device)
                        
                        q_values = self.QNetA(states, actions).squeeze(1)
                        target_q_values = rewards
                        target_q_values = target_q_values + (1-dones)*gamma*(self.QNetA_target(next_states, self.QNetA.optimal_action(next_states)).squeeze(1))
                        td_error = target_q_values - q_values
                        sample.set("td_error", td_error.abs().detach())
                        self.replaybuffer.update_tensordict_priority(sample)
                        loss_fn = torch.nn.MSELoss()
                        loss = loss_fn(q_values, target_q_values.detach())
                        losses.append(loss.detach().cpu().item())
                        self.optimizerA.zero_grad()
                        loss.backward()
                        self.optimizerA.step()
                    

                    # update the target network
                    if steps%sync_freq == 0:
                        # let me do a soft update 
                        tau = 0.9
                        for target_param, param in zip(self.QNetA_target.parameters(), self.QNetA.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        torch.save(self.QNetA_target.state_dict(), os.path.join(checkpoint_dir, "QNetA_target.pth"))
                        torch.save(self.QNetA.state_dict(), os.path.join(checkpoint_dir, "QNetA.pth"))
           
            rewards_l.append(total_reward)
            
            # logging
            if ep%2 == 0:
                self.wandb.log({'total reward' : total_reward}, step = int(ep/2))
                self.wandb.log({'average reward' : np.mean(rewards_l)}, step = int(ep/2))
                self.wandb.log({"avg loss" : np.mean(losses)}, step = int(ep/2))
                self.wandb.log({"steps" : steps}, step = int(ep/2))
        
        # free memory
        del self.QNetA, self.QNetA_target
        del self.optimizerA
        gc.collect()
        torch.cuda.empty_cache()


    def _save_buffer(self):
        file_name = self.env + str(self.config.random_seed) + "replay_buffer_state.pkl"
        with open(file_name, "wb") as f:
                pickle.dump(self.replaybuffer.state_dict(), f)




