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



class RainbowDQN:
    def __init__(self, env, config, wandb, device):
        self.env = env
        self.replaybuffer = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=LazyTensorStorage(max_size=config.buffer_capacity), priority_key="td_error", batch_size=config.batch_size)
        self.config = config
        self.wandb = wandb
        self.device = device

    def train(self, episodes):

        # Create the two Q-Nets
        noisyLayer = self.config.noisyLayer
        QNetA = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, noisyLayer, self.device).to(self.device)
        QNetA_target = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, noisyLayer, self.device).to(self.device)
        QNetB = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, noisyLayer, self.device).to(self.device)
        QNetB_target = DuelingNetwork(self.env.observation_space.shape, self.env.action_space.n, noisyLayer, self.device).to(self.device)
        optimizerA = optim.Adam(QNetA.parameters(), lr=1e-4)
        optimizerB = optim.Adam(QNetB.parameters(), lr=1e-4)

        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay = 1e-6
        batch_size = self.config.batch_size
        gamma = 0.99
        sync_freq = 300
        steps = 0

        for ep in range(0,episodes):
            state , _ = self.env.reset()
            state = np.array(state)
            total_reward = 0
            done = False
            epsilon = max(epsilon_min, epsilon * (1 - epsilon_decay))

            while not done:
                state_v = torch.tensor(np.expand_dims(state, 0), device=self.device, dtype=torch.float32)
                if random.random() < 0.5:
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        q_values = QNetA(state_v)
                        action = int(torch.argmax(q_values, dim=1).item())
                   
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_state = np.array(next_state)
                    error = reward + QNetB_target(state_v, action)  - QNetB(state_v, action)
                    transition = TensorDict({
                        "obs": torch.tensor(state_v),     
                        "action": torch.tensor(action).unsqueeze(0),
                        "reward": torch.tensor(reward).unsqueeze(0),
                        "done": torch.tensor(done).unsqueeze(0),
                        "next": torch.tensor(next_state).unsqueeze(0),
                        "td_error": torch.as_tensor(error).view(1)
                    }, batch_size=[1])
                    self.replaybuffer.extend(transition) 
                else:
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        q_values = QNetB(state_v)
                        action = int(torch.argmax(q_values, dim=1).item())
                   
                    next_state, reward, done, _, _  = self.env.step(action)
                    next_state = np.array(next_state)
                    error = reward + QNetA_target(state_v, action)  - QNetA(state_v, action)
                    transition = TensorDict({
                        "obs": torch.tensor(state_v),    
                        "action": torch.tensor(action).unsqueeze(0),
                        "reward": torch.tensor(reward).unsqueeze(0),
                        "done": torch.tensor(done).unsqueeze(0),
                        "next": torch.tensor(next_state).unsqueeze(0),
                       "td_error": torch.as_tensor(error).view(1)
                    }, batch_size=[1])

                    self.replaybuffer.extend(transition)  
                        
            
                state = next_state
                total_reward += reward

                if len(self.replaybuffer) > 10000:
                    sample = self.replaybuffer.sample(batch_size)
                    states, actions, next_states, rewards, dones = zip(*sample)
                    if random.random() < 0.5:
                        q_values = QNetA(states, actions)
                        target_q_values = rewards + QNetA_target(next_states, QNetB.optimal_action(next_states))
                        td_error = target_q_values - q_values
                        sample.set("td_error", td_error.abs().detach())
                        self.replaybuffer.update_tensordict_priority(sample)
                        loss_fn = torch.nn.MSELoss()
                        loss = loss_fn(q_values, target_q_values.detach())
                        optimizerA.zero_grad()
                        loss.backward()
                        optimizerA.step()
                    else :
                        q_values = QNetB(states, actions)
                        target_q_values = rewards + QNetB_target(next_states, QNetA.optimal_action(next_states))
                        td_error = target_q_values - q_values
                        sample.set("td_error", td_error.abs().detach())
                        self.replaybuffer.update_tensordict_priority(sample)
                        loss_fn = torch.nn.MSELoss()
                        loss = loss_fn(q_values, target_q_values.detach())
                        optimizerB.zero_grad()
                        loss.backward()
                        optimizerB.step()

                if ep%sync_freq == 0:
                    # let me do a soft update 
                    tau = 0.005
                    for target_param, param in zip(QNetA_target.parameters(), QNetA.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for target_param, param in zip(QNetB_target.parameters(), QNetB.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.wandb.log({'total reward' : total_reward})
            self.wandb.log({'average reward' : np.mean(total_reward)})


    def save_buffer(self):
        file_name = self.env + str(self.config.random_seed) + "replay_buffer_state.pkl"
        with open(file_name, "wb") as f:
                pickle.dump(self.replaybuffer.state_dict(), f)




