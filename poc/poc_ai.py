# https://github.com/maurock/snake-ga/blob/master/DQN.py

import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from poc.poc_getState import *

DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.last_score = 0
        self.reward = 0
        self.gamma = 0.9 
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()

    def network(self):
    # Layers
        self.f1 = nn.Linear(8, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")
    
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, player, enemies, n):
        """
        Return the state.
        The state is a numpy array of n*2 values, representing:
            - x,y-Position of the agent/player
            - x,y-position of the closest food
            - x,y-position of the second closest food
        """
        state = np.concatenate(
            (
            np.array([
                player.rect.centerx,
                player.rect.centery]
                ),
                getStateNearest(player, enemies, n))
            )

        return np.asarray(state)
    
    def set_reward(self, game):
        # If the player increased the score -> get a reward
        self.reward = 0
        if game.score > self.last_score:
            self.reward = game.score - self.last_score
            self.last_score = game.score
        return self.reward

    def remember(self, state, action, reward, next_state):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state))
    
    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:  
            minibatch = memory
        for state, action, reward, next_state in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 8)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 8)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()