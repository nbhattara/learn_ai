import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

 #game environment
GRID_SIZE = 5
GOAL = (4, 4)

def reset():
    return (0, 0)

def step(state, action):
    r, c = state

    if action == 0 and r > 0: r -= 1       # up
    if action == 1 and r < GRID_SIZE-1: r += 1  # down
    if action == 2 and c > 0: c -= 1       # left
    if action == 3 and c < GRID_SIZE-1: c += 1  # right

    next_state = (r, c)

    if next_state == GOAL:
        return next_state, 10, True
    else:
        return next_state, -1, False
    
    #neural network for DQN
    
    class DQN(nn.Module):
     def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
     #replay memory

    def forward(self, x):
        return self.fc(x)
    memory = deque(maxlen=2000)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

#training setup
    device = torch.device("cpu")

model = DQN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64

# Experience replay
def replay():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    for state, action, reward, next_state, done in batch:
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)

        target = reward
        if not done:
            target += gamma * torch.max(model(next_state)).item()

        target_f = model(state)
        target_f[action] = target

        loss = loss_fn(model(state), target_f)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Training loop

        episodes = 500

for episode in range(episodes):
    state = reset()
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                action = torch.argmax(
                    model(torch.FloatTensor(state))
                ).item()

        next_state, reward, done = step(state, action)
        remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        replay()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode+1}, Reward: {total_reward}")




