import random
import torch
import numpy as np
from tqdm import tqdm

class EpsilonGreedy:
    def __init__(self, epsilon_start, epsilon_min, epsilon_decay, n_actions):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def choose_action(self, state, q_network, device):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1) # Exploration
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = q_network(state_t)
                return torch.argmax(q_values).item() # Exploitation

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_one_episode(env, q_network, optimizer, loss_fn, eg_policy, gamma, device):
    state, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = eg_policy.choose_action(state, q_network, device)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Calcul de la Target (Q-Learning)
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            max_next_q = q_network(next_state_t).max(1)[0]
            target = reward + gamma * max_next_q * (1 - done)
            
        # Prédiction actuelle
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        current_q = q_network(state_t)[0, action]
        
        # Optimisation
        loss = loss_fn(current_q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_reward += reward
        
    eg_policy.decay()
    return total_reward