# project_v1/ai_modules/dqn_pilot.py
#sensör verisini alıp en iyi "manevrayı" seçer.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DuelingDQN(nn.Module):
    """
    Referans makaledeki yapiyi gelistirerek 'Dueling Architecture' kullaniyoruz.
    Bu, State Value (V) ve Advantage (A) degerlerini ayri ayri ogrenir,
    bu sayede F-16 gibi hassas kontrol gerektiren sistemlerde daha kararli calisir.
    """

    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()

        # Feature Extraction Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value Stream (Bu durumda ne kadar iyiyim?)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage Stream (Hangi aksiyonu almaliyim?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        qvals = values + (advantages - advantages.mean())
        return qvals


class F16Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.05
        self.batch_size = 128
        self.loss_fn = nn.SmoothL1Loss()
        self.train_steps = 0
        self.target_sync_interval = 200

    def select_action(self, state, is_training=True):
        if is_training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)

        # Double-DQN target:
        # - action selection from policy_net
        # - action evaluation from target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=2.0)
        self.optimizer.step()
        self.train_steps += 1

        if self.train_steps % self.target_sync_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
