import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, offset):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim + state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                        )
        # max value of actions
        self.action_bounds = action_bounds
        self.offset = offset
        
    def forward(self, state, goal):
        return (self.actor(torch.cat([state, goal], 1)) * self.action_bounds) + self.offset
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, H):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim + action_dim + state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )
        self.H = H
        
    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        return -self.critic(torch.cat([state, action, goal], 1)) * self.H
    
class DDPG:
    def __init__(self, state_dim, action_dim, action_bounds, offset, lr, H):
        
        self.actor = Actor(state_dim, action_dim, action_bounds, offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, H).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.mseLoss = torch.nn.MSELoss()


    def normalize_state(self, state, state_low=None, state_high=None, state_offset=None):
        if state_low is None:
            state_low = np.array([-1.5, 0, 0, 0]) if isinstance(state, np.ndarray) else torch.tensor([-1.5, 0, 0, 0])
        if state_high is None:
            state_high = np.array([1.5, 1.5, 1, 1]) if isinstance(state, np.ndarray) else torch.tensor([1.5, 1.5, 1, 1])
        if state_offset is None:
            state_offset = np.array([0, 0, 0, 0]) if isinstance(state, np.ndarray) else torch.tensor([0, 0, 0, 0])
        
        if isinstance(state, np.ndarray):
            state_low = np.asarray(state_low, dtype=state.dtype)
            state_high = np.asarray(state_high, dtype=state.dtype)
            state_offset = np.asarray(state_offset, dtype=state.dtype)
            return (state + state_offset) / (state_high - state_low)
        
        elif isinstance(state, torch.Tensor):
            state_low = state_low.to(dtype=state.dtype, device=state.device)
            state_high = state_high.to(dtype=state.dtype, device=state.device)
            state_offset = state_offset.to(dtype=state.dtype, device=state.device)
            return (state + state_offset) / (state_high - state_low)
        
        else:
            raise TypeError(f"Unsupported type for state. Must be either np.ndarray or torch.Tensor. Type received is {type(state)}.")


    
    def select_action(self, state, goal):
        state = torch.FloatTensor(self.normalize_state(state.reshape(1, -1))).to(device)
        goal = torch.FloatTensor(self.normalize_state(goal.reshape(1, -1))).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):
        for i in range(n_iter):
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)

            # Normalize states and goals
            state = torch.FloatTensor(self.normalize_state(state)).to(device)
            next_state = torch.FloatTensor(self.normalize_state(next_state)).to(device)
            goal = torch.FloatTensor(self.normalize_state(goal)).to(device)
            
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size, 1)).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # Select next action
            next_action = self.actor(next_state, goal).detach()

            # Compute target Q-value
            target_Q = self.critic(next_state, next_action, goal).detach()
            target_Q = reward + ((1 - done) * gamma * target_Q)

            # Optimize Critic
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute Actor Loss
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))  
        
        
        
        
      
        
        
