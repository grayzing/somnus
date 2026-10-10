import torch
from torch import nn

class Q(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=7218, hidden_size=512, batch_first=True)
        self.relu1 = nn.ReLU()
        
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.relu2 = nn.ReLU()
        
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.relu3 = nn.ReLU()

        self.q_output = nn.Linear(256, 12)

    def forward(self, x):
        # Handle 1D (single observation), 2D (batched flat), and 3D inputs
        if x.dim() == 1:
            # Shape: (7218,) -> (1, 1, 7218) [batch_size=1, seq_len=1, features=7218]
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            # Shape: (batch_size, 7218) -> (batch_size, 1, 7218)
            x = x.unsqueeze(1)
            
        out, _ = self.lstm1(x)
        out = self.relu1(out)
        
        out, _ = self.lstm2(out)
        out = self.relu2(out)
        
        out, _ = self.lstm3(out)
        out = self.relu3(out)
        
        # Take the features from the final time step in the sequence
        features = out[:, -1, :]
        
        value = self.q_output(features)
        return value

class MixingNetwork(nn.Module):
    def __init__(self, num_agents=19, state_dim=19 * 7218, embed_dim=32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetworks for QMIX
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents * embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim * 1)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, state=None):
        if state is None: # Fallback to VDN if no state is provided[cite: 1]
            return torch.sum(x, dim=1).reshape(x.shape[0], 1)
        
        batch_size = x.shape[0]
        agent_qs = x.view(batch_size, 1, self.num_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        
        hidden = nn.functional.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, 1)