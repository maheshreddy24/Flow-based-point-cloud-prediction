import torch
import torch.nn as nn


import torch
import torch.nn as nn
import math

# class Block(nn.Module):
#     def __init__(self, channels=512):
#         super().__init__()
#         self.ff = nn.Linear(channels, channels)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         return self.act(self.ff(x))

# class PointCloudFlowNetwork(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=512, layers=5, time_dim=512):
#         super(PointCloudFlowNetwork, self).__init__()
        
#         self.time_dim = time_dim
#         self.in_projection = nn.Linear(input_dim, hidden_dim)
#         self.t_projection = nn.Linear(time_dim, hidden_dim)
        
#         self.blocks = nn.Sequential(*[Block(hidden_dim) for _ in range(layers)])
#         self.out_projection = nn.Linear(hidden_dim, input_dim)

#     def gen_t_embedding(self, t, max_positions=10000):
#         t = t * max_positions
#         half_dim = self.time_dim // 2
#         emb = math.log(max_positions) / (half_dim - 1)
#         emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat([emb.sin(), emb.cos()], dim=1)
#         if self.time_dim % 2 == 1:
#             emb = nn.functional.pad(emb, (0, 1), mode='constant')
#         return emb

#     def forward(self, x, t):
#         batch_size, num_points, _ = x.shape

#         x = self.in_projection(x)

#         t = self.gen_t_embedding(t)
#         t = self.t_projection(t).unsqueeze(1).expand(-1, num_points, -1)

#         x = x + t
#         x = self.blocks(x)
#         x = self.out_projection(x)

#         return x

class PointCloudFlowNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(PointCloudFlowNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*4), nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2), nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Predict (dx, dy, dz) velocity
        )

    def forward(self, x, t = 0):
        return self.model(x)  # Predict velocity for each point