import torch.nn as nn


class MultimodalNet(nn.Module):
    def __init__(self, q_dim, i_dim, common_embedding_size, noutput, dropout=0.5):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(q_dim, common_embedding_size), nn.Tanh())
        self.i_proj = nn.Sequential(nn.Linear(i_dim, common_embedding_size), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(common_embedding_size, noutput)

    def forward(self, q, i):
        q_proj = self.dropout(self.q_proj(q))
        i_proj = self.dropout(self.i_proj(i))
        return self.out(q_proj * i_proj)
