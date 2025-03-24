import torch.nn as nn

class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return self.tanh(x)
