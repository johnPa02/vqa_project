import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class QuestionEncoder(nn.Module):
    def __init__(self, embedding_size, lstm_size, num_layers, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(embedding_size, lstm_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

    def forward(self, embeddings, lengths):
        packed = rnn_utils.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        return hn[-1]
