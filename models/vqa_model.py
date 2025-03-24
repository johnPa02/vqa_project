import torch.nn as nn
from embedding import QuestionEmbedding
from encoder import QuestionEncoder
from multimodal import MultimodalNet

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_size, num_layers,
                 image_feat_dim, common_embedding_size, noutput, dropout=0.5):
        super().__init__()
        self.embedding_net = QuestionEmbedding(vocab_size, embedding_size, dropout)
        self.encoder = QuestionEncoder(embedding_size, lstm_size, num_layers, dropout)
        self.multimodal = MultimodalNet(lstm_size, image_feat_dim, common_embedding_size, noutput, dropout)

    def forward(self, question, lengths, image):
        embeddings = self.embedding_net(question)
        q_encoded = self.encoder(embeddings, lengths)
        return self.multimodal(q_encoded, image)
