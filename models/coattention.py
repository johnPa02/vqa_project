import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class CoattentionNet(nn.Module):
    def __init__(self, num_embeddings, num_classes, embed_dim=512):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.unigram_conv = nn.Conv1d(embed_dim, embed_dim, 1)
        self.bigram_conv = nn.Conv1d(embed_dim, embed_dim, 2, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=3, dropout=0.4)
        self.tanh = nn.Tanh()

        self.W_x = nn.Linear(embed_dim, embed_dim)
        self.W_g = nn.Linear(embed_dim, embed_dim)
        self.w_hx = nn.Linear(embed_dim, 1)

        self.W_w = nn.Linear(embed_dim, embed_dim)
        self.W_p = nn.Linear(embed_dim * 2, embed_dim)
        self.W_s = nn.Linear(embed_dim * 2, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, question, length, image):
        words = self.embed(question).permute(0, 2, 1)

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2)
        bigrams = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2)

        words = words.permute(0, 2, 1)
        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1)

        packed = nn.utils.rnn.pack_padded_sequence(phrase, length.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        sentence, _ = pad_packed_sequence(packed_output, batch_first=True)

        image = image.view(image.size(0), 512, -1)

        v_word, q_word = self.alternating_co_attention(words, image)
        v_phrase, q_phrase = self.alternating_co_attention(phrase, image)
        v_sent, q_sent = self.alternating_co_attention(sentence, image)

        h_w = self.tanh(self.W_w(q_word + v_word))
        h_p = self.tanh(self.W_p(torch.cat([q_phrase + v_phrase, h_w], dim=1)))
        h_s = self.tanh(self.W_s(torch.cat([q_sent + v_sent, h_p], dim=1)))

        return self.fc(h_s)

    def alternating_co_attention(self, Q, V):
        V = V.permute(0, 2, 1)
        H_q = self.tanh(self.W_x(Q))
        a_q = F.softmax(self.w_hx(H_q), dim=1)
        attended_q = torch.sum(a_q * Q, dim=1)

        H_v = self.tanh(self.W_x(V) + self.W_g(attended_q).unsqueeze(1))
        a_v = F.softmax(self.w_hx(H_v), dim=1)
        attended_v = torch.sum(a_v * V, dim=1)

        H_q_final = self.tanh(self.W_x(Q) + self.W_g(attended_v).unsqueeze(1))
        a_q_final = F.softmax(self.w_hx(H_q_final), dim=1)
        final_q = torch.sum(a_q_final * Q, dim=1)

        return attended_v, final_q
