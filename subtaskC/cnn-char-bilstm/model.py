import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import math

# initialize seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class CNNWordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, channel_dim1: int, channel_dim2: int, dropout=0.1):
        super(CNNWordEmbedding, self).__init__()
        char_embed_dim = int(math.ceil(np.log2(vocab_size)))
        self.vocab_size = vocab_size
        self.channel_dim1 = channel_dim1
        self.channel_dim2 = channel_dim2
        self.dropout = dropout
        self.char_embeds_layer = nn.Embedding(vocab_size, char_embed_dim)
        self.conv1 = nn.Sequential(
            nn.Conv1d(char_embed_dim, channel_dim1, 3),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        kernel_sizes = [3, 4, 5]
        self.convs2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channel_dim1, channel_dim2//len(kernel_sizes), kernel_size)
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.linear = nn.Sequential(
            nn.Linear(channel_dim2, channel_dim2),
            nn.ReLU()
        )

    def forward(self, words):
        # word should be in batch x sentence x word x dim
        char_embeds = self.char_embeds_layer(words).transpose(-2, -1) #change into dim x word
        batch, sent, emb_dim, word = char_embeds.shape
        conv1_out = self.conv1(char_embeds.view(-1, emb_dim, word)) #from emb_dim x word to channel_dim1 x emb_dim
        conv2_out = [conv_layer(conv1_out).max(dim=-1)[0].squeeze() for conv_layer in self.convs2]
        conv2_out = torch.cat(conv2_out, dim=-1)
        linear_out = self.linear(conv2_out)
        return (conv2_out + linear_out).view(batch, sent, -1)


class BiLSTMCNNWordEmbed(nn.Module):
    def __init__(self, vocab_size: int, lstm_hidden_dim: int, embedding_dim: int, initial_dim: int, target_size: int
                 , lstm_layers=2, dropout=0.25):
        super(BiLSTMCNNWordEmbed, self).__init__()
        self.word_embeddings_layer = CNNWordEmbedding(vocab_size, initial_dim, embedding_dim)
        self.bilstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = lstm_hidden_dim,
            num_layers = lstm_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = True
        )
        self.classifier = nn.Linear(2*lstm_hidden_dim, target_size)
    
    def forward(self, sentences):
        word_embeds = self.word_embeddings_layer(sentences)
        lstm_out, _ = self.bilstm(word_embeds)
        logits = self.classifier(lstm_out)
        return F.log_softmax(logits, dim=-1)
    
