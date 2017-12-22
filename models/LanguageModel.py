#!/bin/bash python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=300, num_rnn=3, rnn_layers=1, rnn_type='lstm', dropout=0.5):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        for param in self.embedding_layer.parameters():
            nn.init.normal(param)
        
        self.rnn_list = nn.ModuleList()
        for i in range(num_rnn):
            rnn_batchnorm_dropout = nn.ModuleList()
            # RNN
            prev_dim = hidden_size
            if i == 0:
                prev_dim = embed_size
            if rnn_type == 'lstm':
                rnn_batchnorm_dropout.append(nn.LSTM(prev_dim, hidden_size, rnn_layers))
            else:
                rnn_batchnorm_dropout.append(nn.GRU(prev_dim, hidden_size, rnn_layers))
            # BatchNorm
            rnn_batchnorm_dropout.append(nn.BatchNorm1d(hidden_size))
            rnn_batchnorm_dropout.append(nn.Dropout(p=dropout))
            self.rnn_list.append(rnn_batchnorm_dropout)

        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, padded_input, lengths):
        padded_embed_out = self.embedding_layer(padded_input)
        padded_embed_out = padded_embed_out.contiguous()
        packed_output = pack_padded_sequence(padded_embed_out, lengths, batch_first=True)

        # get through rnns
        for rnn_batchnorm_dropout in self.rnn_list:
            packed_output, _ = rnn_batchnorm_dropout[0](packed_output)
            padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)
            # change the size B x T x H into BT x H for batchnorm
            B = padded_output.shape[0]
            T = padded_output.shape[1]
            padded_output = padded_output.contiguous()
            padded_output = padded_output.view(B*T, self.hidden_size)
            padded_output = rnn_batchnorm_dropout[1](padded_output)
            padded_output = rnn_batchnorm_dropout[2](padded_output)
            # restore the size BT x H to B x T x H
            padded_output = padded_output.contiguous()
            padded_output = padded_output.view(B, T, self.hidden_size)
            packed_output = pack_padded_sequence(padded_output, lengths, batch_first=True)

        padded_rnns_out, lengths = pad_packed_sequence(packed_output, batch_first=True)
        padded_rnns_out = padded_rnns_out.contiguous()
        output = self.linear(padded_rnns_out)
        return output
