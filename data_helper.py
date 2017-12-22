#!/bin/bash python
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

class RNNDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return {"input": self.data_list[idx][:-1], "output":self.data_list[idx][1:]}

def my_collate_fn_cuda(x):
    lengths = np.array([len(term['input']) for term in x])
    sorted_index = np.argsort(-lengths)

    # build reverse index map to reconstruct the original order
    reverse_sorted_index = np.zeros(len(sorted_index), dtype=int)
    for i, j in enumerate(sorted_index):
        reverse_sorted_index[j]=i
    lengths = lengths[sorted_index]
    # control the maximum length of LSTM
    max_len = lengths[0]
    batch_size = len(x)
    input_tensor = torch.LongTensor(batch_size, int(max_len)).zero_()
    output_tensor = torch.LongTensor(batch_size, int(max_len)).zero_()

    for i, index in enumerate(sorted_index):
        input_tensor[i][:lengths[i]] = torch.LongTensor(x[index]['input'])
        output_tensor[i][:lengths[i]] = torch.LongTensor(x[index]['output'])

    packed_input = Variable(input_tensor).cuda()
    packed_output = Variable(output_tensor).cuda()
    return {'input':packed_input, 'output':packed_output, 'lengths':lengths, 'reverse_sorted_index':reverse_sorted_index}

