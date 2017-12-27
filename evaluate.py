#!/bin/bash python
import torch
import argparse
import pickle
import models.LanguageModel
from data_helper import RNNDataset, my_collate_fn_cuda
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pickle', default='data/tiny-shakespeare.r')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_query', help="the file contains sentence to evaluate", default="query.txt")
    parser.add_argument("--checkpoint", help="the path to the checkpoint")
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--output_loss", default="loss.txt")
    args = parser.parse_args()
    
    # read vocab
    with open(args.input_pickle, 'rb') as f:
        json_data = pickle.load(f)
    token_to_idx = json_data['token_to_idx']
    del(json_data)
    vocab_size = len(token_to_idx)
    embed_size = args.embedding_size
    
    # load model
    model = models.LanguageModel.CharRNN(vocab_size, embed_size)
    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda()
    model.eval()
    query_list = []
    original_query_list = []
    with open(args.input_query) as f:
        line = f.readline()
        while line != "":
            query = []
            for word in line:
                if word in token_to_idx:
                    query.append(token_to_idx[word])
            query_list.append(query)
            original_query_list.append(line.strip())
            line = f.readline()
    
    print(query_list)
    query_iter = DataLoader(RNNDataset(query_list), 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            collate_fn=my_collate_fn_cuda)
    query_loss = []
    criterion = nn.CrossEntropyLoss(reduce=False)
    for batch in query_iter:
        padded_out = model(batch['input'], batch['lengths'])
        packed_out = pack_padded_sequence(padded_out, batch['lengths'], batch_first=True)
        packed_target = pack_padded_sequence(batch['output'], batch['lengths'], batch_first=True)
        loss = criterion(packed_out.data, packed_target.data)
        packed_loss = PackedSequence(loss, packed_out.batch_sizes)
        padded_loss, _ = pad_packed_sequence(packed_loss, batch_first=True, padding_value=0.0)
        batch_loss = np.sum(padded_loss.data.cpu().numpy(), axis=1)
        batch_loss /= batch['lengths']
        batch_loss = batch_loss[batch['reverse_sorted_index']]
        batch_loss = np.exp(-batch_loss)
        query_loss += batch_loss.tolist()
    print(original_query_list)
    for i in range(len(query_loss)):
        print("{}:{}".format(query_loss[i], original_query_list[i]))
    with open(args.output_loss, "w") as f:
        for loss in query_loss:
            f.write("{}\n".format(loss))
        
        