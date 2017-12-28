#!/bin/bash python
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pickle
import argparse
import os
import time
from tensorboardX import SummaryWriter
from data_helper import RNNDataset, my_collate_fn_cuda
import models.LanguageModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_pickle', default='data/tiny-shakespeare.data.pickle')
    parser.add_argument('--input_vocab_pickle', default='data/tiny-shakespeare.vocab.pickle')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rnn_type', default='lstm')
    parser.add_argument('--rnn_layers', type=int, default=3)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2727)
    parser.add_argument('--checkpoint_dir', default="checkpoint/")
    parser.add_argument('--checkpoint_prefix', default="charrnn")
    parser.add_argument('--tensorboardX', default="tensorboardX")
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    # set tensorboardX
    writer = SummaryWriter(args.tensorboardX)
    
    # load preprocessed data
    with open(args.input_data_pickle, 'rb') as f:
        json_data = pickle.load(f)
    
    train_iter = DataLoader(RNNDataset(json_data['train']), 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            collate_fn=my_collate_fn_cuda)
    val_iter = DataLoader(RNNDataset(json_data['val']), 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          collate_fn=my_collate_fn_cuda)
    test_iter = DataLoader(RNNDataset(json_data['test']), 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                           collate_fn=my_collate_fn_cuda)

    # initialize the model
    # load preprocessed data
    with open(args.input_vocab_pickle, 'rb') as f:
        json_data = pickle.load(f)

    vocab_size = len(json_data['idx_to_token'])
    embed_size = args.embedding_size
    print("vocab_size={}".format(vocab_size))
    print("embed_size={}".format(embed_size))
    model = models.LanguageModel.CharRNN(vocab_size, embed_size)
    model.cuda()

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoches):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        num_token = 0
        padded_out = ""
        for i, batch in enumerate(train_iter):
            padded_out = model(batch['input'], batch['lengths'])
            packed_out = pack_padded_sequence(padded_out, batch['lengths'], batch_first=True)
            packed_target = pack_padded_sequence(batch['output'], batch['lengths'], batch_first=True)
            optimizer.zero_grad()
            loss = criterion(packed_out.data, packed_target.data)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            num_token += np.sum(batch['lengths'])
        train_loss /= num_token
        print("[{}/{}] train_loss = {}".format(epoch+1, args.epoches, train_loss))
        #if epoch == 1:
        #    writer.add_graph(model, padded_out)
        writer.add_scalar(args.checkpoint_prefix+"/train_loss", train_loss, epoch)
        
        model.eval()
        val_loss = 0.0
        num_token = 0
        for batch in val_iter:
            padded_out = model(batch['input'], batch['lengths'])
            packed_out = pack_padded_sequence(padded_out, batch['lengths'], batch_first=True)
            packed_target = pack_padded_sequence(batch['output'], batch['lengths'], batch_first=True)
            loss = criterion(packed_out.data, packed_target.data)
            val_loss += loss.data[0]
            num_token += np.sum(batch['lengths'])
        val_loss /= num_token
        print("[{}/{}] val_loss = {}".format(epoch+1, args.epoches, val_loss))
        writer.add_scalar(args.checkpoint_prefix+"/val_loss", val_loss, epoch)
        
        test_loss = 0.0
        num_token = 0
        for batch in test_iter:
            padded_out = model(batch['input'], batch['lengths'])
            packed_out = pack_padded_sequence(padded_out, batch['lengths'], batch_first=True)
            packed_target = pack_padded_sequence(batch['output'], batch['lengths'], batch_first=True)
            loss = criterion(packed_out.data, packed_target.data)
            test_loss += loss.data[0]
            num_token += np.sum(batch['lengths'])
        test_loss /= num_token
        print("[{}/{}] test_loss = {}".format(epoch+1, args.epoches, test_loss))
        writer.add_scalar(args.checkpoint_prefix+"/test_loss", test_loss, epoch)

        # save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, args.checkpoint_prefix+'-'+str(epoch)))
        
        # profiler
        cost_time = time.time()-start_time
        print("[{}/{}] cost time = {} s".format(epoch+1, args.epoches, cost_time))
    
    # write summary to the disk
    writer.close()
