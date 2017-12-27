# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os
import six
import numpy as np
import h5py
import codecs
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
    parser.add_argument('--output_data', default='data/tiny-shakespeare.data.pickle')
    parser.add_argument('--output_vocab', default='data/tiny-shakespeare.vocab.pickle')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--min_freq', type=int, default=3)
    args = parser.parse_args()
    if args.encoding == 'bytes': args.encoding = None
    
    # read original text 
    char_query_list = []
    with codecs.open(args.input_txt, 'r', args.encoding) as f:
        for line in f:
            if len(line) > 1:
                char_list = []
                for char in line.strip():
                    char_list.append(char)
                char_list.append("<EOS>")
                char_query_list.append(char_list)
    
    # Split data into train dev and test data
    total_size = len(char_query_list)
    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size
    np.random.seed(0)
    permutation = np.random.permutation(total_size)

    train_char_query_list = []
    val_char_query_list = []
    test_char_query_list = []
    for idx in permutation[:train_size]:
        train_char_query_list.append(char_query_list[idx])
    for idx in permutation[train_size:train_size+val_size]:
        val_char_query_list.append(char_query_list[idx])
    for idx in permutation[train_size+val_size:]:
        test_char_query_list.append(char_query_list[idx])
    
    # Build the vocab contains <EOS> and <UNK>
    token_to_freq = {}
    for char_list in train_char_query_list:
        for char in char_list:
            if char not in token_to_freq:
                token_to_freq[char] = 1
            else:
                token_to_freq[char] += 1
    
    # build true vocabulary remove low frequent words
    token_to_idx = {"<UNK>":0}
    idx_to_token = ["<UNK>"]
    for token in token_to_freq:
        if token_to_freq[token] > args.min_freq:
            token_to_idx[token] = len(idx_to_token)
            idx_to_token.append(token)
        else:
            print("low frequent word:{}".format(token))
    
    if not args.quiet:
        print('Total vocabulary size: %d' % len(token_to_idx))
        print('Total tokens in file: %d' % total_size)
        print('  Training size: %d' % train_size)
        print('  Val size: %d' % val_size)
        print('  Test size: %d' % test_size)
    
    train_query_list = []
    val_query_list = []
    test_query_list = []
    
    for char_list in train_char_query_list:
        idx_list = []
        for char in char_list:
            if char in token_to_idx:
                idx_list.append(token_to_idx[char])
            else:
                idx_list.append(token_to_idx['<UNK>'])
        train_query_list.append(idx_list)
    
    for char_list in val_char_query_list:
        idx_list = []
        for char in char_list:
            if char in token_to_idx:
                idx_list.append(token_to_idx[char])
            else:
                idx_list.append(token_to_idx['<UNK>'])
        val_query_list.append(idx_list)
    
    for char_list in test_char_query_list:
        idx_list = []
        for char in char_list:
            if char in token_to_idx:
                idx_list.append(token_to_idx[char])
            else:
                idx_list.append(token_to_idx['<UNK>'])
        test_query_list.append(idx_list)

    # Dump a JSON file for the vocab using json has expand the 
    json_data = {
        'train': train_query_list,
        'val': val_query_list,
        'test': test_query_list,
    }
    with open(args.output_data, 'wb') as f:
        pickle.dump(json_data, f, protocol=-1)
    
    json_data = {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
    }
    with open(args.output_vocab, 'wb') as f:
        pickle.dump(json_data, f, protocol=-1)