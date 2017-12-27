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
    parser.add_argument('--output', default='data/tiny-shakespeare.r')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--encoding', default='utf-8')
    args = parser.parse_args()
    if args.encoding == 'bytes': args.encoding = None

    # First go the file once to see how big it is and to build the vocab
    token_to_idx = {}
    idx_to_token = []
    query_list = []
    with codecs.open(args.input_txt, 'r', args.encoding) as f:
        for line in f:
            if len(line) > 1:
                char_list = []
                for char in line:
                    if char not in token_to_idx:
                        token_to_idx[char] = len(token_to_idx)
                        idx_to_token.append(char)
                    char_list.append(token_to_idx[char])
                query_list.append(char_list)

    # Now we can figure out the split sizes
    total_size = len(query_list)
    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size

    if not args.quiet:
        print('Total vocabulary size: %d' % len(token_to_idx))
        print('Total tokens in file: %d' % total_size)
        print('  Training size: %d' % train_size)
        print('  Val size: %d' % val_size)
        print('  Test size: %d' % test_size)

    # Just load data into memory ... we'll have to do something more clever
    # for huge datasets but this should be fine for now
    # Split sentences into train, val, test randomly
    permutation = np.random.permutation(total_size)
    train_query_list = []
    val_query_list = []
    test_query_list = []
    for idx in permutation[:train_size]:
        train_query_list.append(query_list[idx])
    for idx in permutation[train_size:train_size+val_size]:
        val_query_list.append(query_list[idx])
    for idx in permutation[train_size+val_size:]:
        test_query_list.append(query_list[idx])

    # Dump a JSON file for the vocab using json has expand the 
    json_data = {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
        'train': train_query_list,
        'val': val_query_list,
        'test': test_query_list,
    }
    with open(args.output, 'wb') as f:
        pickle.dump(json_data, f, protocol=-1)