# char-rnn implemented in pytorch
## Preprocess data
Put the data into `data` and then feed them to the `preprocess.py` to create a `pickle` file which contains two `dict`. The two `dict` contains a `token_to_idx`, `idx_to_token`, `train_query_list`, `val_query_list` and `test_query_list`.   

```
python preprocess --input_txt data/tiny-shakespeare.txt \
                  --output_vocab data/tiny_shakespear.vocab.pickle \
                  --output_data data/tiny_shakespear.data.pickle \
```
## DataProvider
Return the x and y, y is behind 1 time step of x. 
For example, a query is `How are you!`. Then input will be `How are you!` and then output should be `are you !<EOS>`.

## training data
```
python train.py --input_vocab data/tiny_shakespear.vocab.pickle \
                --input_data data/tiny_shakespear.data.pickle
```

## TensorboardX 
Used for visualizing the result of training