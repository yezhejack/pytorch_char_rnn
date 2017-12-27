# char-rnn implemented in pytorch
## Preprocess data
Put the data into `data` and then feed them to the `preprocess.py` to create a `pickle` file which contains a `dict`. The `dict` contains a `token_to_idx`, `idx_to_token`, `train_query_list`, `val_query_list` and `test_query_list`.   

```
python preprocess --input_txt data/tiny-shakespeare.txt \
                  --output data/tiny_shakespear.r \
```
## DataProvider
Return the x and y, y is behind 1 time step of x. 
For example, a query is `How are you!\n`. Then input will be `How are you!` and then output should be `are you !\n`.

## training data
```
python train.py
```

## TensorboardX 
Used for visualizing the result of training