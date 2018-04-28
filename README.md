# NNLM

## Overview
This is a course project for the course [Natural Language Understanding](http://www.da.inf.ethz.ch/teaching/2018/NLU/) at ETH Zürich. The code implements a simple LSTM language model, and could be used to calculate probability of a sentence and generate sentences given initial words as well.

## How to run the code
```bash
# download corpus and pretrained word embeddings
./get_data.sh

# show help
python model.py -h

# training with default setting
python model.py --n_epoch 3

# using pretrained word embeddings
python model.py --n_epoch 3 --pretrained vec/pretrained_embedding.vec

# increasing hidden dimension
python model.py --n_epoch 3 --pretrained vec/pretrained_embedding.vec --state_dim 1024 --hidden_proj_dim 512

# sentence generation
python model.py --n_epoch 3 --pretrained vec/pretrained_embedding.vec --conti_corpus data/sentences.continuation
```