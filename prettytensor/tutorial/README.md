# Tutorial

These are simple models that highlight some of Pretty Tensor's features and
that hopefully will be useful branching points for your own experiments.

While each tutorial is intended to be standalone, the recommended order is:

1. `mnist.py`
2. `baby_names.py`
3. `shakespeare.py`

All of the tutorials show you how to build a model and run training/evaluation.

## MNIST

`mnist.py` shows a simple image classification model using the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Baby Names

`baby_names.py` is a recurrent network that uses data from the Office of
Retirement and Disability Policy, Social Security Administration about all
children born in the US for the past century.  The model uses an
[Long Short Term Memory](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
(LSTM) to make a prediction on the boy/girl ratio for each name.

## Shakespeare

`shakespeare.py` uses stacked LSTMs to read each character of Shakespeare and
predict the next character.  It can be used to sample your own Shakespearian
comedy.
