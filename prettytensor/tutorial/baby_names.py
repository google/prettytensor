# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tutorial to predict the sex of a baby from the name.

This model takes a dataset of baby names to ratio of sexes for that name and
then trains an LSTM to predict the ratio given the characters of the name.

The input are the characters of the name as ASCII codes (0-127) and it is
unrolled for 15 steps, which is the longest name in the corpus.  The results are
fed through a recurrent network and then to a 2 way classifier that predicts the
boy/girl ratio.

This demonstrates how to train a classifier on the last output using an LSTM,
which can be at any point (some names are short and some are long) by setting
weights on each example. This also demonstrates how to efficiently reshape the
network for the classifier and how to use dropout in both a training and eval
graph.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string(
    'save_path', None, 'Where to save the model checkpoints on local disk. '
    'Checkpoints are in LevelDb.')
FLAGS = tf.app.flags.FLAGS


BATCH_SIZE = 32
CHARS = 128
TIMESTEPS = 15
SEXES = 2

EMBEDDING_SIZE = 16


def create_model(text_in,
                 labels,
                 timesteps,
                 per_example_weights,
                 phase=pt.Phase.train):
  """Creates a model for running baby names."""
  with pt.defaults_scope(phase=phase, l2loss=0.00001):
    # The embedding lookup must be placed on a cpu.
    with tf.device('/cpu:0'):
      embedded = text_in.embedding_lookup(CHARS, [EMBEDDING_SIZE])
    # We need to cleave the sequence because sequence lstm expect each
    # timestep to be in its own Tensor.
    lstm = (embedded.cleave_sequence(timesteps).sequence_lstm(CHARS))

    # The classifier is much more efficient if it runs across the entire
    # batch at once, so we want to squash (i.e. uncleave).
    #
    # Hidden nodes is set to 32 because it seems to work well.
    return (lstm.squash_sequence().fully_connected(32,
                                                   activation_fn=tf.nn.relu)
            .dropout(0.7)
            .softmax_classifier(SEXES,
                                labels,
                                per_example_weights=per_example_weights))


def main(_=None):
  print('Starting Baby Names')

  # Since we are feeding our data as numpy arrays, we need to create
  # placeholders in the graph.
  # These must then be fed using the feed dict.
  input_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, TIMESTEPS])
  output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, SEXES])

  inp = data_utils.reshape_data(input_placeholder)

  # Create a label for each timestep.
  labels = data_utils.reshape_data(
      tf.reshape(
          tf.tile(output_placeholder, [1, TIMESTEPS]), [BATCH_SIZE, TIMESTEPS,
                                                        SEXES]),
      per_example_length=2)

  # We also need to set per example weights so that the softmax doesn't output a
  # prediction on intermediate nodes.
  length_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, 1])

  # We need a dense multiplier for the per example weights.  The only place
  # that has a non-zero loss is the first EOS after the last character of the
  # name; the characters in the name and the trailing EOS characters are given a
  # 0 loss by assigning the weight to 0.0 and in the end only one character in
  # each batch has a weight of 1.0.
  # sparse_to_dense does a lookup using the indices from the first Tensor.
  # Because we are filling in a 2D array, the indices need to be 2 dimensional.
  # Since we want to assign 1 value for each row, the first dimension can just
  # be a sequence.
  t = tf.concat(1,
                [
                    tf.constant(
                        numpy.arange(BATCH_SIZE).reshape((BATCH_SIZE, 1)),
                        dtype=tf.int32), length_placeholder
                ])

  # Squeeze removes dimensions that are equal to 1.  per_example_weights must
  # end up as 1 dimensional.
  per_example_weights = data_utils.reshape_data(tf.sparse_to_dense(
      t, [BATCH_SIZE, TIMESTEPS], 1.0, default_value=0.0)).squeeze()

  # We need 2 copies of the graph that share variables.  The first copy runs
  # training and will do dropout if specified and the second will not include
  # dropout.  Dropout is controlled by the phase argument, which sets the mode
  # consistently throughout a graph.
  with tf.variable_scope('baby_names'):
    result = create_model(inp, labels, TIMESTEPS, per_example_weights)

  # Call variable scope by name so we also create a name scope.  This ensures
  # that we share variables and our names are properly organized.
  with tf.variable_scope('baby_names', reuse=True):
    # Some ops have different behaviors in test vs train and these take a phase
    # argument.
    test_result = create_model(inp,
                               labels,
                               TIMESTEPS,
                               per_example_weights,
                               phase=pt.Phase.test)

  # For tracking accuracy in evaluation, we need to add an evaluation node.
  # We only run this when testing, so we need to specify that in the phase.
  # Some ops have different behaviors in test vs train and these take a phase
  # argument.
  accuracy = test_result.softmax.evaluate_classifier(
      labels,
      phase=pt.Phase.test,
      per_example_weights=per_example_weights)

  # We can also compute a batch accuracy to monitor progress.
  batch_accuracy = result.softmax.evaluate_classifier(
      labels,
      phase=pt.Phase.train,
      per_example_weights=per_example_weights)

  # Grab the inputs, outputs and lengths as numpy arrays.
  # Lengths could have been calculated from names, but it was easier to
  # calculate inside the utility function.
  names, sex, lengths = data_utils.baby_names(TIMESTEPS)

  epoch_size = len(names) // BATCH_SIZE
  # Create the gradient optimizer and apply it to the graph.
  # pt.apply_optimizer adds regularization losses and sets up a step counter
  # (pt.global_step()) for you.
  # This sequence model does very well with initially high rates.
  optimizer = tf.train.AdagradOptimizer(
      tf.train.exponential_decay(1.0,
                                 pt.global_step(),
                                 epoch_size,
                                 0.95,
                                 staircase=True))
  train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

  # We can set a save_path in the runner to automatically checkpoint every so
  # often.  Otherwise at the end of the session, the model will be lost.
  runner = pt.train.Runner(save_path=FLAGS.save_path)
  with tf.Session():
    for epoch in xrange(100):
      # Shuffle the training data.
      names, sex, lengths = data_utils.permute_data((names, sex, lengths))

      runner.train_model(
          train_op,
          [result.loss, batch_accuracy],
          epoch_size,
          feed_vars=(input_placeholder, output_placeholder, length_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, names, sex, lengths),
          print_every=100)
      classification_accuracy = runner.evaluate_model(
          accuracy,
          epoch_size,
          print_every=0,
          feed_vars=(input_placeholder, output_placeholder, length_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, names, sex, lengths))

      print('Accuracy after epoch %d: %g%%' % (
          epoch + 1, classification_accuracy * 100))


if __name__ == '__main__':
  tf.app.run()
