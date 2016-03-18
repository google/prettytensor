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
"""Shakespeare tutorial.

The Shakespeare tutorial downloads a snippet of Shakespeare, munges the data
into the correct format and then creates a 2 layer LSTM to predict the next
character given the current character.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random


import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string(
    'save_path', None, 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_integer(
    'epochs', 10, 'The number of epochs to run training on this model.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 8
CHARS = 128
TIMESTEPS = 100

# The size of the embedding for each character that will be learned.
EMBEDDING_SIZE = 16

# The number of cells in the lower and upper LSTM layers.
LOWER = 128
UPPER = 256


def create_model(text_in, timesteps, phase):
  """Creates a 2 layer LSTM model with dropout.

  Args:
    text_in: The input text as ASCII ordinals in a Tensor.
    timesteps: The number of timesteps in the sequence.
    phase: Phase controls whether or not dropout is active.  In training mode
      we want to perform dropout, but in test we want to disable it.
  Returns:
    The logits.
  """
  with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
    # The embedding lookup must be placed on a cpu.
    with tf.device('/cpu:0'):
      embedded = text_in.embedding_lookup(CHARS, [EMBEDDING_SIZE])
    # Because the sequence LSTM expects each timestep to be its own Tensor,
    # we need to cleave the sequence.
    # Below we can build a stacked 2 layer LSTM by just chaining them together.
    # You can stack as many layers as you want.
    lstm = (embedded
            .cleave_sequence(timesteps)
            .sequence_lstm(LOWER)
            .sequence_lstm(UPPER))

    # The classifier is much more efficient if it runs across the entire
    # dataset at once, so we want to squash (i.e. uncleave).
    # Note: if phase is test, dropout is a noop.
    return (lstm.squash_sequence()
            .dropout(keep_prob=0.8, phase=phase)
            .fully_connected(CHARS, activation_fn=None))


def sample(
    input_placeholder, logits, seed=None, max_length=1024, temperature=1.0):
  """Samples from the LSTM model.

  Sampling is done by first running either the seed or an arbitrary character
  through the model and then drawing the next character from the probability
  distribution definted by `softmax`.

  Args:
    input_placeholder: A placeholder that expects a scalar feed.
    logits: The logits.  This works with the logits so that it can apply the
      temperature.
    seed: Either a string of characters to prime the network or None.
    max_length: The maximum length to draw in case EOS is not reached.
    temperature: A value that is used to renormalize the inputs.  A higher value
      selects less likely choices.
  Returns:
    A string that was sampled from the model.
  """
  assert temperature > 0, 'Temperature must be greater than 0.'
  if not seed:
    # The model expects an input to do inference, so seed with a single letter.
    seed = chr(ord('A') + random.randint(0, 25))
  result = ''

  # The recurrent runner takes care of tracking the model's state at each step
  # and provides a reset call to zero it out for each query.
  recurrent_runner = pt.train.RecurrentRunner()

  # We need to reset the hidden state for each query.
  recurrent_runner.reset()
  # Initialize the system
  for c in seed[:-1]:
    recurrent_runner.run([logits],
                         {input_placeholder: data_utils.convert_to_int(c)})
    result += c

  # Start sampling!
  ci = ord(seed[-1])
  while len(result) < max_length and ci != data_utils.EOS:
    result += chr(ci)
    # The softmax is probability normalized and would have been appropriate here
    # if we weren't applying the temperature (temperature could also be done in
    # TensorFlow).
    logit_result = recurrent_runner.run([logits],
                                        {input_placeholder: ci})[0][0]
    logit_result /= temperature

    # Apply the softmax in numpy to convert from logits to probabilities.
    # Subtract off the max for numerical stability -- logits are invariant to
    # additive scaling and this eliminates overflows.
    logit_result -= logit_result.max()

    distribution = numpy.exp(logit_result)
    distribution /= distribution.sum()

    # Numpy multinomial needs the value to be strictly < 1
    distribution -= .00000001
    ci = numpy.argmax(numpy.random.multinomial(1, distribution))
  result += chr(ci)  # Add the last letter.
  return result


def main(_=None):
  print('Starting Shakespeare')

  # Since we are feeding our data as numpy arrays, we need to create
  # placeholders in the graph.
  # These must then be fed using the feed dict.
  input_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, TIMESTEPS])
  output_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, TIMESTEPS])

  merged_size = BATCH_SIZE * TIMESTEPS

  inp = data_utils.reshape_data(input_placeholder)

  # We need a dense output to calculate loss and accuracy.
  # sparse_to_dense does a lookup using the indices from the first Tensor.
  # Because we are filling in a 2D array, the indices need to be 2 dimensional.
  t = tf.concat(1,
                [
                    tf.constant(
                        numpy.arange(merged_size).reshape((merged_size, 1)),
                        dtype=tf.int32),
                    data_utils.reshape_data(output_placeholder)
                ])

  labels = tf.sparse_to_dense(t, [merged_size, CHARS], 1.0, 0.0)

  # Some ops have different behaviors in test vs train and these take a phase
  # argument.
  with tf.variable_scope('shakespeare'):
    training_logits = create_model(inp, TIMESTEPS, pt.Phase.train)
    # Create the result.  Softmax applies softmax and creates a cross entropy
    # loss.  The result is a namedtuple.
    training_result = training_logits.softmax(labels)

  # Create the gradient optimizer and apply it to the graph.
  # pt.apply_optimizer adds regularization losses and sets up a step counter
  # (pt.global_step()) for you.
  optimizer = tf.train.AdagradOptimizer(0.5)
  train_op = pt.apply_optimizer(optimizer, losses=[training_result.loss])

  # For tracking accuracy in evaluation, we need to add an evaluation node.
  # We only run this when testing, so we need to specify that in the phase.
  # We also want to disable dropout, so we pass the phase to create_model.

  # Call variable scope by name so we also create a name scope.  This ensures
  # that we share variables and our names are properly organized.
  with tf.variable_scope('shakespeare', reuse=True):
    test_logits = create_model(inp, TIMESTEPS, pt.Phase.test)
    test_result = test_logits.softmax(labels)

  # Accuracy creates variables, so make it outside of the above scope.
  accuracy = test_result.softmax.evaluate_classifier(labels,
                                                     phase=pt.Phase.test)

  # Create an inference model so that we can sample.  The big difference is
  # that the input is a single character and it requires reset nodes.
  # Also place summaries in a different collection. The default summaries have
  # dependencies on running the graph and would introduce a dependence on the
  # inference placeholder.
  with tf.variable_scope('shakespeare', reuse=True), pt.defaults_scope(
      summary_collections=['INFERENCE_SUMMARIES']):
    inference_input = tf.placeholder(tf.int32, [])
    # Needs to be 2 dimensional so that it matches the dims of the other models.
    reshaped = pt.wrap(inference_input).reshape([1, 1])
    inference_logits = create_model(reshaped, 1, pt.Phase.infer)

  # Grab the data as numpy arrays.
  shakespeare = data_utils.shakespeare(TIMESTEPS + 1)
  shakespeare_in = shakespeare[:, :-1]
  shakespeare_out = shakespeare[:, 1:]

  # We can set a save_path in the runner to automatically checkpoint every so
  # often.  Otherwise at the end of the session, the model will be lost.
  runner = pt.train.Runner(save_path=FLAGS.save_path)
  with tf.Session():
    for epoch in xrange(FLAGS.epochs):
      # Shuffle the training data.
      shakespeare_in, shakespeare_out = data_utils.permute_data(
          (shakespeare_in, shakespeare_out))

      runner.train_model(train_op,
                         training_result.loss,
                         len(shakespeare_in) // BATCH_SIZE,
                         feed_vars=(input_placeholder, output_placeholder),
                         feed_data=pt.train.feed_numpy(
                             BATCH_SIZE, shakespeare_in, shakespeare_out),
                         print_every=10)
      classification_accuracy = runner.evaluate_model(
          accuracy,
          len(shakespeare_in) // BATCH_SIZE,
          feed_vars=(input_placeholder, output_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, shakespeare_in,
                                        shakespeare_out))

      print('Next character accuracy after epoch %d: %g%%' % (
          epoch + 1, classification_accuracy * 100))

      # Use a temperature smaller than 1 because the early stages of the model
      # don't assign much confidence.
      print(sample(inference_input,
                   inference_logits,
                   max_length=128,
                   temperature=0.5))

    # Print a sampling from the model.
    print(sample(inference_input, inference_logits))


if __name__ == '__main__':
  tf.app.run()
