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
"""Data utils bundles the utilties to download and munge data in numpy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import os.path
import sys



import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves.urllib import request
import tensorflow as tf

import prettytensor as pt


WORK_DIRECTORY = '/tmp/data'
MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'
UNK = 0
EOS = 1


def maybe_download(url, filename):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(WORK_DIRECTORY):
    os.mkdir(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not os.path.exists(filepath):
    filepath, _ = request.urlretrieve(url + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def mnist_extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].

  Args:
    filename: The local filename.
    num_images: The number of images in this file.
  Returns:
    The data as a numpy array with the values centered.
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data -= 255 / 2.0
    data /= 255.0
    data = data.reshape(num_images, 28, 28, 1)
    return data


def mnist_extract_labels(filename, num_images):
  """Extract the labels into a 1-hot matrix [image index, label index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8)
  # Convert to dense 1-hot representation.
  return (np.arange(10) == labels[:, None]).astype(np.float32)


def permute_data(arrays, random_state=None):
  """Permute multiple numpy arrays with the same order."""
  if any(len(a) != len(arrays[0]) for a in arrays):
    raise ValueError('All arrays must be the same length.')
  if not random_state:
    random_state = np.random
  order = random_state.permutation(len(arrays[0]))
  return [a[order] for a in arrays]


def mnist(training):
  """Downloads MNIST and loads it into numpy arrays."""
  if training:
    data_filename = 'train-images-idx3-ubyte.gz'
    labels_filename = 'train-labels-idx1-ubyte.gz'
    count = 60000
  else:
    data_filename = 't10k-images-idx3-ubyte.gz'
    labels_filename = 't10k-labels-idx1-ubyte.gz'
    count = 10000
  data_filename = maybe_download(MNIST_URL, data_filename)
  labels_filename = maybe_download(MNIST_URL, labels_filename)

  return (mnist_extract_data(data_filename, count),
          mnist_extract_labels(labels_filename, count))


def convert_to_int(char):
  i = ord(char)
  if i >= 128:
    return UNK
  return i


def shakespeare(chunk_size):
  """Downloads Shakespeare, converts it into ASCII codes and chunks it.

  Args:
    chunk_size: The dataset is broken down so that it is shaped into batches x
      chunk_size.
  Returns:
    A numpy array of ASCII codes shaped into batches x chunk_size.
  """
  file_name = maybe_download('http://cs.stanford.edu/people/karpathy/char-rnn/',
                             'shakespear.txt')
  with open(file_name) as f:
    shakespeare_full = f.read()

  # Truncate the data.
  length = (len(shakespeare_full) // chunk_size) * chunk_size
  if length < len(shakespeare_full):
    shakespeare_full = shakespeare_full[:length]
  arr = np.array([convert_to_int(c) for c in shakespeare_full])[
      0:len(shakespeare_full) / chunk_size * chunk_size]
  return arr.reshape((len(arr) / chunk_size, chunk_size))


def baby_names(max_length=15):
  """Opens the baby_names csv file and produces numpy array.

  Args:
    max_length: The maximum length, 15 was the longest name when this was
      written.  Short entries will be padded with the EOS marker.
  Returns:
    A numpy array of the names converted to ascii codes, the labels and an
    array of lengths.
  Raises:
    ValueError: if max_length is too small.
  """
  names = []
  lengths = []
  targets = []
  with open(os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                         'baby_names.csv'), 'rb') as f:
    first = True
    for l in csv.reader(f, delimiter=','):
      if first:
        first = False
        continue
      assert len(l) == 4, l
      name = l[0]
      if max_length < len(name):
        raise ValueError('Max length is too small: %d > %d' %
                         (max_length, len(name)))
      chars = [convert_to_int(c) for c in name]
      names.append(chars + ([EOS] * (max_length - len(chars))))
      lengths.append([len(name)])
      values = [float(l[2]), float(l[3])]
      if abs(sum(values) - 1) > 0.001:
        raise ValueError('Each row must sum to 1: %s' % l)
      targets.append(values)
  return np.array(names), np.array(targets), np.array(lengths)


def reshape_data(tensor, per_example_length=1):
  """Reshapes input so that it is appropriate for sequence_lstm..

  The expected format for sequence lstms is
  [timesteps * batch, per_example_length] and the data produced by the utilities
  is [batch, timestep, *optional* expected_length].  The result can be cleaved
  so that there is a Tensor per timestep.

  Args:
    tensor: The tensor to reshape.
    per_example_length: The number of examples at each timestep.
  Returns:
    A Pretty Tensor that is compatible with cleave and then sequence_lstm.

  """
  # We can put the data into a format that can be easily cleaved by
  # transposing it (so that it varies fastest in batch) and then making each
  # component have a single value.
  # This will make it compatible with the Pretty Tensor function
  # cleave_sequence.
  dims = [1, 0]
  for i in xrange(2, tensor.get_shape().ndims):
    dims.append(i)
  return pt.wrap(tf.transpose(tensor, dims)).reshape([-1, per_example_length])
