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
"""Tests for replay_queue."""


import numpy as np
import tensorflow as tf

import prettytensor as pt


class ReplayableQueueTest(tf.test.TestCase):

  def test_replay_queue_with_queue_input(self):
    # Put a lot of replay.output on the queue
    q = tf.FIFOQueue(1000, tf.float32, [])
    enqueue = q.enqueue_many(tf.to_float(tf.range(0, 1000)))
    replay = pt.train.ReplayableQueue.build_from_queue(q, 100, 10)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())

      sess.run(enqueue)

      d = sess.run(replay.output)
      self.assertAllClose(np.arange(10).astype(np.float), d)

      # Now fill the queue
      replay.refill(sess)

      self.assertEqual(100, replay._replay_queue.size().eval())

      d = sess.run(replay.output)

      # Replay is still false, but the queue has advanced
      self.assertAllClose(np.arange(110, 120).astype(np.float), d)

      # Now set replay.
      replay.set_replay(sess, True)
      for i in range(10):
        d = sess.run(replay.output)
        range_start = 10 + i * 10
        self.assertAllClose(
            np.arange(range_start, range_start + 10).astype(np.float), d)

      # And again
      for i in range(10):
        d = sess.run(replay.output)
        range_start = 10 + i * 10
        self.assertAllClose(
            np.arange(range_start, range_start + 10).astype(np.float), d)

      replay.set_replay(sess, False)

      # Back to the replay.output stream
      d = sess.run(replay.output)
      self.assertAllClose(np.arange(120, 130).astype(np.float), d)

      # And refill the queue
      replay.refill(sess)
      replay.set_replay(sess, True)
      d = sess.run(replay.output)
      self.assertAllClose(np.arange(130, 140).astype(np.float), d)

      replay.set_replay(sess, False)
      d = sess.run(replay.output)
      self.assertAllClose(np.arange(230, 240).astype(np.float), d)

if __name__ == '__main__':
  tf.test.main()
