from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import pandas as pd
import tables

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

# We import the modified optimizers
from dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer

flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_integer('dpsgd_type',0, 'types: 0-gaussian, 1-laplace, 2-cactus')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', np.sqrt(0.1),'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_boolean('logging', False, 'If True, records will be saved in files.')

FLAGS = flags.FLAGS

NUM_TRAIN_EXAMPLES = 60000

def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / NUM_TRAIN_EXAMPLES
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  
  var=(FLAGS.noise_multiplier*FLAGS.l2_norm_clip)**2
  
  
  if FLAGS.logging:
    # Initialize the norm file
    filename = ('norm_%d_v%1.2f_gradient.npy' %(FLAGS.dpsgd_type, var))
    np.save(filename, np.empty(0))
  
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Define a sequential Keras model
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  if FLAGS.dpsgd:
      optimizer = VectorizedDPKerasSGDOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate,
            dpsgd_type=FLAGS.dpsgd_type,
            logging=FLAGS.logging)
      # Compute vector of per-example loss rather than its mean over a minibatch.
      loss = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True, reduction=tf.losses.Reduction.NONE)
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Information Session:
  print('\n *** Information: dpsgd_type=%d, stddev=%f, epochs=%d *** \n' %(FLAGS.dpsgd_type, FLAGS.noise_multiplier*FLAGS.l2_norm_clip, FLAGS.epochs))
    
  # Compile model with Keras 
  print('\n *** Complie model with Keras. *** \n')
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Train model with Keras
  print('\n *** Train model with Keras. *** \n')
  history = model.fit(train_data, train_labels,
            epochs=FLAGS.epochs,
            validation_data=(test_data, test_labels),
            batch_size=FLAGS.batch_size)
    
 

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = ('privacy_data/log_e%d_%d_d%1.1f_v%1.2f.csv' 
                     %(FLAGS.epochs, FLAGS.dpsgd_type,FLAGS.l2_norm_clip,(FLAGS.noise_multiplier*FLAGS.l2_norm_clip)**2))
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = ('privacy_data/logB_e%d_np.csv' %(FLAGS.epochs))
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    print('Trained with vanilla non-private SGD optimizer')

if __name__ == '__main__':
  app.run(main)
