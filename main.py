from sys import stdin
import tensorflow as tf


with tf.device("/cpu:0"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 100, 1)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimiser = tf.keras.optimizers.Adam(lr=0.0001)

    loss = tf.keras.losses.MeanAbsoluteError()

    metrics = tf.keras.metrics.MeanSquaredError()

    model.compile(loss=loss, optimizer=optimiser, metrics=metrics)

    weights_file = 'weight.h5'

    model.load_weights('weight.h5')

for line in stdin:
    if line == '':
        break
    d=np.array([float(x) for x in line.split(',')])
    print(d)
