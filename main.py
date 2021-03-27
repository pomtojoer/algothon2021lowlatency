from sys import stdin
import numpy as np
import tensorflow as tf

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

# Read all then predict for all
data = np.array([[float(x) for x in line.split(',')] for line in stdin if line != '\n'])
data = data.reshape((data.shape[0], 5, 100, 1))
prediction = model.predict(data)
prediction = prediction.reshape(prediction.shape[0])
[print(prediction_point) for prediction_point in prediction]

# # if you really need it line by line, then fine, use this instead.....
# for line in stdin:
#     if line == '':
#         break
#     data=np.array([float(x) for x in line.split(',')])
#     data=data.reshape((1, 5, 100, 1))
#     print(model.predict(data)[0,0])
