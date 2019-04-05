import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

celsius_q = np.array([-50, -40, -30, -20, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
fahrenheit_a = np.array([-58, -40, -22, -4, 14, 15.8, 17.6, 19.4, 21.2, 23, 24.8, 26.6, 28.4, 30.2, 32, 33.8, 35.6, 37.4, 39.2, 41, 42.8, 44.6, 46.4, 48.2, 50, 68, 86, 104, 122, 140, 158, 176, 194, 212, 392, 572, 752, 932, 1112, 1292, 1472, 1652, 1832], dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenhet".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
history = model.fit(celsius_q, fahrenheit_a, epochs=10000, verbose=True)
print("Finished treaning the model")

plt.xlabel('Epochs Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([100.0]))
print(model.predict([300.0]))
