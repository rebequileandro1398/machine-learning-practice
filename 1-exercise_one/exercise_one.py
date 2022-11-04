import tensorflow as tf
import numpy as np

#celsius to fahrenheit
celsius = np.array([1, -10, 0, 21, 13, 23, 50, -30], dtype=float)
fahrenheit = np.array([33.8, 14, 32, 69.8, 55.4, 73.4, 122, -22], dtype=float)

#layers
#example single layer
# layer = tf.keras.layers.Dense(units=1, input_shape=[1])
# model = tf.keras.Sequential([layer])

hidden_layers_one = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden_layers_two = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hidden_layers_one, hidden_layers_two, output])

#compile with the adam optimizer
history = model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
#training
print("training...")
model.fit(celsius, fahrenheit, epochs=100, verbose=False)
print("trained!")

#testing
print("test")
input_celsius = int(input("enter degrees celsius: "))
prediction = model.predict([input_celsius])
print(str(input_celsius) + " degrees Celsius is " + str(prediction[0][0]) + " Fahrenheit")