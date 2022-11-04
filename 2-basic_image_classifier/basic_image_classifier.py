import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math

#load data
data, meta_data = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

#classification
training_data, test_data = data['train'], data['test']
class_names = meta_data.features["label"].names

#normalize data 
def normalize_data(img, tag):
    img = tf.cast(img, tf.float32)
    img /= 255
    return img, tag

training_data = training_data.map(normalize_data)
test_data = test_data.map(normalize_data)
#cache 
training_data = training_data.cache()
test_data = test_data.cache()
test_data_test = test_data
# #show image
# for img, tag in training_data.take(1):
#     break
# img = img.numpy().reshape((28, 28))
# plt.figure()
# plt.imshow(img, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# #show image
# plt.figure(figsize=(10,10))
# for i, (img, tag) in enumerate(training_data.take(25)):
#     img = img.numpy().reshape((28, 28))
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(img, cmap=plt.cm.binary)
#     plt.xlabel(class_names[tag])
# plt.show()

#model layers
model = tf.keras.Sequential([
    #flatten layer converts array to single dimension
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #hidden layers with relu activation
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    #output layer
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compile with the adam optimizer
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#to train faster it is divided into training batches
LOT_SIZE = 32

num_exp_training = meta_data.splits['train'].num_examples
#random order 
# print(num_exp_training) ----> 60000
# print(meta_data.splits['test'].num_examples) ----> 10000

training_data = training_data.repeat().shuffle(num_exp_training).batch(LOT_SIZE)
test_data = test_data.batch(LOT_SIZE)

#training
print("training...")
history = model.fit(training_data, epochs=5, steps_per_epoch=math.ceil(num_exp_training/LOT_SIZE))
print("trained!")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(history.history['loss'])
# plt.show() 

#test prediction
for (img, tag) in test_data_test.take(1):
    img = img.numpy().reshape((28, 28))
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(class_names[tag])
plt.show()

image = np.array([img])
prediccion = model.predict(image)
print("Prediccion: " + class_names[np.argmax(prediccion[0])])