'''
crappy convnet for MNIST
99.21% accuracy after 25 epochs
'''

# math
import matplotlib.pyplot as plt
import tensorflow as tf

# keras / tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    InputLayer, Input, Conv2D, Dense)
from tensorflow.python.keras.layers import (
    MaxPooling2D, Flatten, BatchNormalization)
from tensorflow.python.keras.optimizers import Adam
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale pixel values to range of 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# one-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def create_model():
    model = Sequential()

    # block 1
    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), strides=1,
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # block 2
    model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # block 3
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # flatten outputs
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    return model

optimizer = Adam(lr=0.001)
model = create_model()
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

'''
training
'''
history = model.fit(x_train, y_train,
                    batch_size=100,
                    epochs=25,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='best')
plt.axis([0, 25, 0, 1])

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='best')
plt.axis([0, 25, 0, 2.25])
