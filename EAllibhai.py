"""
https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

"""
import matplotlib.pyplot as plt


# load the MNIST data set. Consists of 70k images for digits.
 from keras.datasets import mnist

 #split into training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#check an image
plt.imshow(x_train[0])
#dimensions ?
x_train[0].shape

#reshape data to fit model
train_size = 60000 #number of training images
test_size = 10000
img_shape = [28, 28]

x_train = x_train.reshape(train_size, img_shape[0], img_shape[1], 1)
x_test = x_test.reshape(test_size, img_shape[0], img_shape[1], 1)

#one-hot-encode target
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]

#build model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#initiate
model = Sequential()

#add layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)