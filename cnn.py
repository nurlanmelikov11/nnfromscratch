from sequential import *
from layers import *
import numpy as np


# load data and filter
filtered_data = np.load('filtered_mnist.npz')
x_train_filtered = np.expand_dims(filtered_data['x_train'], axis=-1)[:1]
y_train_filtered = filtered_data['y_train'][:1]
x_test_filtered = np.expand_dims(filtered_data['x_test'], axis=-1)
y_test_filtered = filtered_data['y_test']

# check shape potential mismatch
print(x_train_filtered.shape, y_train_filtered.shape, "x, y train shapes")

# defining layers
l1 = Conv2d(input_channel=1, kernel_size=3, number_of_kernels=32,
            activation='relu', stride=1, padding=0)
l2 = MaxPool(pool_size=2, stride=1)
l3 = Conv2d(input_channel=32, kernel_size=3, number_of_kernels=64,
            activation='relu', stride=1, padding=0)
l4 = MaxPool(pool_size=2, stride=1)

# make shapes fit itself auto;
l5 = Flatten()
l6 = Dense(30976, 15000, activation='relu')
l7 = Dense(15000, 300, activation='relu')
l8 = Dense(300, 1, activation='sigmoid')


# stacking and creating model
layer_stack = [
    l1, l2, l3, l4, l5, l6, l7, l8
]

model = Sequential(layer_stack)
model.compile(metric='accuracy')
epoch = 100


# starting train
cost = model.fit(x_train_filtered, y_train_filtered,
                 epoch=epoch, batch_size=50, alpha=0.01)


# report error and test

preds = model.predict(x_test_filtered[:10])
preds = preds.flatten()
acc = accuracy(preds, y_test_filtered[:10], threshold=0.5)


print(f"Epoch: {epoch}, cost: ", cost, "acc", acc)

# model = Sequential(layer_stack)
# model.compile(metric='accuracy')
# model.fit(X_train, y_train, epoch=3000, batch_size=1, alpha=0.045)

# print(model.predict(X_train))
