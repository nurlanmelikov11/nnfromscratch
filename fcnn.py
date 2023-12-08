from sequential import *
from layers import *


l1 = Dense(2, 8, activation='relu')
l4 = Dense(8, 1, activation='sigmoid', initialization='normal')

layer_stack = [
    l1, l4
]


X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


y_train = np.array([
    [0],
    [1],
    [1],
    [0]
])


model = Sequential(layer_stack)
model.compile(metric='accuracy')
epoch = 10000
cost = model.fit(X_train, y_train, epoch=epoch, batch_size=4, alpha=0.09)
print(f"Epoch: {epoch}, cost: ", cost)

# x_test = np.array([[0, 0]])
print(model.predict(X_train))


# model = Sequential(layer_stack)
# model.compile(metric='accuracy')
# model.fit(X_train, y_train, epoch=3000, batch_size=1, alpha=0.045)

# print(model.predict(X_train))
