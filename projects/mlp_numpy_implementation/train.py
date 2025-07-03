# train.py

import numpy as np
from model import MLPModel

X_train = np.array([
    [0.1, 0.5],
    [0.9, 0.7],
    [0.4, 0.1],
    [0.8, 0.9]
])
y_train = np.array([0, 1, 0, 1])

model = MLPModel(
    input_dim=2,
    output_dim=1,
    hidden_layers=2,
    hidden_dim=4,
    activation="relu"
)

lr = 0.01
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]

        y_pred = model.forward(x)
        loss = (y_pred - y) ** 2
        total_loss += loss

        model.backward(y)
        model.update(lr)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")