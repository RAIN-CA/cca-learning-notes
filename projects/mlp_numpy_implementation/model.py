# model.py

import numpy as np
from utils import activation_functions

class NeuronModel:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim)
        self.b = np.random.randn()

        # 初始化梯度
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = 0.0

    def forward(self, x):
        """
        前向传播，保存中间变量以供反向传播使用
        """
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        self.y_hat = sigmoid(self.z)
        return self.y_hat

    def backward(self, y_true):
        """
        反向传播，根据 y_true 和 y_hat 计算损失的梯度
        """
        dL_dyhat = 2 * (self.y_hat - y_true)
        dyhat_dz = sigmoid_derivative(self.z)
        dL_dz = dL_dyhat * dyhat_dz

        # 梯度
        self.grad_w = dL_dz * self.x
        self.grad_b = dL_dz * 1

    def update(self, lr):
        """
        根据梯度更新权重和偏置
        """
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b



class MLPModel:
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim, activation="sigmoid"):
        assert activation in activation_functions, f"不支持激活函数: {activation}"

        self.activation_name = activation
        self.act_fn, self.act_deriv = activation_functions[activation]
        self.output_fn, self.output_deriv = activation_functions["sigmoid"]  # 输出默认 sigmoid

        # 构造层结构列表，如 [2, 4, 4, 1]
        layer_sizes = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        self.num_layers = len(layer_sizes) - 1

        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            W = np.random.randn(layer_sizes[i + 1], layer_sizes[i])
            b = np.random.randn(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

        self.cache = {}

    def forward(self, x):
        activations = [x]
        zs = []

        a = x
        for i in range(self.num_layers):
            W, b = self.weights[i], self.biases[i]
            z = np.dot(W, a) + b

            if i == self.num_layers - 1:
                a = self.output_fn(z)
            else:
                a = self.act_fn(z)

            zs.append(z)
            activations.append(a)

        self.cache['activations'] = activations
        self.cache['zs'] = zs
        return activations[-1][0]

    def backward(self, y_true):
        activations = self.cache['activations']
        zs = self.cache['zs']

        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # 输出层误差
        delta = 2 * (activations[-1] - y_true) * self.output_deriv(zs[-1])
        grads_w[-1] = np.outer(delta, activations[-2])
        grads_b[-1] = delta

        for l in range(self.num_layers - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].T, delta) * self.act_deriv(zs[l])
            grads_w[l] = np.outer(delta, activations[l])
            grads_b[l] = delta

        self.grads_w = grads_w
        self.grads_b = grads_b

    def update(self, lr):
        for i in range(self.num_layers):
            self.weights[i] -= lr * self.grads_w[i]
            self.biases[i] -= lr * self.grads_b[i]