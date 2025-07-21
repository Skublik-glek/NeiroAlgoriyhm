import numpy as np


class Neuron:
    def __init__(self, inputs=10):
        self.weights = []
        for _ in range(inputs):
            self.weights.append(np.random.normal()) # весы
        self.bias = np.random.normal() # порог
        self.learnArgs = None

    def __sigmoid(self, x):
        # Функция активации: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs, learn=False):
        sigmArg = 0
        # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
        for n, input in enumerate(inputs):
            sigmArg += self.weights[n] * input
        sigmArg += self.bias
        if learn:
            self.learnArgs = sigmArg
        return self.__sigmoid(sigmArg)
