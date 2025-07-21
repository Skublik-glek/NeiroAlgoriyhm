import numpy as np
from neuron import Neuron
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
    __inside = []
    __results = []

    def __init__(self, insidesLayerNeuronCount=3, outSideLayerCount=1, inputsCount=3, empty=False):
        if not empty:
            self.insidesLayerNeuronCount = insidesLayerNeuronCount
            self.outSideLayerCount = outSideLayerCount
            self.inputsCount = inputsCount
            for _ in range(insidesLayerNeuronCount):
                self.__inside.append(Neuron(inputsCount))
            for _ in range(outSideLayerCount):
                self.__results.append(Neuron(insidesLayerNeuronCount))
        else:
            pass

    def getModel(self, fileName): # Выгрузка модели
        data = ""
        data += "Характеристики сети:\n"
        data += f"\n\nВходы: {self.inputsCount}\n"
        data += f"\n\nНейроны внутреннего слоя: {self.insidesLayerNeuronCount}\n"
        for neuron in self.__inside:
            data += f"Число весов: {len(neuron.weights)}\n"
            data += f"Весы: "
            for we in neuron.weights:
                data += f"{we} "
            data += f"\nПорог: {neuron.bias}\n"
        data += f"\n\nНейроны выходного слоя: {self.outSideLayerCount}\n"
        for neuron in self.__results:
            data += f"Число весов: {len(neuron.weights)}\n"
            data += f"Весы: "
            for we in neuron.weights:
                data += f"{we} "
            data += f"\nПорог: {neuron.bias}"
        with open(fileName, mode="w", encoding="utf-8") as file:
            file.write(data)
        return data
        

    def load(self, fileName): # Загрузка модели
        self.__inside.clear()
        self.__results.clear()
        inside = True
        tempWeights = []
        
        with open(fileName, mode="r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                if "Входы:" in line:
                    self.inputsCount = int(line.split()[1])
                if "Нейроны внутреннего слоя:" in line:
                    inside = True
                    self.insidesLayerNeuronCount = int(line.split()[3])
                    tempWeights = []
                if "Нейроны выходного слоя:" in line:
                    inside = False
                    self.outSideLayerCount = int(line.split()[3])
                    tempWeights = []
                if "Весы" in line:
                    for we in line.split()[1::]:
                        tempWeights.append(float(we))
                if "Порог:" in line:
                    if inside:
                        newNeuron = Neuron(self.inputsCount)
                        newNeuron.weights = tempWeights
                        newNeuron.bias = float(line.split()[1])
                        self.__inside.append(newNeuron)
                    else:
                        newNeuron = Neuron(self.insidesLayerNeuronCount)
                        newNeuron.weights = tempWeights
                        newNeuron.bias = float(line.split()[1])
                        self.__results.append(newNeuron)
                    tempWeights = []

    def feedforward(self, inputs): # Подаём данные на вход
        values = []
        for neuron in self.__inside:
            neuron: Neuron
            values.append(neuron.feedforward(inputs))
        results = []
        for neuron in self.__results:
            results.append(neuron.feedforward(values))
        return results

    def train(self, data, all_y_trues, learn_rate=0.01, epochs=100000, debug=True, show=False):
        learn_rate = learn_rate
        epochs = epochs # сколько раз пройти по всему набору данных
        start = time.time()

        for _ in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Прямой проход
                values = []
                insideSums = []
                for neuron in self.__inside:
                    neuron: Neuron
                    values.append(neuron.feedforward(x, learn=True))
                    insideSums.append(neuron.learnArgs)

                outs = []
                outSums = []
                for neuron in self.__results:
                    neuron: Neuron
                    outs.append(neuron.feedforward(values, learn=True))
                    outSums.append(neuron.learnArgs)

                # Считаем частные производные.
                d_L_d_ypreds = []
                for n, result in enumerate(outs):
                    d_L_d_ypreds.append(-2 * (y_true[n] - result))

                # Нейрон выхода
                d_ypred_d_ows = []
                d_ypred_d_obs = []
                for out in outSums:
                    owsForCur = []
                    for value in values:
                        owsForCur.append(value * deriv_sigmoid(out))
                    d_ypred_d_ows.append(owsForCur)
                    d_ypred_d_obs.append(deriv_sigmoid(out))

                d_ypred_d_hs = []
                for n, result in enumerate(self.__results):
                    result: Neuron
                    d_ypred_d_hs_combs = []
                    for weight in result.weights:
                        d_ypred_d_hs_combs.append(weight * deriv_sigmoid(outSums[n]))
                    d_ypred_d_hs.append(d_ypred_d_hs_combs)

                # Нейроны внутреннего слоя
                d_h1_d_ws = []
                d_h1_d_bs = []
                for n, neuron in enumerate(self.__inside):
                    weights = []
                    for elem in x:
                        weights.append(elem * deriv_sigmoid(insideSums[n]))
                    d_h1_d_ws.append(weights)
                    d_h1_d_bs.append(deriv_sigmoid(insideSums[n]))

                # Обновляем веса и пороги
                for rn, pred in enumerate(d_L_d_ypreds):
                    for n, neuron in enumerate(self.__inside):
                        for n_weight in range(len(neuron.weights)):
                            self.__inside[n].weights[n_weight] -= learn_rate * pred * d_ypred_d_hs[rn][n] * d_h1_d_ws[n][n_weight]
                        self.__inside[n].bias -= learn_rate * pred * d_ypred_d_hs[rn][n] * d_h1_d_bs[n]
                    for n, neuron in enumerate(self.__results):
                        for n_weight in range(len(neuron.weights)):
                            self.__results[n].weights[n_weight] -= learn_rate * pred * d_ypred_d_ows[n][n_weight]
                        self.__results[n].bias -= learn_rate * pred * d_ypred_d_obs[n]
                if show:
                    print(outs, y_true)
            if _ % 10000 == 0 and debug:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (_, loss))
        if debug:
            print(f"Обучение заняло {time.time() - start}")
            print(f"Параметры: {epochs} - проходов, {learn_rate} - коэффициент обучения")
        print("Обучено")
    
    def getResult(self, inputs):
        answer = self.feedforward(inputs)
        if answer[0] <= 0.5:
            return f"Это женщина | ~ {1 - round(answer[0], 2)}"
        else:
            return f"Это мужчина | ~ {round(answer[0], 2)}"
