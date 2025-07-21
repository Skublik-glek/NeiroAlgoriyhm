from network import NeuralNetwork
from inputController import Input
import numpy as np
from time import sleep


while True:
    net = NeuralNetwork(empty=True)
    net.load("model1.txt")
    getter = Input()
    getter.setRandom()
    getter.putNewTrainData("traindata.txt")
    # print(getter.array, net.getResult(getter.array))
    tt = getter.getTrainData()
    print(tt[0], tt[1])
    net.train(tt[0], tt[1])
    break
