import numpy as np
import random


class Input():
    ways = [
        [50-70, 160-172, 23-50], #М
        [58-70, 166-172, 22-50], #М
        [64-70, 172-172, 24-50], #М
        [85-70, 178-172, 25-50], #М
        [80-70, 184-172, 21-50], #М
        [90-70, 190-172, 27-50], #М
        [52-70, 160-172, 31-50], #М
        [61-70, 166-172, 33-50], #М
        [66-70, 172-172, 36-50], #М
        [77-70, 178-172, 38-50], #М
        [83-70, 184-172, 30-50], #М
        [90-70, 190-172, 27-50], #М
        [74-70, 195-172, 19-50], #М
        [51-70, 162-172, 20-50], #Ж
        [59-70, 168-172, 23-50], #Ж
        [65-70, 174-172, 26-50], #Ж
        [71-70, 180-172, 21-50], #Ж
        [80-70, 186-172, 28-50], #Ж
        [83-70, 190-172, 29-50], #Ж
        [57-70, 162-172, 30-50], #Ж
        [66-70, 168-172, 32-50], #Ж
        [72-70, 174-172, 35-50], #Ж
        [76-70, 180-172, 37-50], #Ж
        [85-70, 186-172, 39-50], #Ж
        [88-70, 190-172, 38-50], #Ж
    ]

    all_y_trues = np.array([
        [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
        [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]
    ])
    
    def __init__(self) -> None:
        self.array = np.array([0, 0, 0])

    def getParams(self, weight, tall, age):
        self.array[0] = int(weight) - 70
        self.array[1] = int(tall) - 172
        self.array[2] = int(age) - 50

    def setRandom(self):
        way = random.choice(self.ways)
        self.array = np.array(way)
    
    def getTrainData(self):
        return [np.array(self.ways), self.all_y_trues]
    
    def putNewTrainData(self, fileName):
        newWays = []
        newTrues = []
        with open(fileName, mode="r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                newWay = []
                for n, param in enumerate(line.split()):
                    if n < 3:
                        newWay.append(int(param))
                    else:
                        newTrues.append([int(param)])
                newWays.append(newWay)
        self.ways = newWays
        self.all_y_trues = newTrues
