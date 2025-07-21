import tkinter as tk
from tkinter import ttk

from network import NeuralNetwork
from inputController import Input

class App:
    def __init__(self, root):
        self.net = NeuralNetwork()
        # self.net.load("model1.txt")
        self.getter = Input()
        self.root = root
        self.root.title("Определитель пола")
        self.root.geometry("310x500")
        self.root.resizable(False, False)

        # Поля ввода
        self.weight_label = ttk.Label(root, text="Вес:")
        self.weight_label.grid(row=0, column=0, padx=10, pady=10)
        self.weight_entry = ttk.Entry(root)
        self.weight_entry.grid(row=0, column=1, padx=10, pady=10)

        self.height_label = ttk.Label(root, text="Рост:")
        self.height_label.grid(row=1, column=0, padx=10, pady=10)
        self.height_entry = ttk.Entry(root)
        self.height_entry.grid(row=1, column=1, padx=10, pady=10)

        self.age_label = ttk.Label(root, text="Возраст:")
        self.age_label.grid(row=2, column=0, padx=10, pady=10)
        self.age_entry = ttk.Entry(root)
        self.age_entry.grid(row=2, column=1, padx=10, pady=10)

        # Кнопка "Посчитать"
        self.calculate_button = ttk.Button(root, text="Посчитать", command=self.calculate)
        self.calculate_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Окно вывода текста
        self.result_text = tk.Text(root, height=5, width=30)
        self.result_text.grid(row=4, column=0, columnspan=2, pady=10)

        # Кнопка "Тренировать"
        self.train_button = ttk.Button(root, text="Тренировать", command=self.train_model)
        self.train_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Поле для тренировки
        self.training_file_entry = ttk.Entry(root)
        self.training_file_entry.grid(row=6, column=0, padx=10, pady=10)

        # Кнопка "Загрузить модель"
        self.load_model_button = ttk.Button(root, text="Загрузить модель", command=self.load_model)
        self.load_model_button.grid(row=7, column=0, columnspan=2, pady=10)

        # Поле для текстовой модели
        self.model_file_entry = ttk.Entry(root)
        self.model_file_entry.grid(row=8, column=0, padx=10, pady=10)

        # Кнопка "Сохранить модель"
        self.save_model_button = ttk.Button(root, text="Сохранить модель", command=self.save_model)
        self.save_model_button.grid(row=9, column=0, columnspan=2, pady=10)

    def calculate(self):
        self.getter.getParams(self.weight_entry.get(), self.height_entry.get(), self.age_entry.get())
        result = self.net.getResult(self.getter.array)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

    def train_model(self):
        if self.training_file_entry.get() == "":
            train = self.getter.getTrainData()
            print(train)
            self.net.train(train[0], train[1])
        else:
            self.getter.putNewTrainData(self.training_file_entry.get())
            train = self.getter.getTrainData()
            self.net.train(train[0], train[1])

    def load_model(self):
        if self.model_file_entry.get() != "":
            self.net.load(self.model_file_entry.get())

    def save_model(self):
        self.net.getModel("model1.txt")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
