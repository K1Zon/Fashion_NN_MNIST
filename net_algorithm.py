import numpy as np
import utility_file

""" Загружаем данные из файлов датасета, заранее приведя их в нужный нам вид"""
images, labels = utility_file.load_dataset()

""" Веса, сгенерированные рандомно с помощью numpy (в виде матрицы)"""
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784)) # Итого 784 нейрона на входном слое
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20)) # 20 на скрытом слое и 10 на выходном слое
                                                                            # Так как у нас 10 возможных вариантов ответа
""" Нейроны смещения, также сегенерированные с помощью numpy(в виде матрицы) """
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 3 # Количество эпох обучения
e_loss = 0  # Переменные для вычисления ошибки
e_correct = 0
learning_rate = 0.01 # Лучшая скорость обучения

for epoch in range(epochs):     # Обучаем нейросеть конкретное количесто эпох
    print(f"Epoch №{epoch}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Передача данных от входного слоя на скрытый
        # Forward propagation (прямое распространение)
        # Складываем матрицы скрытого слоя и произведение весов на входе с данными изображений
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw)) # Сигмоида - функция активации

        # Forward propagation от скрытого к выходному слою
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        # Вычисляем ошибку в выхлопе нейросети
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Обучение нейросети (алгоритм Backpropagation) на выходном слое -> к скрытому
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Обучение нейросети (алгоритм Backpropagation) на скрытом слое -> к входному
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    # Визуализация процесса обучения, мы видим ошибки и точность выраженные в процентах
    print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
    print(output)
    e_loss = 0
    e_correct = 0

# Сохранение параметров обученной нейросети в один файл
np.savez('neural_network_params.npz',
         weights_input_to_hidden=weights_input_to_hidden,
         weights_hidden_to_output=weights_hidden_to_output,
         bias_input_to_hidden=bias_input_to_hidden,
         bias_hidden_to_output=bias_hidden_to_output)
