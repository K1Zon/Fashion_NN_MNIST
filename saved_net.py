import numpy as np
import matplotlib.pyplot as plt
import utility_file
from utility_file import image_converter


classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто',
           'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

""" Загружаем данные из файлов датасета, заранее приведя их в нужный нам вид"""
images, labels = utility_file.load_dataset_test()


# Загрузка параметров из файла
loaded_params = np.load('neural_network_params.npz')
weights_input_to_hidden = loaded_params['weights_input_to_hidden']
weights_hidden_to_output = loaded_params['weights_hidden_to_output']
bias_input_to_hidden = loaded_params['bias_input_to_hidden']
bias_hidden_to_output = loaded_params['bias_hidden_to_output']

path = 'file_path.jpg'
test_image = image_converter(path)
image = np.reshape(test_image, (-1, 1))

# Передаем данные об изображении на вход нейросети
# Forward propagation от входного слоя к скрытому слою
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) # Сигмоида - функция активации

# Forward propagation от скрытого к выходному слою
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))


plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"Предполагаемый нейросетью объект: {classes[output.argmax()]}")
plt.show()