import numpy as np
from PIL import Image

def load_dataset():
    image_path = 'datasets+test_photo/train-images-idx3-ubyte'

    with open(image_path, 'rb') as f:
        # Чтение магического числа и количества изображений
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Чтение пиксельных значений изображений
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Конвертация данных изображений в матрицу с размерностью (60000, 784)
        # Таким образом у нас на входном слое будет 784 нейрона
        x_train = image_data.reshape(60000, 784)
        x_train = x_train.astype('float32') / 255


    labels_path = 'datasets+test_photo/train-labels-idx1-ubyte'

    with open(labels_path, 'rb') as f:
        # Чтение магического числа и количества меток
        l_magic_number = int.from_bytes(f.read(4), 'big')
        l_num_images = int.from_bytes(f.read(4), 'big')


        # Чтение и конвертация значений меток в матрицу с размерностью (60000,10), т.к. у нас 10 возможных значений меток
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        y_train = label_data
        #y_train = utils.to_categorical(y_train, 10)
        y_train = np.eye(10)[y_train]

    return x_train, y_train

def load_dataset_test():
    image_path = 'datasets+test_photo/t10k-images-idx3-ubyte'

    with open(image_path, 'rb') as f:
        # Чтение магического числа и количества изображений
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Чтение пиксельных значений изображений
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Конвертация данных изображений в матрицу с размерностью (60000, 784)
        # Таким образом у нас на входном слое будет 784 нейрона
        # x_train = image_data.reshape(num_images, num_rows * num_cols)
        x_train = image_data.reshape(10000, 784)
        x_train = x_train.astype('float32') / 255

    labels_path = 'datasets+test_photo/t10k-labels-idx1-ubyte'

    with open(labels_path, 'rb') as f:
        # Чтение магического числа и количества меток
        l_magic_number = int.from_bytes(f.read(4), 'big')
        l_num_images = int.from_bytes(f.read(4), 'big')

        # Чтение и конвертация значений меток в матрицу с размерностью (60000,10), т.к. у нас 10 возможных значений меток
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        y_train = label_data
        #y_train = utils.to_categorical(y_train, 10)
        y_train = np.eye(10)[y_train]

    return x_train, y_train

def image_converter(path):
    orig_image = Image.open(path)
    resized_image = orig_image.resize((28, 28))
    gray_image = resized_image.convert('L')
    inverted_image = Image.eval(gray_image, lambda x: 255 - x)
    image_array = np.array(inverted_image)
    normalized_image = image_array / 255.0
    flattened_image = normalized_image.flatten()
    return flattened_image