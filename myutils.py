import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import csv

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def display_some_examples(examples, labels, num_images=25):
    plt.figure(figsize=(10,10))
    
    for i in range(num_images):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()


def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:            
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            print(f'copying training data to folder {path_to_folder}')
            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
                
            print(f'copying validation data to folder {path_to_folder}')
            shutil.copy(x, path_to_folder)

def order_test_set(path_to_save_test, path_to_images, path_to_csv):
    if not os.path.isdir(path_to_save_test):
        os.makedirs(path_to_save_test)
    print(path_to_csv)
    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):
                if i == 0:
                    continue
                img_name = row[-1].replace('Test/', '')
                label    = row[-2]
                
                path_to_folder = os.path.join(path_to_save_test, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, f'{label}/{img_name}')
                
                print(f'Copying test data to folder {path_to_folder}')
                shutil.copy(img_full_path, path_to_folder)
    except:
        print('[INFO]: Reading csv file')

# create different generators for the different datasets
def create_generators(batch_size, train_data_path, val_data_path, test_data_path, target_size=(60,60)):

    preprocessor = ImageDataGenerator(
        rescale= 1 / 255
    )
    # all images in a folder, tensorflow knows they are of the same class
    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode='categorical',
        target_size=target_size,
        color_mode='rgb',
        shuffle=True,  # randomness more important here for the training generator
        batch_size=batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode='categorical',
        target_size=target_size,
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )
    
    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode='categorical',
        target_size=target_size,
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator