import tensorflow
import matplotlib.pyplot as plt

import numpy as np

def display_some_examples(examples, labels):
    plt.figure(figsize=(10,10))
    
    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.imshow(img)

    plt.show()



# This part is not called if this script is imported
if __name__ == '__main__':
    #print(tensorflow.config.list_physical_devices())
    # Load the training and test data set
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    # Print the same of the datasets
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape  = ', x_test.shape)
    print('y_test.shape  = ', y_test.shape)

    display_some_examples(x_train, y_train)