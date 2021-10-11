import tensorflow
import matplotlib.pyplot as plt
import numpy as np
# import some layers used for computer vision tasks
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

# methods to build neural network layers
# 1. use tensorflow.keras.Sequential
# 2. functional approach: function that returns a model
# 3. tensorflow.keras.Model: inherit from base class and override some methods

# 1. Using tensorflow.keras.Sequential model
def create_sequential_model():
    model = tensorflow.keras.Sequential(
        [
            Input(shape=(28,28,1)),
            # create 32 filters, each filter of size 3x3 (these are hyperparameters used for now)
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            # MaxPool downsamples the input along its spatial dimensions height, and width, default pool_size = 2,2
            MaxPool2D(),
            BatchNormalization(),

            Conv2D(128, (3, 3), activation='relu'),        
            # MaxPool downsamples the input along its spatial dimensions height, and width, default pool_size = 2,2
            MaxPool2D(),
            # batch normalization activations to help with the gradient descent
            BatchNormalization(),

            GlobalAvgPool2D(),
            Dense(64, activation='relu'),
            # Number units in the output later is 10, because we have 0-9 classes
            # look at the probabilities of the output, that determines the predicted class
            Dense(10, activation='softmax'),
        ]
    )
    return model

# function form is more flexible that the sequential form, and it is recommended
def create_function_model():
    my_input = Input(shape=(28,28,1))
    # create 32 filters, each filter of size 3x3 (these are hyperparameters used for now)
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    # MaxPool downsamples the input along its spatial dimensions height, and width, default pool_size = 2,2
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    # MaxPool downsamples the input along its spatial dimensions height, and width, default pool_size = 2,2
    x = MaxPool2D()(x)
    # batch normalization activations to help with the gradient descent
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    # Number units in the output later is 10, because we have 0-9 classes
    # look at the probabilities of the output, that determines the predicted class
    x = Dense(10, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model

def display_some_examples(examples, labels):
    plt.figure(figsize=(10,10))
    
    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()

def run_model_with_sparse_labels(model, x_train, y_train, x_test, y_test):
    """Run the neural network model with sparse labels."""
    if model:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)

        model.evaluate(x_test, y_test, batch_size=64)

        return model

def run_model_with_onehot_labels(model):
    """Run the neural network model with one-hot encoded labels."""    
    if model:
        # conver the target labels to one-hot encoded equivalents
        y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        y_test  = tensorflow.keras.utils.to_categorical(y_test, 10)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

        model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)

        model.evaluate(x_test, y_test, batch_size=64)

        return model

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

    if False:
        display_some_examples(x_train, y_train)

    # normalize the dataset, need to convert the data to float32 format so we can get fractional numbers
    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255

    # reshape the data to the images have a dimension of (28,28,1) from (28,28)
    # we can use np.exand_dims(x_train, axis=3) or better to use axis=-1, as in expand after last axes
    # this transformation allows the images to be in the right form for the images
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # create the neural network model
    model1 = create_sequential_model()
    model2 = create_sequential_model()
    # run the neural network model 
    modelv1 = run_model_with_sparse_labels(model1, x_train, y_train, x_test, y_test)
    # run the neural network model with one-hot encoded labels
    modelv2 = run_model_with_onehot_labels(model2, x_train, y_train, x_test, y_test)
    