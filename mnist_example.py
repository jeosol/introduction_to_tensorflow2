import tensorflow
import matplotlib.pyplot as plt
import numpy as np
# import some layers used for computer vision tasks
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

# Illustrate different methods to build neural network model
# 1. use tensorflow.keras.Sequential
# 2. functional approach: function that returns a model
# 3. tensorflow.keras.Model: inherit from base class and override some methods

# Method 1
# Using tensorflow.keras.Sequential model
def create_sequential_nn_model():
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

# Method 2
# function form is more flexible that the sequential form, and it is recommended
def create_function_nn_model():
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

# Method 3
# Inherit from tensorflow.keras.Model
class MyCustomModel(tensorflow.keras.Model):
    def __init__(self):
        super().__init__() # call the __init__ function of the base class

        # Create the layers and name them
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1

        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def create_classbased_nn_model():
    return MyCustomModel()

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

def run_model_with_onehot_labels(model, x_train, y_train, x_test, y_test):
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
    
    # display_some_examples(x_train, y_train)

    # normalize the dataset, need to convert the data to float32 format so we can get fractional numbers
    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255

    # reshape the data to the images have a dimension of (28,28,1) from (28,28)
    # we can use np.exand_dims(x_train, axis=3) or better to use axis=-1, as in expand after last axes
    # this transformation allows the images to be in the right form for the images
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # create the neural network model
    # sequential model
    # model1 = create_sequential_nn_model()
    # model2 = create_sequential_nn_model()

    # functional model
    # model1 = create_function_nn_model()
    # model2 = create_function_nn_model()

    # class-based model
    model1 = create_classbased_nn_model()
    model2 = create_classbased_nn_model()
    
    # run the neural network model 
    modelv1 = run_model_with_sparse_labels(model1, x_train, y_train, x_test, y_test)
    # run the neural network model with one-hot encoded labels
    modelv2 = run_model_with_onehot_labels(model2, x_train, y_train, x_test, y_test)
    
    # both the sparse_labels and onehot_labels generate returns similar results as shown below:
    # the loss and accuracy are similar in both cases and this is expected
    # ----------------------------------
    # Results from the sequential model
    # ----------------------------------
    # results from run_model_with_sparse_labels(...)
    #750/750 [==============================] - 49s 64ms/step - loss: 0.2348 - accuracy: 0.9362 - val_loss: 0.1596 - val_accuracy: 0.9525
    #157/157 [==============================] - 2s 12ms/step - loss: 0.1586 - accuracy: 0.9511

    # results from run_model_with_onehot_labes(...)
    #750/750 [==============================] - 45s 60ms/step - loss: 0.2344 - accuracy: 0.9361 - val_loss: 0.1666 - val_accuracy: 0.9512
    #157/157 [==============================] - 2s 12ms/step - loss: 0.1597 - accuracy: 0.9518

    # ----------------------------------
    # Results from the functional model
    # ----------------------------------
    

    # ----------------------------------
    # Results from using the class-based model
    # ----------------------------------
