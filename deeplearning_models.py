import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

# Illustrate different methods to build neural network model and architecture
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
        x = self.batchnorm1(x)

        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def create_classbased_nn_model():
    return MyCustomModel()

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