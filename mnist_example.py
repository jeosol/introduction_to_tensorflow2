# import some layers used for computer vision tasks
import tensorflow
import numpy as np
from deeplearning_models import create_sequential_nn_model, create_function_nn_model, create_classbased_nn_model, run_model_with_onehot_labels, run_model_with_sparse_labels

# Run the file as a script
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
    x_test  = np.expand_dims(x_test, axis=-1)

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
