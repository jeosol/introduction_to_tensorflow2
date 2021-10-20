from myutils import split_data, order_test_set, create_generators
from deeplearning_models import street_signs_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == "__main__":
    """ path_to_data = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Train'
    path_to_save_train = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/train'
    path_to_save_val   = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/val'

    #split_data(path_to_data, path_to_save_train, path_to_save_val)

    path_to_save_test = '/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/'
    path_to_images = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Test/'
    path_to_csv    = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Test.csv'

    order_test_set(path_to_save_test, path_to_images, path_to_csv) """

    path_to_train = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/train'
    path_to_val   = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/val'
    path_to_test  = '/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/'
    batch_size = 64 # this is hyperparameter, can be varied, use smaller batch sizes if we get errors
    epochs = 1
    learning_rate=0.0001 # another hyperparameter

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    # need to save the model during training, so we use callbacks
    path_to_save_model = './models/'

    TRAIN = False
    TEST  = True

    if TRAIN:
        # Callback for saving best models during run
        # Specify parameters for the ModelCheckpoint: use monitor='val_accuracy', mode='max'; may also use monitor='val_loss', mode='min'
        chkpt_save = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True, # save only one of the models, replace best
            save_freq='epoch',
            verbose=1
        )

        # using early stop
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
        )

        # we can pass the generators to a deep learning model  

        model = street_signs_model(nbr_classes)

        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')

        # fit the data
        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[chkpt_save, early_stop], # info for saving model and stopping the training of no improvement at all
        )

    if TEST:
        # Evaluate the dataset using the save model 
        model = tf.keras.models.load_model('./models')
        model.summary()

        print('Evaluating the validation set:')
        model.evaluate(val_generator)

        print('Evaluating the test set:')
        model.evaluate(test_generator)
    
        # Retrain the model for 15 epochs and check that the accuracy for the val data = 0.9967, and for test  0.9681

        # There are techniques to improve the model
        # e.g, hyperparameter tuning: vary the epochs, batch_size, etc
        # e.g., vary NN model architecture: vary the architecture, layers, number of filters etc
        # e.g., data augmentation techniques: in the generators, there are parameters to set for augment data and prevent overfitting
        #       the data augmentation can be done in the preprocessor steps using parameters like rotation_range=10 (rootate images between -10 and 10),
        #       width_shift_range=0.1 (shift image 10% to left or right)
        #       Do not use the synthetic augmentations for the validation and the test set because we want to use the real images for the
        #       the latter
        #       What to do in this case is to use different preprocessors for the training (contains the data augmentation features) and for the
        #       another for the test/validation images (only scaling, no data augmentation)
        #       May also change the image size used as input: smaller or larger
        #
        #       vary parameters to the optimizer