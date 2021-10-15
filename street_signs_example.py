from myutils import split_data, order_test_set

if __name__ == "__main__":
    path_to_data = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Train'
    path_to_save_train = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/train'
    path_to_save_val   = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/val'

    #split_data(path_to_data, path_to_save_train, path_to_save_val)

    path_to_save_test = '/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/'
    path_to_images = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Test/'
    path_to_csv    = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Test.csv'

    order_test_set(path_to_save_test, path_to_images, path_to_csv)