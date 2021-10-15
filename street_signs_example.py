from myutils import split_data

if __name__ == "__main__":
    path_to_data = '/home/onwunalu/data/datasets/machine-learning/gtrsb/Train'
    path_to_save_train = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/train'
    path_to_save_val   = '/home/onwunalu/data/datasets/machine-learning/gtrsb/training_data/val'

    split_data(path_to_data, path_to_save_train, path_to_save_val)