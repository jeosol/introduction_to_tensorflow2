import tensorflow as tf
import numpy as np
 
# How to use the pretrained and saved model to do standalone prediction of images
def predict_with_model(model, img_path):

    image = tf.io.read_file(img_path) # read image file at path
    image = tf.image.decode_png(image, channels=3) # decode file as png and specific number of channels = 3
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # also scales the values by 255 to fall between 0 and 1
    image = tf.image.resize(image, [60,60]) # resize the image, shape is 60,60,3
    image = tf.expand_dims(image, axis=0) # shape is (1, 60, 60, 3) 

    predictions = model.predict(image) # return list of probabilities = [0.001, 0.002, 0.99, 0.000, ...]
    predictions = np.argmax(predictions)

    return predictions

def main():
    img_path = "/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/2/00409.png"
    img_path = "/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/0/00807.png"
    model    = tf.keras.models.load_model('./models')

    prediction = predict_with_model(model, img_path)

    print(f'prediction = {prediction}')

if __name__ == '__main__':
    main()
