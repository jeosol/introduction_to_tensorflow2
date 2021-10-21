import tensorflow as tf
import numpy as np
 
# How to use the pretrained and saved model to do standalone prediction of images
def predict_with_model(model, img_path, resize=[60,60], channels=3):
    """Predict the class of the model at img_path using save model."""

    image = tf.io.read_file(img_path) # read image file at path
    image = tf.image.decode_png(image, channels=channels) # decode file as png and specific number of channels = 3
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # also scales the values by 255 to fall between 0 and 1
    image = tf.image.resize(image, resize) # resize the image, shape is 60,60,3
    image = tf.expand_dims(image, axis=0) # shape is (1, 60, 60, 3) 

    prediction = model.predict(image) # return list of probabilities = [0.001, 0.002, 0.99, 0.000, ...]
    prediction = np.argmax(prediction)

    return prediction

# Driver for image prediction
def main():
    # Example images
    img_path = "/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/2/00409.png"
    img_path = "/home/onwunalu/data/datasets/machine-learning/gtrsb/test_data/0/00807.png"

    # Load the save models
    model    = tf.keras.models.load_model('./models')
    # call model to get prediction of model
    if model:
        prediction = predict_with_model(model, img_path)
        print(f'prediction = {prediction}')

if __name__ == '__main__':
    main()
