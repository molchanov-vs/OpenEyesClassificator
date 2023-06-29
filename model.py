import numpy as np
from PIL import Image
import tensorflow as tf

class OpenEyesClassificator:
    def __init__(self):
        self.model = tf.keras.models.load_model('./classifier_model.h5')

    def load_image(self, inpIm):
        return np.array(Image.open(inpIm)).reshape(-1, 24, 24, 1) / 255

    def predict(self, inpIm):
        image = self.load_image(inpIm)
        prediction = self.model.predict(image)
        is_open_score = prediction

        return is_open_score[0][0]

