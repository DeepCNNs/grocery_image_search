# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import Model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import cv2

class GroceryClassificationModel:
    def __init__(self, threshold):
        # load json and create model
        json_file = open('/home/adity/Desktop/projects/image_search/code/model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/home/adity/Desktop/projects/image_search/code/model/groceries_basic_model_dropout.h5")
        print("Loaded model from disk")
        self.model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('dense').output)
        self.threshold = threshold
        
    def compare_image(self, imageA, imageB):
        featureA = extract(self, img=image_1)
        featureB = extract(self, img=image_2)
        dists = np.linalg.norm(featureA-featureB, axis=1)

        if dists[0] < self.threshold:
            return YType.SAME
        return YType.DIFFERENT
        
    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)
        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = np.resize(img,(224, 224, 3))  # VGG must take a 224x224 img as an input
        img = Image.fromarray(img, 'RGB')
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)
        return feature / np.linalg.norm(feature)  # Normalize
        