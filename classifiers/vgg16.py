from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2
from ytype import YType

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self, threshold):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.threshold = threshold

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
    
    def compare_image(self, imageA, imageB):
        fe = FeatureExtractor()
        featureA = fe.extract(img=imageA)
        featureB = fe.extract(img=imageB)
        dists = np.linalg.norm(featureA-featureB, axis=1)  # L2 distances to features
        
        if dists[0] < self.threshold:
            return YType.SAME
        return YType.DIFFERENT
