import imutils
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from ytype import YType

class SSIMComparator:
    def __init__(self, threshold):
        self.threshold = threshold
        
    def compare_image(self, imageA, imageB):
        score = structural_similarity(imageA, imageB, full=True, multichannel=True)
        if score[0] > self.threshold:
            return YType.SAME
        return YType.DIFFERENT