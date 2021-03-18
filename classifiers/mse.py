import imutils
import cv2
import numpy as np
from skimage.measure import compare_ssim
from ytype import YType

class MSEComparator:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def compare_image(self, imageA, imageB):
        sum_A = np.sum(imageA.astype("float")) / float(imageA.shape[0] * imageA.shape[1])
        sum_B = np.sum(imageB.astype("float")) / float(imageB.shape[0] * imageB.shape[1])
        err= np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        if err < self.threshold:
            return YType.SAME
        return YType.DIFFERENT

    def calculate_ssim(self, imageA, imageB):
        (score, diff) = compare_ssim(imageA, imageB, full=True)
        return score, diff

# tic = time.time()
# (score, diff) = compare_ssim(imageA, imageB, full=True)
# mse = calculateMse(imageA, imageB)
# toc = time.time()
# print("For {} & {}, SSIM: {}, MSE: {}, Execution Time: {}".format("1", "2", score, mse, toc - tic))
