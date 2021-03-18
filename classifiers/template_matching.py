import cv2
from ytype import YType

class TemplateMatching:
    def __init__(self, threshold):
        self.minimum_commutative_image_diff = threshold

    def compare_image(self, image_1, image_2):
        if image_1.shape[2] == 3:
            img_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        if image_2.shape[2] == 3:
            img_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
            
        commutative_image_diff = self.get_image_difference(img_gray_1, img_gray_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            return YType.SAME
        return YType.DIFFERENT
    
    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff
    
# if __name__ == '__main__':
#         compare_image = OpencvCompareImage("/home/adity/Desktop/projects/image_search/datasets/Test Data/Client1/['aussie aussome hair conditioner treatment volumizing treatment 250'].jpg", "/home/adity/Desktop/projects/image_search/datasets/Test Data/Client2/['Aussie Aussome Hair Conditioner Treatment Volumizing Treatment 250ml'].jpg")
#         image_difference = compare_image.compare_image()
#         print(image_difference)