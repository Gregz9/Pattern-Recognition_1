import cv2
import os
import copy
import numpy as np
from utils import *

class RectangleExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.original_img = copy.deepcopy(self.img)
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.extracted_areas = []

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (0, 255, 0), 1)

                if self.ix > x:
                    self.ix, x = x, self.ix
                if self.iy > y:
                    self.iy, y = y, self.iy

                extracted_area = self.img[self.iy + 1 : y - 1, self.ix + 1 : x - 1]
                self.extracted_areas.append(extracted_area)
                self.drawing = False
                self.original_img = copy.deepcopy(self.img)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = copy.deepcopy(self.original_img)
                cv2.rectangle(temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
                self.img = temp_img

    def start_extraction(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.draw_rectangle)

        while True:
            cv2.imshow("image", self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # Press 'Esc' to exit
                break

        cv2.destroyAllWindows()
        return self.extracted_areas

if __name__ == "__main__":
    image_path = os.path.dirname(os.path.dirname(__file__)) + "/data/Bilde1.png"
    extractor = RectangleExtractor(image_path)
    extracted_areas = extractor.start_extraction()

    dataset = create_dataset(extracted_areas)
    norm_dataset = normalize_dataset(dataset)
    probs = estimate_pixels_apriori(dataset)
    means = estimate_pixels_mean(dataset)
    covs = estimate_pixels_cov(dataset, means)
    disc1 = class_discriminant(means[0, 1:], covs[0], probs[0])
    # for i, area in enumerate(extracted_areas):
    #     cv2.imshow(f"Area {i}", area)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
