
import numpy as np


class FakeImgCandidate(object):
    def __init__(self, img):
        self.img = img
        self.probability = 0

    def set_pixel_value(self, x, y, value):
        self.img[x, y] = value

    def get_pixel_value(self, x, y):
        return self.img[x, y]

    def clip_all_image(self, image_min_values, image_max_values):
        self.img = np.clip(self.img, image_min_values, image_max_values)
        self.img = np.clip(self.img, -1.0, 1.0)
