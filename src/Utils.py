import os
import csv
from keras.preprocessing import image
import numpy as np
import time

class Utils(object):
    @staticmethod
    def get_path(path):
        return os.path.join(os.path.dirname(__file__), path)

    @staticmethod
    def get_test_image_path(path, test_images_parent_dir="test_images", extension=".jpg"):
        return os.path.join(test_images_parent_dir, path + extension)

    @staticmethod
    def load_classes_csv(classes_csv_path):
        classes_csv = {}
        with open(classes_csv_path, newline='') as csvfile:
            classes = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(classes, None)
            for id_name in classes:
                classes_csv[id_name[1]] = int(id_name[0])
        return classes_csv

    @staticmethod
    def prepare_img(img_path):
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        img_array /= 255.
        img_array -= 0.5
        img_array *= 2.

        return img_array

    @staticmethod
    def save_img(img_candidate):
        current_time = time.strftime("%m%d_%H%M%S")
        print("TODO, save img")