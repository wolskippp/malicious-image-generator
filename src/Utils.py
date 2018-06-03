import os
import csv
from keras.preprocessing import image
import numpy as np
from PIL import Image
import time

from Config import ROOT_PATH


class Utils(object):
    @staticmethod
    def get_path(path, basepath=ROOT_PATH):
        return os.path.join(basepath, path)

    @staticmethod
    def get_test_image_path(imgname, test_images_parent_dir="test_images", extension=".jpg"):
        return os.path.join(test_images_parent_dir, imgname + extension)

    @staticmethod
    def get_new_output_dir():
        outputs_dir_path = Utils.create_child_dir("output")
        new_output_dir_path = Utils.create_child_dir(basepath=outputs_dir_path,
                                                     child_dir=time.strftime("%d%m_%H%M%S"))
        return new_output_dir_path

    @staticmethod
    def create_child_dir(child_dir, basepath=ROOT_PATH):
        child_dir_path = os.path.join(basepath, child_dir)
        if not os.path.exists(child_dir_path):
            os.mkdir(child_dir_path)
        return child_dir_path

    @staticmethod
    def load_classes_csv(classes_csv_path):
        classes_csv = {}
        with open(Utils.get_path(classes_csv_path), newline='') as csvfile:
            classes = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(classes, None)
            for id_name in classes:
                classes_csv[id_name[1]] = int(id_name[0])
        return classes_csv

    @staticmethod
    def prepare_img(img_path):
        img = image.load_img(Utils.get_path(img_path), target_size=(299, 299))
        img_array = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        img_array /= 255.
        img_array -= 0.5
        img_array *= 2.

        return img_array

    @staticmethod
    def save_img(img, filename_sufix=None, img_path=None):
        img /=2
        img += 0.5
        img *= 255
        img_to_save = Image.fromarray(img.astype(np.uint8))

        if img_path == None:
            current_time = time.strftime("%d%m_%H%M%S")
            result_path = Utils.get_path("test_images\output\{}_{}.jpg".format(current_time, filename_sufix))
            img_to_save.save(result_path)
            return result_path
        else:
            img_to_save.save(img_path)
            return img_path

    @staticmethod
    def save_chart(chart, chart_path):
        chart.savefig(chart_path)
        chart.close()