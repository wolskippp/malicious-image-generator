import os


class Utils(object):
    @staticmethod
    def get_path(path):
        return os.path.join(os.path.dirname(__file__), path)

    @staticmethod
    def get_test_image(path, test_images_parent_dir="test_images", extension=".jpg"):
        return os.path.join(test_images_parent_dir, path + extension)
