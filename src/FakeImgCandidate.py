class FakeImgCandidate(object):
    def __init__(self, img):
        self.img = img
        self.probability = 0

    def set_pixel_value(self, x, y, value):
        self.img[x, y] = value

    def get_pixel_value(self, x, y):
        return self.img[x, y]