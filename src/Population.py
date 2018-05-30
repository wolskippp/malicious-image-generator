class Population(object):
    def __init__(self):
        self.fakeImgCandidates = []
        self.phenotype = []

    def add_pixel_coordinates_to_phenotype(self, x, y):
        self.phenotype.append((x, y))

    def add_img(self, fakeImgCandidate):
        self.fakeImgCandidates.append(fakeImgCandidate)