from src.Utils import Utils
from src.Keras import Keras
import random
import numpy as np
from src.Population import Population
from src.FakeImgCandidate import FakeImgCandidate
from src.Result import Result


class FakeImgGenerator(object):
    def __init__(self, img_path, class_name_to_fake, p_max, p_min, fake_class_prob_to_get):
        self.img = Utils.prepare_img(img_path)

        self.p_max = p_max
        self.p_min = p_min
        self.fake_class_prob_to_get = fake_class_prob_to_get
        self.keras = Keras("src/imagenet_classes.csv", self.img)

    def run(self, population_size, pixels_percentage_to_change, max_generations_count, population_percentage_to_keep,
            mutation_prob, crossover_prob):

        result = Result(class_name_to_fake=self.keras.class_name_to_fake,
                        p_max=self.p_max,
                        p_min=self.p_min,
                        fake_class_prob_to_get=self.fake_class_prob_to_get,
                        population_size=population_size,
                        pixels_percentage_to_change=pixels_percentage_to_change,
                        max_generation_count=max_generations_count,
                        population_percentage_to_keep=population_percentage_to_keep,
                        mutation_prob=mutation_prob,
                        crossover_prob=crossover_prob)

        pixels_to_change_count = int(self.img.shape[0] * self.img.shape[1] * pixels_percentage_to_change)
        population = self._init_population(population_size, pixels_to_change_count)

        generations_counter = 0
        population_size_to_keep = int(population_size * population_percentage_to_keep)
		result.start()
        while True:
            if generations_counter > max_generations_count:
                print("Generations count exceeded the limit, "
                      "generated image has probability {}.".format(result.get_max_probability()))
                break

            for i, img_candidate in enumerate(population.fakeImgCandidates):
                # population.fakeImgCandidates[i].probability = random.uniform(0,0.5)
                # population.fakeImgCandidates[i].probability = random.uniform(0,self.fake_class_prob_to_get - 0.1)
                population.fakeImgCandidates[i].probability = self.keras.get_prediction_on_custom_class(
                    img_candidate.img)

            best_img = self._get_best_img_candidate(population.fakeImgCandidates)
            result.add_best_current_probability(best_img.probability)
            result.set_result_img(best_img.img)

            print('The best probablity from {} population : {}'.format(generations_counter, str(best_img.probability)))
            if best_img.probability <= self.fake_class_prob_to_get:
                print("Fake image generation succeeded, "
                      "the generated image has probability {}.".format(result.get_max_probability()))
				self.keras.get_prediction(best_img.img) # po co to?
                break

                #return Utils.save_img(best_img.img, "fake_{}".format(self.keras.class_name_to_fake))

            selected_imgs = self._selection(population.fakeImgCandidates, population_size_to_keep)
            new_population = self._crossover(population, selected_imgs, crossover_prob)
            new_population = self._mutation(new_population, mutation_prob)
            population = new_population
            generations_counter += 1
        result.stop()
        return result

    def _init_population(self, population_size, pixels_to_change_count):

        population = Population()
        width = self.img.shape[0] - 1
        height = self.img.shape[1] - 1
        all_indices = []
        for i in range(0, width):
            for j in range(0, height):
                all_indices.append((i, j))
        random.shuffle(all_indices)
        population.phenotype = all_indices[0:pixels_to_change_count]

        for i in range(population_size):
            fake_img_candidate = FakeImgCandidate(np.copy(self.img))
            for x, y in population.phenotype:
                current_pixel_value = fake_img_candidate.get_pixel_value(x, y)
                new_pixel_value = self._generate_new_pixel_value(current_pixel_value)
                fake_img_candidate.set_pixel_value(x, y, new_pixel_value)

            population.add_img(fake_img_candidate)

        return population

    def _generate_new_pixel_value(self, current_pixel_value):
        def cropp_to_valid_range(current_value):
            if current_value < -1:
                return -1
            elif current_value > 1:
                return 1
            else:
                return current_value

        current_R = current_pixel_value[0]
        current_G = current_pixel_value[1]
        current_B = current_pixel_value[2]

        change_R = random.uniform(self.p_min, self.p_max)
        change_G = random.uniform(self.p_min, self.p_max)
        change_B = random.uniform(self.p_min, self.p_max)

        new_R = cropp_to_valid_range(current_R + change_R)
        new_G = cropp_to_valid_range(current_G + change_G)
        new_B = cropp_to_valid_range(current_B + change_B)

        return np.array([new_R, new_G, new_B])

    def _get_best_img_candidate(self, populationImages):
        sorted_population = sorted(populationImages, key=lambda img_candidate: img_candidate.probability, reverse=False)
        return sorted_population[0]

    def _selection(self, populationImages, population_size_to_keep):
        sorted_population = sorted(populationImages, key=lambda img_candidate: img_candidate.probability, reverse=False)
        return sorted_population[0:population_size_to_keep]

    def _crossover(self, population, selected_imgs, crossover_prob):
        new_population = Population()
        new_population.phenotype = population.phenotype
        range_of_selected_images = range(len(selected_imgs))
        for i in range_of_selected_images:
            selected_img_index_range = list(range_of_selected_images)
            random.shuffle(selected_img_index_range)
            first_index = selected_img_index_range[0]
            second_index = selected_img_index_range[1]
            parent_1 = selected_imgs[first_index]
            parent_2 = selected_imgs[second_index]
            child = FakeImgCandidate(np.copy(self.img))
            for x, y in population.phenotype:
                r = random.random()
                if r <= crossover_prob:
                    new_pixel_value = (parent_1.get_pixel_value(x, y) + parent_2.get_pixel_value(x, y)) / 2
                    child.set_pixel_value(x, y, new_pixel_value)
            new_population.add_img(child)
        for selected_img in selected_imgs:
            new_population.add_img(selected_img)
        return new_population

    def _mutation(self, population, mutation_prob):
        for i, img in enumerate(population.fakeImgCandidates):
            child = FakeImgCandidate(np.copy(self.img))
            r = random.random()
            if r <= mutation_prob:
                for x, y in population.phenotype:
                    current_pixel_value = img.get_pixel_value(x, y)
                    new_pixel_value = self._generate_new_pixel_value(current_pixel_value)
                    child.set_pixel_value(x, y, new_pixel_value)
                population.add_img(child)
        return population
